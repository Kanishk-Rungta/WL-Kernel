from pathlib import Path
from time import perf_counter
import itertools, json, threading
import numpy as np
import pandas as pd
import psutil
from tqdm import tqdm
from scipy import stats as scipy_stats
from threadpoolctl import threadpool_limits
from .dataset import load_dataset
from .models import METHODS
from .feature_extraction import extract
from .swwl_kernel import squared_distances,gaussian_kernel
from .evaluation import splits,fit_score,metrics
from .utils import write_json,environment
from .visualization import figures

def configuration(mode):
    if mode=='paper': return dict(H=[0,1,2,3],gamma=[1e-4,1e-3,1e-2,1e-1],C=[10.**i for i in range(-3,4)],tau=[.5,1.,2.,5.],outer=10,inner=5,P=20,Q=20)
    if mode=='quick': return dict(H=[1,2,3],gamma=[1e-3,1e-2,1e-1],C=[.1,1.,10.],tau=[.5,2.,5.],outer=5,inner=3,P=20,Q=20)
    return dict(H=[1],gamma=[.01],C=[1.],tau=[2.],outer=2,inner=2,P=20,Q=20)

def run(mode,seed,methods,root):
    cfg=configuration(mode); graphs,y,dataset=load_dataset(root/'data')
    if mode=='smoke':
        idx=np.concatenate([np.flatnonzero(y==c)[:8] for c in np.unique(y)])
        graphs=[graphs[i] for i in idx]; y=y[idx]
    output=root/'results' if mode=='quick' else root/'results'/mode
    for sub in ['tables','figures','logs','cache']: (output/sub).mkdir(parents=True,exist_ok=True)
    cv=splits(y,cfg['outer'],cfg['inner'],seed)
    write_json(output/'logs/splits.json',[dict(train=tr.tolist(),test=te.tolist(),inner=[dict(train=a.tolist(),validation=b.tolist()) for a,b in inside]) for tr,te,inside in cv])
    write_json(output/'logs/config.json',dict(mode=mode,seed=seed,methods=methods,grid=cfg,dataset=dataset,environment=environment(),quantile='linear',projection_rng='numpy default_rng SeedSequence([seed,dimension,P])',scaling='none; raw log1p(degree) appended',selection='mean inner accuracy; deterministic first tie',timing='cold feature computation, distances and all gamma kernels once per method; CV includes search and outer refit; cache IO excluded',memory='sampled absolute process RSS every 10ms; includes loaded data and interpreter; not allocation attribution',baseline='independent paper-formula reproduction'))
    rows=[]; validation=[]; timings=[]; predictions=[]
    with threadpool_limits(limits=1):
      for method in methods:
        print(f'Running {method}',flush=True)
        process=psutil.Process(); peak=[process.memory_info().rss]; stop=threading.Event()
        def sample():
            while not stop.wait(.01): peak[0]=max(peak[0],process.memory_info().rss)
        monitor=threading.Thread(target=sample,daemon=True); monitor.start()
        aggregation,feature_mode=METHODS[method]; taus=cfg['tau'] if aggregation=='degree_adaptive' else [None]
        kernels={}; embedding_time=0.; kernel_time=0.
        for H,tau in itertools.product(cfg['H'],taus):
            t=perf_counter(); phi=extract(graphs,H,aggregation,tau if tau is not None else 1.,feature_mode,cfg['P'],cfg['Q'],seed); embedding_time+=perf_counter()-t
            t=perf_counter(); d=squared_distances(phi)
            for gamma in cfg['gamma']: kernels[(H,tau,gamma)]=gaussian_kernel(d,gamma)
            kernel_time+=perf_counter()-t
            # These transformations depend only on each graph and fixed randomness; safe across folds.
            np.savez_compressed(output/'cache'/f'{method}_H{H}_tau{tau}_seed{seed}_{mode}_P{cfg["P"]}_Q{cfg["Q"]}_linear.npz',features=phi,distances=d)
        search_start=perf_counter(); fits=0
        for fold,(tr,te,inside) in enumerate(tqdm(cv,desc=method)):
            best_score=-np.inf; best=None
            for (H,tau,gamma),K in kernels.items():
              for C in cfg['C']:
                scores=[accuracy_score_local(y[b],fit_score(K,y,a,b,C)) for a,b in inside]; fits+=len(inside)
                score=float(np.mean(scores))
                validation.append(dict(method=method,fold=fold,H=H,tau=tau,gamma=gamma,C=C,validation_accuracy=100*score))
                if score>best_score: best_score=score; best=(H,tau,gamma,C)
            H,tau,gamma,C=best
            p=fit_score(kernels[(H,tau,gamma)],y,tr,te,C); fits+=1
            row=dict(method=method,fold=fold,H=H,tau=tau,gamma=gamma,C=C,inner_accuracy=100*best_score,**metrics(y[te],p)); rows.append(row)
            for i,true,pred in zip(te,y[te],p): predictions.append(dict(method=method,fold=fold,index=int(i),true=int(true),prediction=int(pred)))
            print(f'  fold {fold+1}: accuracy={row["accuracy"]:.2f}%, H={H}, tau={tau}, gamma={gamma}, C={C}',flush=True)
            pd.DataFrame(rows).to_csv(output/'tables/fold_results.csv',index=False)
        search_time=perf_counter()-search_start
        peak[0]=max(peak[0],process.memory_info().rss); stop.set(); monitor.join()
        timings.append(dict(method=method,embedding_seconds=embedding_time,kernel_seconds=kernel_time,search_seconds=search_time,total_seconds=embedding_time+kernel_time+search_time,peak_rss_mib=peak[0]/2**20,svm_fits=fits,candidates_per_fold=len(kernels)*len(cfg['C'])))
        del kernels
    folds=pd.DataFrame(rows); valid=pd.DataFrame(validation); timing=pd.DataFrame(timings); pred=pd.DataFrame(predictions)
    summary=folds.groupby('method',sort=False)[['accuracy','macro_f1','weighted_f1']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index().merge(timing,on='method')
    if 'original' in methods:
        base=float(summary.loc[summary.method=='original','accuracy_mean'].iloc[0]); summary['accuracy_difference_pp']=summary.accuracy_mean-base; summary['relative_improvement_percent']=100*(summary.accuracy_mean-base)/base
    summary.to_csv(output/'tables/comparison.csv',index=False)
    folds[['method','fold','H','tau','gamma','C','inner_accuracy']].to_csv(output/'tables/hyperparameters.csv',index=False)
    timing.to_csv(output/'tables/timing.csv',index=False); valid.to_csv(output/'tables/inner_validation.csv',index=False); pred.to_csv(output/'tables/oof_predictions.csv',index=False)
    paired=[]; paired_folds=[]
    if 'original' in methods:
      baseline=folds[folds.method=='original'].set_index('fold')
      for method in methods:
        if method=='original': continue
        other=folds[folds.method==method].set_index('fold')
        for metric in ['accuracy','macro_f1','weighted_f1']:
            delta=other[metric]-baseline[metric]
            for fold,difference in delta.items(): paired_folds.append(dict(method=method,fold=fold,metric=metric,difference=difference))
            n=len(delta); mean=delta.mean(); sem=delta.std(ddof=1)/np.sqrt(n); half=scipy_stats.t.ppf(.975,n-1)*sem
            paired.append(dict(method=method,metric=metric,mean_difference=mean,std_difference=delta.std(ddof=1),descriptive_ci_low=mean-half,descriptive_ci_high=mean+half,paired_t_p=float(scipy_stats.ttest_rel(other[metric],baseline[metric]).pvalue) if delta.std()>0 else None))
    pd.DataFrame(paired_folds).to_csv(output/'tables/paired_fold_differences.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'tables/paired_analysis.csv',index=False)
    figures(output)
    from .report import report
    report(root,output,summary,folds,timing,mode,paired)
    print(summary.to_string(index=False),flush=True)
    return summary

def accuracy_score_local(y,p): return np.mean(y==p)

