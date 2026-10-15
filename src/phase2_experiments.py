"""Matched nested evaluation using original splits and unchanged classifier/grid."""
from pathlib import Path
from time import perf_counter
from datetime import datetime
import hashlib
import json
import threading
import numpy as np
import pandas as pd
import psutil
from scipy import stats
from threadpoolctl import threadpool_limits
from .experiments import configuration
from .dataset import load_dataset
from .evaluation import splits, fit_score, metrics
from .models import METHODS
from .utils import write_json, environment
from .phase2_core import (structure, operator, history, project, distances, kernel,
                          Cache, cache_spec, graph_fingerprint, VERSION, STYLES)


def temperatures(method, cfg):
    return cfg['tau'] if METHODS[method][0]=='degree_adaptive' else [None]


def output_directory(root, mode, style, seed, run_id=None):
    base = root/'results/phase2'
    path = base/'authors_style' if style=='authors' else base/mode
    if style=='authors' and mode!='quick':
        path = path/mode
    if seed!=42:
        path = path/f'seed{seed}'
    if run_id:
        if Path(run_id).name!=run_id or run_id in ('.','..'):
            raise ValueError('run-id must be a simple directory name')
        path = path/run_id
    return path


def validate_cv(cv, n):
    tests=[]
    for tr,te,inside in cv:
        assert len(set(tr))==len(tr) and len(set(te))==len(te)
        assert not set(tr)&set(te) and set(tr)|set(te)==set(range(n))
        tests.extend(te)
        inner_valid=[]
        for a,b in inside:
            assert not set(a)&set(b) and set(a)|set(b)==set(tr)
            assert not set(b)&set(te)
            inner_valid.extend(b)
        assert sorted(inner_valid)==sorted(tr)
    assert sorted(tests)==list(range(n))


def get_splits(root, mode, seed, y, dataset):
    cfg=configuration(mode)
    if mode!='smoke' and seed==42:
        source=root/'results' if mode=='quick' else root/'results/paper'
        previous=json.loads((source/'logs/config.json').read_text())
        if previous['dataset']['sha256']!=dataset['sha256']:
            raise ValueError('Dataset archive differs; refusing historical split reuse')
        old=pd.read_csv(source/'tables/oof_predictions.csv')
        for _,part in old.groupby('method'):
            ordered=part.sort_values('index')
            if ordered['index'].tolist()!=list(range(len(y))) or not np.array_equal(ordered.true,y):
                raise ValueError('Dataset ordering/labels differ from Phase 1')
        raw=json.loads((source/'logs/splits.json').read_text())
        cv=[(np.array(f['train']),np.array(f['test']),[(np.array(s['train']),np.array(s['validation'])) for s in f['inner']]) for f in raw]
        generated=splits(y,cfg['outer'],cfg['inner'],seed)
        for saved,new in zip(cv,generated):
            for a,b in zip([saved[0],saved[1]]+[v for pair in saved[2] for v in pair],
                           [new[0],new[1]]+[v for pair in new[2] for v in pair]):
                np.testing.assert_array_equal(a,b)
        provenance=str(source/'logs/splits.json')
    else:
        cv=splits(y,cfg['outer'],cfg['inner'],seed)
        provenance='generated with existing splitter; additional seed or smoke'
    validate_cv(cv,len(y))
    return cv,provenance


def paired_analysis(folds):
    summary=[]; changes=[]
    pairs=[('self_calibrated',reference) for reference in ['original','adaptive','augmented']]
    pairs += [('self_calibrated_augmented','augmented')]
    for method,reference in pairs:
        if method not in set(folds.method) or reference not in set(folds.method): continue
        a=folds[folds.method==method].set_index('fold'); b=folds[folds.method==reference].set_index('fold')
        for metric in ['accuracy','macro_f1','weighted_f1']:
            delta=a[metric]-b[metric]; n=len(delta); sd=delta.std(ddof=1)
            half=stats.t.ppf(.975,n-1)*sd/np.sqrt(n)
            p=float(stats.ttest_rel(a[metric],b[metric]).pvalue) if sd>0 else (1. if delta.mean()==0 else np.nan)
            summary.append(dict(method=method,reference=reference,metric=metric,mean_difference=delta.mean(),std_difference=sd,descriptive_ci_low=delta.mean()-half,descriptive_ci_high=delta.mean()+half,exploratory_t_p=p))
            changes += [dict(method=method,reference=reference,metric=metric,fold=int(f),difference=d) for f,d in delta.items()]
    return pd.DataFrame(summary),pd.DataFrame(changes)


def run_phase2(mode, seed, methods, root, style='formula', cache_mode='cold', run_id=None):
    root=Path(root); output=output_directory(root,mode,style,seed,run_id)
    if (output/'logs/config.json').exists():
        output=output/datetime.now().strftime('rerun_%Y%m%d_%H%M%S')
    for sub in ['tables','figures','logs']: (output/sub).mkdir(parents=True,exist_ok=True)
    cfg=configuration(mode); graphs,y,dataset=load_dataset(root/'data')
    if mode=='smoke':
        selected=np.concatenate([np.flatnonzero(y==c)[:8] for c in np.unique(y)])
        graphs=[graphs[i] for i in selected]; y=y[selected]
    fingerprint=graph_fingerprint(graphs)
    cv,provenance=get_splits(root,mode,seed,y,dataset)
    serial=[dict(train=tr.tolist(),test=te.tolist(),inner=[dict(train=a.tolist(),validation=b.tolist()) for a,b in inside]) for tr,te,inside in cv]
    write_json(output/'logs/splits.json',serial)
    write_json(output/'logs/config.json',dict(phase=2,mode=mode,seed=seed,methods=methods,grid=cfg,kernel_style=style,conventions=STYLES[style],implementation=VERSION,dataset=dataset,graph_order_fingerprint=fingerprint,split_source=provenance,environment=environment(),cache_mode=cache_mode,calibration='1+median(abs(degree(u)-degree(v))) over unique undirected edges; edgeless=1',projections='shared NumPy SeedSequence([seed,dimension,P]); float64',timing='Each method starts cold with independent structures/operators/histories. total_seconds is per-method wall time including cache IO and incremental result writes, excluding shared dataset loading, final summaries/figures. compute_seconds excludes cache IO. Search includes refit/evaluation. RSS sampled at 10ms; allocator retention possible.',exploratory=True))
    cache=Cache(root/'results/phase2/cache')
    rows=[]; validation=[]; predictions=[]; timings=[]; tau_rows=None; edge_rows=None
    with threadpool_limits(limits=1):
      for method in methods:
        print(f'Phase 2 {mode}/{style}: {method}',flush=True)
        start=perf_counter(); process=psutil.Process(); peak=[process.memory_info().rss]; stop=threading.Event()
        def sample():
            while not stop.wait(.01): peak[0]=max(peak[0],process.memory_info().rss)
        monitor=threading.Thread(target=sample,daemon=True); monitor.start()
        stage={k:0. for k in ['preprocessing_seconds','operator_seconds','wl_seconds','projection_quantile_seconds','distance_seconds','kernel_seconds','cache_io_seconds']}
        t=perf_counter(); structures=[structure(g) for g in graphs]; stage['preprocessing_seconds']+=perf_counter()-t
        if tau_rows is None:
            tau_rows=[dict(graph_index=i,temperature=s.temperature,undirected_edges=len(s.edge_gaps)) for i,s in enumerate(structures)]
            edge_rows=[dict(graph_index=i,degree_gap=gap) for i,s in enumerate(structures) for gap in s.edge_gaps]
        aggregation,features=METHODS[method]; ts=temperatures(method,cfg); maps={}; kernels={}; hits=0
        # One operator and one max-H history per graph and eligible temperature.
        for tau in ts:
            spec0=cache_spec(fingerprint,method,features,max(cfg['H']),tau,seed,cfg['P'],cfg['Q'],style)
            t=perf_counter(); ops=[operator(s,aggregation,tau) for s in structures]; stage['operator_seconds']+=perf_counter()-t
            t=perf_counter(); histories=[history(g,s,op,max(cfg['H']),features) for g,s,op in zip(graphs,structures,ops)]; stage['wl_seconds']+=perf_counter()-t
            t=perf_counter()
            for i,op in enumerate(ops): cache.save_operator(spec0,i,op)
            stage['cache_io_seconds']+=perf_counter()-t
            for H in cfg['H']:
                spec=cache_spec(fingerprint,method,features,H,tau,seed,cfg['P'],cfg['Q'],style)
                t=perf_counter(); cached=cache.load(spec) if cache_mode=='warm' else None; stage['cache_io_seconds']+=perf_counter()-t
                if cached is None:
                    t=perf_counter(); phi=project(histories,H,cfg['P'],cfg['Q'],seed,style); stage['projection_quantile_seconds']+=perf_counter()-t
                    t=perf_counter(); d=distances(phi,style); stage['distance_seconds']+=perf_counter()-t
                    t=perf_counter(); cache.save(spec,features=phi,distances=d); stage['cache_io_seconds']+=perf_counter()-t
                else:
                    phi,d=cached['features'],cached['distances']; hits+=1
                maps[(H,tau)]=d
            del histories,ops
        # Preserve Phase 1 H/tau/gamma/C candidate ordering and tie decisions.
        for H in cfg['H']:
          for tau in ts:
            for gamma in cfg['gamma']:
                t=perf_counter(); kernels[(H,tau,gamma)]=kernel(maps[(H,tau)],gamma,style); stage['kernel_seconds']+=perf_counter()-t
        del maps,structures
        search_start=perf_counter(); fits=0
        for fold,(tr,te,inside) in enumerate(cv):
            best_score=-np.inf; best=None
            for (H,tau,gamma),K in kernels.items():
              for C in cfg['C']:
                scores=[np.mean(y[b]==fit_score(K,y,a,b,C)) for a,b in inside]; fits+=len(inside)
                score=float(np.mean(scores))
                validation.append(dict(method=method,fold=fold,H=H,tau=tau,gamma=gamma,C=C,validation_accuracy=100*score))
                if score>best_score: best_score=score; best=(H,tau,gamma,C)
            H,tau,gamma,C=best
            p=fit_score(kernels[(H,tau,gamma)],y,tr,te,C); fits+=1
            rows.append(dict(method=method,fold=fold,H=H,tau=tau,gamma=gamma,C=C,inner_accuracy=100*best_score,**metrics(y[te],p)))
            predictions += [dict(method=method,fold=fold,index=int(i),true=int(a),prediction=int(b)) for i,a,b in zip(te,y[te],p)]
            print(f'  fold {fold+1}/{len(cv)}: {rows[-1]["accuracy"]:.2f}% H={H} tau={tau} gamma={gamma} C={C}',flush=True)
            pd.DataFrame(rows).to_csv(output/'tables/fold_results.csv',index=False)
        search_time=perf_counter()-search_start
        wall=perf_counter()-start; peak[0]=max(peak[0],process.memory_info().rss); stop.set(); monitor.join()
        embedding=sum(stage[k] for k in ['preprocessing_seconds','operator_seconds','wl_seconds','projection_quantile_seconds'])
        timings.append(dict(method=method,**stage,embedding_seconds=embedding,search_seconds=search_time,compute_seconds=sum(v for k,v in stage.items() if k!='cache_io_seconds')+search_time,total_seconds=wall,svm_fits=fits,candidates_per_fold=len(kernels)*len(cfg['C']),peak_rss_mib=peak[0]/2**20,cache_hits=hits))
        del kernels,phi,d
        pd.DataFrame(timings).to_csv(output/'tables/timing.csv',index=False)
    folds=pd.DataFrame(rows); timing=pd.DataFrame(timings)
    summary=folds.groupby('method',sort=False)[['accuracy','macro_f1','weighted_f1']].agg(['mean','std'])
    summary.columns=['_'.join(c) for c in summary.columns]; summary=summary.reset_index().merge(timing,on='method')
    for ref in ['original','adaptive']:
        if ref in methods: summary[f'accuracy_difference_vs_{ref}_pp']=summary.accuracy_mean-float(summary.loc[summary.method==ref,'accuracy_mean'].iloc[0])
    summary.to_csv(output/'tables/comparison.csv',index=False)
    folds[['method','fold','H','tau','gamma','C','inner_accuracy']].to_csv(output/'tables/hyperparameters.csv',index=False)
    pd.DataFrame(validation).to_csv(output/'tables/validation_scores.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'tables/oof_predictions.csv',index=False)
    paired,changes=paired_analysis(folds)
    paired.to_csv(output/'tables/paired_analysis.csv',index=False); changes.to_csv(output/'tables/paired_fold_differences.csv',index=False)
    tau_table=pd.DataFrame(tau_rows); tau_table.to_csv(output/'tables/graph_temperatures.csv',index=False)
    pd.DataFrame(edge_rows).to_csv(output/'tables/edge_degree_gaps.csv',index=False)
    values=tau_table.temperature
    pd.DataFrame([dict(minimum=values.min(),maximum=values.max(),mean=values.mean(),median=values.median(),std=values.std(ddof=1),unique=values.nunique(),proportion_equal_one=np.mean(values==1))]).to_csv(output/'tables/temperature_statistics.csv',index=False)
    tradeoff=[]
    if {'adaptive','self_calibrated'}<=set(methods):
        a=summary.set_index('method').loc['adaptive']; s=summary.set_index('method').loc['self_calibrated']
        tradeoff=[dict(total_speedup=a.total_seconds/s.total_seconds,search_speedup=a.search_seconds/s.search_seconds,feature_speedup=a.embedding_seconds/s.embedding_seconds,search_seconds_saved=a.search_seconds-s.search_seconds,search_percent_saved=100*(1-s.search_seconds/a.search_seconds),accuracy_difference_pp=s.accuracy_mean-a.accuracy_mean)]
        pd.DataFrame(tradeoff).to_csv(output/'tables/tradeoff.csv',index=False)
    # Independent check against existing labels/predictions, never used for tuning.
    if style=='formula' and mode!='smoke' and seed==42:
        olddir=root/'results' if mode=='quick' else root/'results/paper'
        old=pd.read_csv(olddir/'tables/oof_predictions.csv'); new=pd.DataFrame(predictions); checks=[]
        for method in set(methods)&set(old.method):
            a=old[old.method==method].sort_values('index'); b=new[new.method==method].sort_values('index')
            checks.append(dict(method=method,identical_predictions=bool(np.array_equal(a.prediction,b.prediction))))
            if not checks[-1]['identical_predictions']: raise AssertionError(f'Legacy predictions changed: {method}')
        write_json(output/'logs/phase1_regression_check.json',checks)
    from .phase2_visualization import figures
    figures(output)
    print(summary.to_string(index=False),flush=True)
    print(f'Saved: {output}',flush=True)
    return output
