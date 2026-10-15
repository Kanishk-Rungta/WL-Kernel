"""Figures use only recorded results; Pareto status is descriptive."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS={'original':'Original','adaptive':'Tuned adaptive','augmented':'Augmented',
        'combined':'Combined','self_calibrated':'SCDA','self_calibrated_augmented':'SCDA + augmented'}


def pareto_frontier(accuracy, runtime):
    accuracy=np.asarray(accuracy); runtime=np.asarray(runtime)
    return np.array([not np.any((accuracy>=a)&(runtime<=t)&((accuracy>a)|(runtime<t))) for a,t in zip(accuracy,runtime)])


def figures(output):
    output=Path(output); tab=output/'tables'; dest=output/'figures'; dest.mkdir(parents=True,exist_ok=True)
    s=pd.read_csv(tab/'comparison.csv'); folds=pd.read_csv(tab/'fold_results.csv')
    import json
    style=json.loads((output/'logs/config.json').read_text())['kernel_style']
    names=[LABELS[m] for m in s.method]; x=np.arange(len(s)); colors=plt.get_cmap('Set2')(np.linspace(0,1,len(s)))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':300})
    def save(fig,name):
        if style=='authors': fig.suptitle('Authors-style conventions (matched comparison)',fontsize=11)
        fig.tight_layout(); fig.savefig(dest/f'{name}.png',bbox_inches='tight'); plt.close(fig)
    for metric,label in [('accuracy','Accuracy (%)'),('macro_f1','Macro-F1')]:
        fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(x,s[f'{metric}_mean'],yerr=s[f'{metric}_std'],color=colors,capsize=4)
        ax.set_xticks(x,names,rotation=15); ax.set(ylabel=label,title=f'{label}: matched outer folds'); save(fig,metric)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    axes[0].bar(x,s.total_seconds,color=colors); axes[0].set(ylabel='Wall time (s)',title='Total evaluation including cache IO')
    axes[1].bar(x-.18,s.embedding_seconds,.36,label='Feature extraction'); axes[1].bar(x+.18,s.search_seconds,.36,label='SVM search + refit'); axes[1].set(ylabel='Seconds',title='Feature and model-selection costs'); axes[1].legend()
    for ax in axes: ax.set_xticks(x,names,rotation=20)
    save(fig,'runtime')
    front=pareto_frontier(s.accuracy_mean,s.total_seconds); s.assign(pareto=front).to_csv(tab/'pareto_frontier.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5))
    ax.scatter(s.total_seconds,s.accuracy_mean,c=colors,s=65)
    offsets={'original':(10,-15),'adaptive':(10,12),'augmented':(55,12),
             'self_calibrated':(25,-28),'self_calibrated_augmented':(15,18)}
    for _,r in s.iterrows():
        ax.annotate(LABELS[r.method],(r.total_seconds,r.accuracy_mean),
                    xytext=offsets.get(r.method,(10,12)),textcoords='offset points',
                    arrowprops=dict(arrowstyle='-',color='gray',lw=.7))
    ax.set_xlim(left=max(0,s.total_seconds.min()*.8),right=s.total_seconds.max()*1.24)
    ax.margins(y=.2)
    f=s[front].sort_values('total_seconds'); ax.plot(f.total_seconds,f.accuracy_mean,'k--',alpha=.5,label='Observed Pareto frontier')
    ax.set(xlabel='Total evaluation wall time (s)',ylabel='Mean accuracy (%)',title='Accuracy versus evaluation cost'); ax.legend(); save(fig,'accuracy_runtime')
    tau=pd.read_csv(tab/'graph_temperatures.csv').temperature
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(tau,bins=np.arange(tau.min()-.25,tau.max()+.75,.5),color='#2A9D8F',edgecolor='white')
    ax.set(xlabel='Graph-specific temperature',ylabel='Number of graphs',title='Prespecified calibration: 1 + median edge degree gap'); save(fig,'temperature_distribution')
    gaps=pd.read_csv(tab/'edge_degree_gaps.csv')
    fig,ax=plt.subplots(figsize=(7,4.5)); ax.hist(gaps.degree_gap,bins=np.arange(-.5,gaps.degree_gap.max()+1.5),color='#355C7D',edgecolor='white')
    ax.set(xlabel='Absolute endpoint degree difference',ylabel='Undirected edge count',title='Structural degree gaps (each edge counted once)'); save(fig,'edge_degree_gaps')
    if {'adaptive','self_calibrated'}<=set(s.method):
        data=s.set_index('method'); a=data.loc['adaptive']; b=data.loc['self_calibrated']
        fig,ax=plt.subplots(figsize=(7,4.5)); ratios=[a.search_seconds/b.search_seconds,a.embedding_seconds/b.embedding_seconds,a.total_seconds/b.total_seconds]
        ax.bar(['Model selection','All eligible features','Total wall time'],ratios,color=['#E76F51','#2A9D8F','#355C7D']); ax.axhline(1,color='gray',ls='--')
        ax.set(ylabel='Tuned adaptive time / SCDA time',title='Measured cost reduction; search spaces differ'); save(fig,'speedup')
    if 'self_calibrated' in set(folds.method):
        fig,ax=plt.subplots(figsize=(8,4.5)); calibrated=folds[folds.method=='self_calibrated'].set_index('fold').accuracy
        for ref in ['original','adaptive','augmented']:
            if ref not in set(folds.method): continue
            delta=calibrated-folds[folds.method==ref].set_index('fold').accuracy
            ax.plot(delta.index+1,delta,marker='o',label=f'SCDA minus {LABELS[ref]}')
        ax.axhline(0,color='gray',ls='--'); ax.set(xlabel='Outer fold',ylabel='Accuracy difference (percentage points)',title='Paired held-out accuracy differences'); ax.legend(); save(fig,'paired_differences')
    (dest/'captions.md').write_text('Bars show outer-fold means and sample SD, not standard errors. Total wall time includes cache IO; feature time includes preprocessing, operator construction, WL, projections and quantiles. The observed Pareto frontier consists of methods for which no other method has accuracy at least as high and runtime at most as large, with one strict inequality. It ignores uncertainty and is not a model-selection rule. Speedup compares full eligible feature grids, not equal-work per-graph complexity. Temperature and degree-gap histograms use structures only. Each undirected edge is counted once. Paired differences use matching held-out folds. Authors-style outputs are separate and use the same style for every method in their figures.',encoding='utf-8')
