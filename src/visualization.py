from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix
LABELS={'original':'Original','adaptive':'Adaptive','augmented':'Augmented','combined':'Combined'}
def figures(output):
    output=Path(output); tab=output/'tables'; dest=output/'figures'
    summary=pd.read_csv(tab/'comparison.csv'); valid=pd.read_csv(tab/'inner_validation.csv'); pred=pd.read_csv(tab/'oof_predictions.csv')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':300})
    names=[LABELS[x] for x in summary.method]; colors=['#355C7D','#2A9D8F','#E9C46A','#E76F51'][:len(names)]
    def save(fig,name): fig.tight_layout(); fig.savefig(dest/f'{name}.png',bbox_inches='tight'); plt.close(fig)
    for metric,title,unit in [('accuracy','Classification accuracy','Accuracy (%)'),('macro_f1','Macro-F1 comparison','Macro-F1')]:
        fig,ax=plt.subplots(figsize=(7,4)); ax.bar(names,summary[f'{metric}_mean'],yerr=summary[f'{metric}_std'],capsize=5,color=colors); ax.set(title=title+' (outer folds)',ylabel=unit); ax.set_ylim(bottom=0); save(fig,metric)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    x=np.arange(len(names)); axes[0].bar(x-.18,summary.embedding_seconds,.36,label='Feature extraction'); axes[0].bar(x+.18,summary.kernel_seconds,.36,label='Distances + Gram matrices'); axes[0].set_xticks(x,names); axes[0].legend(); axes[0].set(ylabel='Seconds',title='Representation construction')
    axes[1].bar(names,summary.total_seconds,color=colors); axes[1].set(ylabel='Seconds',title='Total compute including nested search'); save(fig,'computational_cost')
    for parameter,title in [('tau','Temperature sensitivity'),('H','WL iteration sensitivity')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for method,data in valid.groupby('method',sort=False):
            data=data.dropna(subset=[parameter])
            if data.empty: continue
            # Best remaining hyperparameters at fixed parameter within each outer training set.
            grouped=data.groupby(['fold',parameter]).validation_accuracy.max().groupby(parameter).agg(['mean','std'])
            ax.errorbar(grouped.index,grouped['mean'],yerr=grouped['std'],marker='o',capsize=3,label=LABELS[method])
        ax.set(xlabel='Temperature tau' if parameter=='tau' else 'WL iterations H',ylabel='Inner-validation accuracy (%)',title=title); ax.legend(); save(fig,'temperature' if parameter=='tau' else 'wl_iterations')
    fig,axes=plt.subplots(1,len(summary),figsize=(4*len(summary),4),squeeze=False)
    for ax,method in zip(axes[0],summary.method):
        data=pred[pred.method==method]; ConfusionMatrixDisplay(confusion_matrix(data.true,data.prediction,labels=sorted(data.true.unique())),display_labels=sorted(data.true.unique())).plot(ax=ax,colorbar=False,cmap='Blues',values_format='d'); ax.set_title(LABELS[method])
    fig.suptitle('Out-of-fold confusion matrices (matching splits)'); save(fig,'confusion_matrices')
    (dest/'captions.md').write_text('Accuracy and macro-F1 bars show outer-fold means and sample standard deviations. Runtime uses cold feature extraction and includes the additional adaptive search cost. Sensitivity curves show the best inner-validation accuracy at each fixed parameter, averaged across outer training sets; they are optimistic tuning diagnostics, not held-out estimates. Confusion matrices aggregate each graph exactly once from outer-test predictions. All plots use real recorded data.',encoding='utf-8')
