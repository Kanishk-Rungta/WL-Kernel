"""Nested CV: selection by inner accuracy only, deterministic first-candidate ties."""
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,f1_score
def splits(y,outer,inner,seed):
    result=[]
    for fold,(tr,te) in enumerate(StratifiedKFold(outer,shuffle=True,random_state=seed).split(np.zeros(len(y)),y)):
        inside=[(tr[a],tr[b]) for a,b in StratifiedKFold(inner,shuffle=True,random_state=seed+fold+1).split(np.zeros(len(tr)),y[tr])]
        assert not set(tr)&set(te)
        for a,b in inside: assert not set(a)&set(b) and not set(a)&set(te) and not set(b)&set(te)
        result.append((tr,te,inside))
    return result
def fit_score(K,y,tr,te,C):
    model=SVC(C=C,kernel='precomputed',cache_size=128)
    model.fit(K[np.ix_(tr,tr)],y[tr])
    return model.predict(K[np.ix_(te,tr)])
def metrics(y,p):
    return dict(accuracy=100*accuracy_score(y,p),macro_f1=f1_score(y,p,average='macro',zero_division=0),weighted_f1=f1_score(y,p,average='weighted',zero_division=0))
