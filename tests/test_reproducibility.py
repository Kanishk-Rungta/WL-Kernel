import numpy as np
from scipy.sparse import csr_matrix
from src.dataset import Graph
from src.feature_extraction import extract,directions
from src.evaluation import splits,fit_score

def test_reproducible():
    g=Graph(np.arange(12.).reshape(4,3),csr_matrix(np.ones((4,4))-np.eye(4)))
    np.testing.assert_array_equal(extract([g],2,seed=42),extract([g],2,seed=42))
    assert not np.array_equal(directions(3,20,42),directions(3,20,43))
def test_nested_splits():
    y=np.repeat(np.arange(6),10); a=splits(y,5,3,42); b=splits(y,5,3,42)
    tests=[]
    for (tr,te,inside),(tr2,te2,inside2) in zip(a,b):
        np.testing.assert_array_equal(tr,tr2); np.testing.assert_array_equal(te,te2)
        assert not set(tr)&set(te); tests.extend(te)
        for u,v in inside: assert not set(u)&set(v) and set(u)|set(v)==set(tr)
    assert sorted(tests)==list(range(len(y)))
def test_classifier_reproducible():
    y=np.array([1,1,2,2,1,2]); K=np.exp(-.1*(np.arange(6)[:,None]-np.arange(6)[None,:])**2)
    np.testing.assert_array_equal(fit_score(K,y,np.arange(4),np.arange(4,6),1),fit_score(K,y,np.arange(4),np.arange(4,6),1))
