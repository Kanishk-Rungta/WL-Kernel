import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.dataset import Graph
from src.feature_extraction import extract,directions
from src.swwl_kernel import squared_distances,gaussian_kernel
from src.models import METHODS
@pytest.mark.parametrize('method',list(METHODS))
def test_features_and_kernel(method):
    graphs=[Graph(np.arange(n*2,dtype=float).reshape(n,2),csr_matrix(np.ones((n,n))-np.eye(n))) for n in [1,3,5]]
    aggregation,features=METHODS[method]
    phi=extract(graphs+[graphs[1]],2,aggregation,2.,features,P=4,Q=5)
    assert phi.shape==(4,20); assert np.isfinite(phi).all()
    np.testing.assert_array_equal(phi[1],phi[3])
    K=gaussian_kernel(squared_distances(phi),.01)
    np.testing.assert_allclose(K,K.T); np.testing.assert_allclose(np.diag(K),1)
    assert np.isfinite(K).all() and np.linalg.eigvalsh(K).min()>-1e-10
def test_scaling_and_quantiles():
    g=Graph(np.array([[0.],[2.]]),csr_matrix((2,2)))
    phi=extract([g],0,P=1,Q=3,seed=4)[0]
    sign=directions(1,1,4)[0,0]
    np.testing.assert_allclose(phi,np.sort([0,sign,2*sign])/np.sqrt(3))
