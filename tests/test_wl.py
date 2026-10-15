import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.dataset import Graph
from src.wl_refinement import refine,aggregation_matrix

def graph(): return Graph(np.array([[0.],[2.],[4.],[7.]]),csr_matrix([[0,1,0,0],[1,0,1,0],[0,1,0,0],[0,0,0,0]]))
def test_hand_refinement():
    np.testing.assert_allclose(refine(graph(),1)[:,1],[1,2,3,7])
def test_weights():
    g=graph(); a=aggregation_matrix(g,'degree_adaptive',.5)
    np.testing.assert_allclose(np.asarray(a.sum(axis=1)).ravel(),[1,1,1,0])
@pytest.mark.parametrize('tau',[0,-1,np.nan,-np.inf])
def test_invalid_temperature(tau):
    with pytest.raises(ValueError): aggregation_matrix(graph(),'degree_adaptive',tau)
def test_stable_temperature():
    assert np.isfinite(aggregation_matrix(graph(),'degree_adaptive',1e-200).data).all()
def test_infinite_temperature():
    np.testing.assert_allclose(refine(graph(),3),refine(graph(),3,'degree_adaptive',np.inf))
def test_isolate():
    x=refine(graph(),3,'degree_adaptive',.5); assert np.isfinite(x).all(); np.testing.assert_equal(x[-1],[7]*4)
def test_nonuniform_weights():
    g=Graph(np.arange(4.)[:,None],csr_matrix([[0,1,1,0],[1,0,0,1],[1,0,0,0],[0,1,0,0]]))
    a=aggregation_matrix(g,'degree_adaptive',.5)
    assert a[0,1]>a[0,2]
