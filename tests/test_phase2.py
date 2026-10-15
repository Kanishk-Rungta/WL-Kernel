import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.sparse import csr_matrix
from src.dataset import Graph
from src.wl_refinement import refine, aggregation_matrix
from src.feature_extraction import extract
from src.phase2_core import structure,calibrated_temperature,operator,history,project,distances,kernel,Cache,cache_spec
from src.phase2_experiments import temperatures,validate_cv,get_splits,run_phase2
from src.experiments import configuration
from src.evaluation import splits


def graph():
    return Graph(np.arange(12.).reshape(6,2),csr_matrix([[0,1,1,1,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],[1,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]))


def test_temperature_positive_and_once():
    g=graph(); s=structure(g)
    expected=[1,1,2,0]
    np.testing.assert_equal(np.sort(s.edge_gaps),np.sort(expected))
    assert len(s.edge_gaps)==g.adjacency.nnz//2
    assert calibrated_temperature(g)==2.
    assert np.isfinite(s.temperature) and s.temperature>0


def test_edgeless_and_labels_not_accessed():
    class StructuralOnly:
        adjacency=csr_matrix((4,4))
        @property
        def y(self): raise AssertionError('labels accessed')
        @property
        def x(self): raise AssertionError('attributes accessed during calibration')
    assert calibrated_temperature(StructuralOnly())==1.


def test_relabeling_and_feature_independence():
    g=graph(); p=np.array([4,2,0,5,1,3]); perm=Graph(g.x[p],g.adjacency[p][:,p])
    assert calibrated_temperature(g)==calibrated_temperature(perm)
    assert calibrated_temperature(g)==calibrated_temperature(Graph(g.x*100,g.adjacency))
    s=structure(g); t=structure(perm)
    a=history(g,s,operator(s,'self_calibrated'),3)
    b=history(perm,t,operator(t,'self_calibrated'),3)
    np.testing.assert_allclose(np.concatenate(a,axis=1)[p],np.concatenate(b,axis=1),atol=1e-14)
    np.testing.assert_allclose(project([a],3),project([b],3),atol=1e-14)


def test_operator_weights_and_sparse_reference():
    g=graph(); s=structure(g); a=operator(s,'self_calibrated')
    assert np.all(a.data>=0)
    np.testing.assert_allclose(np.asarray(a.sum(axis=1)).ravel(),[1,1,1,1,0,0])
    assert a[0,1]==a[0,2] # equal neighbor degrees
    assert operator(s,'self_calibrated') is a
    x=g.x.copy(); expected=x.copy()
    for v in range(len(x)):
        neighbors=g.adjacency[v].indices
        if not len(neighbors): continue
        scores=-np.abs(s.degree[v]-s.degree[neighbors])/s.temperature
        w=np.exp(scores-scores.max()); w/=w.sum()
        expected[v]=.5*(x[v]+np.sum(w[:,None]*x[neighbors],axis=0))
    values=history(g,s,a,3)
    np.testing.assert_allclose(values[1],expected)
    np.testing.assert_array_equal(np.concatenate(values,axis=1),np.concatenate(history(g,s,a,3),axis=1))


@pytest.mark.parametrize('tau',[.5,1.,2.,5.,np.inf,np.nextafter(0.,1.)])
def test_adaptive_stability(tau):
    s=structure(graph()); a=operator(s,'degree_adaptive',tau)
    assert np.isfinite(a.data).all()
    if np.isinf(tau): np.testing.assert_allclose(a.toarray(),operator(s).toarray())


@pytest.mark.parametrize('aggregation,tau',[('uniform',None),('degree_adaptive',.5),('degree_adaptive',2.),('degree_adaptive',5.)])
@pytest.mark.parametrize('features',['original','degree_augmented'])
def test_optimized_preserves_legacy(aggregation,tau,features):
    g=graph(); s=structure(g); a=operator(s,aggregation,tau); values=history(g,s,a,3,features)
    for H in range(4):
        old=refine(g,H,aggregation,tau or 1.,features)
        np.testing.assert_allclose(np.concatenate(values[:H+1],axis=1),old,rtol=0,atol=1e-14)
        np.testing.assert_allclose(project([values],H),extract([g],H,aggregation,tau or 1.,features),rtol=0,atol=1e-14)


@pytest.mark.parametrize('style',['formula','authors'])
def test_styles_psd_and_cache(tmp_path,style):
    g=graph(); s=structure(g); values=history(g,s,operator(s),2)
    phi=project([values,[v*2 for v in values]],2,kernel_style=style)
    d=distances(phi,style); K=kernel(d,.01,style)
    np.testing.assert_allclose(K,K.T)
    np.testing.assert_allclose(np.diag(K),1.+(1e-7 if style=='authors' else 0))
    assert np.isfinite(K).all() and np.linalg.eigvalsh(K).min()>-1e-10
    cache=Cache(tmp_path); spec=cache_spec('dataset','original','original',2,None,42,20,20,style)
    cache.save(spec,features=phi,distances=d); saved=cache.load(spec)
    np.testing.assert_array_equal(saved['features'],phi)
    np.testing.assert_array_equal(kernel(saved['distances'],.01,style),K)
    cache.save_operator(spec,0,operator(s)); np.testing.assert_array_equal(cache.load_operator(spec,0).toarray(),operator(s).toarray())
    other=cache_spec('dataset','original','original',2,None,42,20,20,'authors' if style=='formula' else 'formula')
    assert cache.key(spec)!=cache.key(other) and cache.load(other) is None


def test_no_tau_search_and_matching_folds():
    cfg=configuration('paper')
    assert temperatures('self_calibrated',cfg)==[None]
    assert temperatures('self_calibrated_augmented',cfg)==[None]
    assert temperatures('adaptive',cfg)==[.5,1.,2.,5.]
    y=np.repeat(np.arange(6),10); cv=splits(y,5,3,42); validate_cv(cv,len(y))
    bad=[(cv[0][0],cv[0][1],[(cv[0][1],cv[0][1])])]
    with pytest.raises(AssertionError): validate_cv(bad,len(y))


def test_lower_quantile_reference_and_nontrivial_psd():
    from src.feature_extraction import directions
    rng=np.random.default_rng(123)
    histories=[[rng.normal(size=(n,3))] for n in [1,2,4,7,10,15]]
    for style in ['authors','formula']:
        phi=project(histories,0,P=3,Q=4,kernel_style=style)
        K=kernel(distances(phi,style),.4,style)
        assert np.linalg.eigvalsh(K).min()>-1e-10
        if style=='authors':
            projected=histories[-1][0]@directions(3,3,42)
            ranks=np.floor(np.linspace(0,1,4)*(len(projected)-1)).astype(int)
            reference=np.sort(projected,axis=0)[ranks].T.ravel()/np.sqrt(12)
            np.testing.assert_allclose(phi[-1],reference)


def test_cache_identity_fields(tmp_path):
    cache=Cache(tmp_path); spec=cache_spec('ordered-dataset','original','original',2,None,42,20,20,'formula')
    for key,value in [('dataset','reordered'),('method','self_calibrated'),('features','degree_augmented'),('H',3),('temperature',2.),('seed',43),('P',21),('Q',21),('distance','euclidean'),('interpolation','lower'),('version','future-version')]:
        other=dict(spec); other[key]=value
        assert cache.key(spec)!=cache.key(other)


def test_full_smoke_execution(tmp_path,monkeypatch):
    import src.phase2_experiments as runner
    monkeypatch.setattr(runner,'load_dataset',lambda directory: ([graph() for _ in range(48)],np.repeat(np.arange(1,7),8),{'sha256':'synthetic-test-only'}))
    output=run_phase2('smoke',42,['original','adaptive','augmented','self_calibrated','self_calibrated_augmented'],tmp_path)
    import pandas as pd
    result=pd.read_csv(output/'tables/comparison.csv')
    assert len(result)==5 and set(result.svm_fits)=={6}
    assert (output/'figures/accuracy_runtime.png').exists()
    validate_cv(splits(np.repeat(np.arange(6),8),2,2,42),48)
