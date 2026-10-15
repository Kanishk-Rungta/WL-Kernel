import numpy as np
from .wl_refinement import refine
def directions(d,P,seed):
    r=np.random.default_rng(np.random.SeedSequence([seed,d,P])).normal(size=(d,P))
    return r/np.linalg.norm(r,axis=0)
def extract(graphs,H,aggregation='uniform',tau=1.,features='original',P=20,Q=20,seed=42):
    if P<1 or Q<2: raise ValueError('P>=1 and Q>=2 required')
    d=(graphs[0].x.shape[1]+(features=='degree_augmented'))*(H+1)
    projection=directions(d,P,seed)
    return np.stack([np.quantile(refine(g,H,aggregation,tau,features)@projection,np.linspace(0,1,Q),axis=0,method='linear').T.ravel()/np.sqrt(P*Q) for g in graphs])
