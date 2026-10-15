import numpy as np
from scipy.sparse import csr_matrix
def aggregation_matrix(graph, aggregation='uniform', tau=1.):
    if aggregation not in ('uniform','degree_adaptive'): raise ValueError(aggregation)
    if aggregation=='degree_adaptive' and (np.isnan(tau) or tau<=0): raise ValueError('tau must be positive')
    a=graph.adjacency.tocsr(); degree=graph.degree
    weights=np.zeros(a.nnz)
    for v in range(a.shape[0]):
        start,end=a.indptr[v:v+2]
        if start==end: continue
        score=np.zeros(end-start) if aggregation=='uniform' or np.isinf(tau) else -np.abs(degree[v]-degree[a.indices[start:end]])/tau
        w=np.exp(score-score.max()); weights[start:end]=w/w.sum()
    return csr_matrix((weights,a.indices.copy(),a.indptr.copy()),shape=a.shape)
def refine(graph,H,aggregation='uniform',tau=1.,features='original'):
    if H<0: raise ValueError('H must be nonnegative')
    if features not in ('original','degree_augmented'): raise ValueError(features)
    x=graph.x.copy()
    if features=='degree_augmented': x=np.column_stack([x,np.log1p(graph.degree)])
    a=aggregation_matrix(graph,aggregation,tau); history=[x]
    for _ in range(H):
        new=.5*(x+a@x)
        new[graph.degree==0]=x[graph.degree==0] # self-retention convention for isolates
        x=new; history.append(x)
    return np.concatenate(history,axis=1)
