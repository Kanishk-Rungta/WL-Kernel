"""Official TU text loader; categorical labels are deliberately ignored."""
from dataclasses import dataclass
from pathlib import Path
import urllib.request, zipfile, hashlib
import numpy as np
from scipy.sparse import csr_matrix
@dataclass
class Graph:
    x: np.ndarray
    adjacency: csr_matrix
    @property
    def degree(self): return np.asarray(self.adjacency.sum(axis=1)).ravel()
def load_dataset(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=True)
    archive=directory/'ENZYMES.zip'
    if not archive.exists(): urllib.request.urlretrieve('https://www.chrsmrrs.com/graphkerneldatasets/ENZYMES.zip',archive)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (directory/name).resolve().is_relative_to(directory.resolve()): raise ValueError('Unsafe archive')
        z.extractall(directory)
    base=directory/'ENZYMES'
    read=lambda name, **kw: np.loadtxt(base/f'ENZYMES_{name}.txt',delimiter=',',**kw)
    indicators=read('graph_indicator',dtype=int)-1
    x=read('node_attributes'); y=read('graph_labels',dtype=int)
    edges=read('A',dtype=int)-1
    if np.any(indicators[edges[:,0]]!=indicators[edges[:,1]]): raise ValueError('Cross-graph edges')
    adj=csr_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(x),len(x)))
    adj.data[:]=1
    assert (adj!=adj.T).nnz==0
    graphs=[]
    for i in range(len(y)):
        idx=np.flatnonzero(indicators==i); graphs.append(Graph(x[idx],adj[idx][:,idx]))
    assert len(graphs)==600 and x.shape[1]==18 and len(np.unique(y))==6
    stats=dict(graphs=len(y),classes={str(k):int(v) for k,v in zip(*np.unique(y,return_counts=True))},attributes=x.shape[1],mean_nodes=float(np.mean([len(g.x) for g in graphs])),mean_edges=float(np.mean([g.adjacency.nnz/2 for g in graphs])),sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    print(stats,flush=True)
    return graphs,y,stats
