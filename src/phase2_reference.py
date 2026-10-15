"""Optional validation against inspected local official source; no Torch dependency for experiments."""
from pathlib import Path
import hashlib
import importlib.util
import numpy as np
from scipy.sparse import csr_matrix
from .dataset import Graph
from .feature_extraction import directions
from .phase2_core import structure, operator, history, project, distances, kernel
from .utils import write_json


def verify_reference(project_root):
    project_root=Path(project_root)
    source=project_root.parent/'reference_swwl/graph_gp/encoders/sw_encoder.py'
    kernel_source=project_root.parent/'reference_swwl/graph_gp/models/kernel.py'
    dest=project_root/'results/phase2/profiling'; dest.mkdir(parents=True,exist_ok=True)
    if not source.exists() or importlib.util.find_spec('torch') is None:
        result=dict(status='not run',reason='Optional local official source or Torch unavailable')
        write_json(dest/'official_intermediate_check.json',result); return result
    import torch
    def unbatch(x,batch):
        return [x[batch==value] for value in torch.unique(batch,sorted=True)]
    # Execute the actual encoder source, substituting only the unavailable PyG
    # unbatch helper with equivalent splitting for this already sorted fixture.
    code=source.read_text().replace('from torch_geometric.utils import unbatch','')
    namespace={'unbatch':unbatch}; exec(compile(code,str(source),'exec'),namespace)
    g=Graph(np.arange(18.,dtype=np.float64).reshape(6,3),csr_matrix(np.ones((6,6))-np.eye(6)))
    s=structure(g); histories=history(g,s,operator(s),2); E=np.concatenate(histories,axis=1)
    encoder=namespace['SWEncoder'](9,20,20)
    encoder.set_projections(torch.from_numpy(directions(9,20,42)))
    encoder.cum_weights=torch.from_numpy(np.linspace(0,1,20))
    actual=encoder(torch.from_numpy(E),torch.zeros(6,dtype=torch.long)).detach().numpy()[0]
    actual=actual.reshape(20,20).T.ravel() # quantile-major versus projection-major
    ours=project([histories],2,kernel_style='authors')[0]
    np.testing.assert_allclose(actual,ours,rtol=1e-13,atol=1e-13)
    ns={}; exec(compile(kernel_source.read_text(),str(kernel_source),'exec'),ns)
    maps=np.stack([ours,ours*1.2]); d=distances(maps,'authors')
    hparams=dict(precision_embeddings=[.01],variance=1.,nugget=1e-7)
    reference_kernel=ns['custom_kernel_exp'](hparams,[d])
    np.testing.assert_array_equal(reference_kernel,kernel(d,.01,'authors'))
    result=dict(status='passed',torch_version=torch.__version__,quantile_max_absolute_difference=float(np.max(np.abs(actual-ours))),kernel_max_absolute_difference=float(np.max(np.abs(reference_kernel-kernel(d,.01,'authors')))),encoder_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),kernel_source_sha256=hashlib.sha256(kernel_source.read_bytes()).hexdigest(),scope='Actual official SWEncoder and custom_kernel_exp with shared directions, float64 and shared quantile grid; equivalent unbatch helper. Does not validate Torch RNG equivalence, original precision or full official training environment.')
    write_json(dest/'official_intermediate_check.json',result)
    print(result,flush=True); return result


if __name__=='__main__':
    verify_reference(Path(__file__).resolve().parents[1])
