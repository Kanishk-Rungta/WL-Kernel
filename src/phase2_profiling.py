"""Equal-work legacy/optimized feature profiling, independent of SVM search."""
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .dataset import load_dataset
from .feature_extraction import extract
from .phase2_core import structure,operator,history,project


def profile(root, repeats=3):
    root=Path(root); graphs,_,_=load_dataset(root/'data'); rows=[]
    with threadpool_limits(limits=1):
      for repeat in range(repeats):
        # Alternate order to reduce systematic warm-up/order effects.
        results={}
        for engine in (['legacy','optimized'] if repeat%2==0 else ['optimized','legacy']):
            t=perf_counter()
            if engine=='legacy':
                maps=[extract(graphs,H,'degree_adaptive',2.,seed=42) for H in range(4)]
            else:
                ss=[structure(g) for g in graphs]
                ops=[operator(s,'degree_adaptive',2.) for s in ss]
                histories=[history(g,s,op,3) for g,s,op in zip(graphs,ss,ops)]
                maps=[project(histories,H,seed=42) for H in range(4)]
            rows.append(dict(repeat=repeat,engine=engine,seconds=perf_counter()-t,graphs=len(graphs),H='0,1,2,3',tau=2.,cache='none'))
            results[engine]=maps
        for a,b in zip(results['legacy'],results['optimized']): np.testing.assert_allclose(a,b,rtol=1e-13,atol=1e-12)
    output=root/'results/phase2/profiling'; output.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame(rows); frame.to_csv(output/'equal_work_feature_profile.csv',index=False)
    print(frame.to_string(index=False),flush=True)
    return frame


if __name__=='__main__':
    profile(Path(__file__).resolve().parents[1])
