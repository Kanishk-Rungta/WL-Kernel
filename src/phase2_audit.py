"""Audit preservation, OOF coverage, nested selection, and summary consistency."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .utils import write_json
from .phase2_experiments import validate_cv


def audit(root):
    root=Path(root); base=root/'results/phase2'
    manifest=json.loads((base/'logs/phase1_manifest.json').read_text())
    checked=0; omitted_caches=0
    for rel,expected in manifest.items():
        path=root/rel
        if not path.exists() and 'cache' in path.parts:
            omitted_caches+=1; continue
        assert path.exists(), f'Missing Phase 1 file: {rel}'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==expected, f'Changed Phase 1 file: {rel}'
        checked+=1
    runs=[]
    for config_path in base.rglob('config.json'):
        output=config_path.parents[1]; cfg=json.loads(config_path.read_text())
        if not (output/'tables/comparison.csv').exists(): continue
        summary=pd.read_csv(output/'tables/comparison.csv'); folds=pd.read_csv(output/'tables/fold_results.csv')
        pred=pd.read_csv(output/'tables/oof_predictions.csv'); valid=pd.read_csv(output/'tables/validation_scores.csv')
        raw=json.loads((output/'logs/splits.json').read_text())
        cv=[(f['train'],f['test'],[(v['train'],v['validation']) for v in f['inner']]) for f in raw]
        count=48 if cfg['mode']=='smoke' else 600; validate_cv(cv,count)
        for method,data in folds.groupby('method'):
            p=pred[pred.method==method]
            assert len(p)==count and p['index'].nunique()==count
            for fold,r in data.set_index('fold').iterrows():
                subset=p[p.fold==fold]
                assert set(subset['index'])==set(cv[fold][1])
                np.testing.assert_allclose(r.accuracy,100*np.mean(subset.true==subset.prediction))
                selected=valid[(valid.method==method)&(valid.fold==fold)]
                np.testing.assert_allclose(r.inner_accuracy,selected.validation_accuracy.max())
                if method.startswith('self_calibrated'): assert selected.tau.isna().all()
            s=summary[summary.method==method].iloc[0]
            for metric in ['accuracy','macro_f1','weighted_f1']:
                np.testing.assert_allclose(s[f'{metric}_mean'],data[metric].mean())
                np.testing.assert_allclose(s[f'{metric}_std'],data[metric].std(ddof=1))
            assert len(data)==cfg['grid']['outer']
        runs.append(dict(directory=str(output.relative_to(root)),methods=summary.method.tolist(),unique_oof_per_method=count,status='passed'))
    result=dict(preserved_phase1_files=checked,omitted_regenerable_phase1_caches=omitted_caches,runs=runs)
    write_json(base/'logs/artifact_audit.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__': audit(Path(__file__).resolve().parents[1])
