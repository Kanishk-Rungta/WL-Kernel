import json,platform,importlib.metadata
from pathlib import Path
def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')
def environment():
    return dict(os=platform.platform(),python=platform.python_version(),libraries={p:importlib.metadata.version(p) for p in ['numpy','scipy','scikit-learn','pandas','matplotlib','psutil','pytest','tqdm']})
