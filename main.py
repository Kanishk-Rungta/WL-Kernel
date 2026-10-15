from pathlib import Path
import argparse
from src.models import METHODS
from src.experiments import run

def main():
    parser=argparse.ArgumentParser(description='Degree-adaptive SWWL: CPU nested-CV experiments on ENZYMES')
    parser.add_argument('--mode',choices=['smoke','quick','paper'],default='quick')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--methods',nargs='+',choices=list(METHODS),default=list(METHODS))
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    if args.download_only:
        from src.dataset import load_dataset
        load_dataset(root/'data'); return
    if args.figures_only:
        from src.visualization import figures
        figures(root/'results' if args.mode=='quick' else root/'results'/args.mode); return
    run(args.mode,args.seed,args.methods,root)
if __name__=='__main__': main()
