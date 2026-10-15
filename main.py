from pathlib import Path
import argparse
from src.models import METHODS,PHASE1_METHODS,PHASE2_METHODS
from src.experiments import run

def main():
    parser=argparse.ArgumentParser(description='Degree-adaptive SWWL: CPU nested-CV experiments on ENZYMES')
    parser.add_argument('--mode',choices=['smoke','quick','paper'],default='quick')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--methods',nargs='+',choices=list(METHODS))
    parser.add_argument('--phase',type=int,choices=[1,2],default=1)
    parser.add_argument('--kernel-style',choices=['formula','authors'],default='formula')
    parser.add_argument('--cache-mode',choices=['cold','warm'],default='cold')
    parser.add_argument('--run-id',help='Optional Phase 2 output subdirectory; existing runs are preserved')
    parser.add_argument('--download-only',action='store_true')
    parser.add_argument('--figures-only',action='store_true')
    args=parser.parse_args(); root=Path(__file__).resolve().parent
    phase2=args.phase==2 or args.kernel_style=='authors' or any(m.startswith('self_calibrated') for m in (args.methods or []))
    methods=args.methods or (['original','self_calibrated'] if args.kernel_style=='authors' else PHASE2_METHODS if phase2 else PHASE1_METHODS)
    if args.download_only:
        from src.dataset import load_dataset
        load_dataset(root/'data'); return
    if args.figures_only:
        if phase2:
            from src.phase2_experiments import output_directory
            from src.phase2_visualization import figures
            figures(output_directory(root,args.mode,args.kernel_style,args.seed,args.run_id)); return
        from src.visualization import figures
        figures(root/'results' if args.mode=='quick' else root/'results'/args.mode); return
    if phase2:
        from src.phase2_experiments import run_phase2
        run_phase2(args.mode,args.seed,methods,root,args.kernel_style,args.cache_mode,args.run_id)
    else:
        run(args.mode,args.seed,methods,root)
if __name__=='__main__': main()
