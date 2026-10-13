"""Read-only diagnostics/history and explicit reconstruction queue execution."""
import argparse
import json
import os
import sys
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command',required=True)
    commands.add_parser('doctor',help='Print environment and allocation identifiers; no model is loaded.')
    history = commands.add_parser('history',help='Regenerate plots from an existing training log.')
    history.add_argument('log',type=Path)
    history.add_argument('--output',type=Path)
    history.add_argument('--moving-window',type=int,default=10)
    inspect = commands.add_parser('inspect-queue',help='Inspect plans without running reconstruction.')
    inspect.add_argument('queue',type=Path)
    execute = commands.add_parser('run-queue',help='Reconstruct frozen queued plans in an ALS compute environment.')
    execute.add_argument('queue',type=Path)
    args = parser.parse_args(argv)
    if args.command=='doctor':
        from . import __version__
        print(json.dumps(dict(workflow_version=__version__,python=sys.executable,
            python_version=sys.version,host=os.environ.get('HOSTNAME') or os.environ.get('COMPUTERNAME'),
            job=os.environ.get('SLURM_JOB_ID'),gpus=os.environ.get('CUDA_VISIBLE_DEVICES')),
            indent=2))
        return
    if args.command=='history':
        os.environ['MPLBACKEND'] = 'Agg'
        from .history import plot_training_history
        plot_training_history(args.log,args.output,args.moving_window,show=False)
        return
    from .runtime import configure_reconstruction_threads, require_compute_session
    configure_reconstruction_threads()  # Before ALS/TomoPy/NumExpr imports.
    from .queue import load_queue, inspect_queue, execute_queue
    if args.command=='inspect-queue':
        report = [{k:v for k,v in item.items() if k!='plan'} for item in inspect_queue(load_queue(args.queue))]
    else:
        require_compute_session()
        report = execute_queue(args.queue)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
