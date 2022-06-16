import sys
import subprocess
import time
from tqdm import tqdm
import argparse


if __name__ == "__main__":
    t1_start = time.process_time()
    root_path = "/home/tpinho/IJGIS/Datasets/Australia_Election_2019"
    # val_method = sys.argv[2]
    # dataset_names = [sys.argv[1]]
    # fold_arg = [sys.argv[3]]
    CLI = argparse.ArgumentParser()
    CLI.add_argument(
        "--val_method",  # name on the CLI - drop the `--` for positional/required parameters
        nargs=1,  # 0 or more values expected => creates a list
        type=str,
        default="Optimistic",  # default if nothing is provided
    )
    CLI.add_argument(
        "--dataset_names",  # name on the CLI - drop the `--` for positional/required parameters
        nargs=1,  # 0 or more values expected => creates a list
        type=str,
        default="Original",  # default if nothing is provided
    )
    CLI.add_argument(
        "--folds",
        nargs="*",
        type=int,  # any type/callable can be used here
        default=[],
    )
    CLI.add_argument(
        "--fs_method",  # name on the CLI - drop the `--` for positional/required parameters
        nargs="*",  # 0 or more values expected => creates a list
        type=int,
        default="CFS",  # default if nothing is provided
    )

    index_col = "INDEX"
    target_col = "TARGET"
    fold_col = "INDEX_FOLDS"
    procs = []
    args = CLI.parse_args()
    print(args.dataset_names)

    for dataset_name in args.dataset_names:
        for fold in args.folds:
            for context in args.list_contexts:
                if context != int(fold):
                    cmd = f'python optimistic.py {root_path} "{dataset_name}" {args.fs_method[0]} {index_col} {fold_col} {target_col} {args.val_method[0]}'
                    cmd = f"python fs_cfs_local.py {args.val_method[0]} {root_path} {dataset_name} {fold} {index_col} {target_col} {fold_col} {context}"
                    procs.append(subprocess.Popen(cmd, shell=True))
    exit_codes = [p.wait() for p in procs]
    t1_stop = time.process_time()
    print(f"time -- {(t1_start-t1_stop)/60}")
