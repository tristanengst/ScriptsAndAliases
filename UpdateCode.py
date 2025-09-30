"""Nicer version of tar -rf path_to_code.tar files to update."""

import argparse
import copy
import glob
import os
import os.path as osp
import subprocess
import sys

import FileFinding
import Utils
import UtilsBase
from UtilsBase import twrite, tqdm

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--substrs", nargs="+",)
    P.add_argument("--files", nargs="+", default=[])
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("--exp_search_dirs", nargs="+", default=FileFinding.exp_search_dirs)
    P.add_argument("-v", "--verbose", action="store_true")
    args = P.parse_args()

    experiment_names = [FileFinding.str_to_exp_folder(s, search_dirs=args.exp_search_dirs, verbose=True, resolve="half_then_user") for s in args.substrs]
    empty = [e for e in experiment_names if not e]
    if empty:
        _ = tqdm.write(f"[INFO] Some experiment substrings did not resolve to anything:\n\t" + "\n\t".join(empty))
    
    experiment_names = [e for e in experiment_names if e]

    if experiment_names:
        exp_name_str = f"====================== EXPERIMENTS TO UPDATE ======================\n\t" + "\n\t".join(experiment_names)
        tqdm.write(exp_name_str)
        tqdm.write("===================================================================")

        proceed = UtilsBase.query_yes_no(f"Proceed with updating these experiments (dry_run={args.dry_run})?\t[y/n]")
        if not proceed:
            _ = tqdm.write("Exiting.")
            sys.exit(0)
    else:
        _ = tqdm.write("[INFO] No experiment names found. Exiting.")
        sys.exit(0)

    

    all_files_orig = copy.deepcopy(args.files)
    all_files = [UtilsBase.strip_left(UtilsBase.strip_right(f, "*"), "*") for f in args.files]
    all_files = [glob.glob(f) for f in args.files]

    for f,f_globbed in zip(all_files_orig, all_files):
        if len(f_globbed) == 0:
            _ = tqdm.write(f"[WARNING] No files matched {f}")

    all_files = UtilsBase.flatten(all_files)
    if not all_files:
        _ = tqdm.write("[INFO] No files matched anything. Exiting.")
        sys.exit(0)

    if args.verbose:
        _ = tqdm.write(f"[INFO] Files to add to tarballs:\n\t" + "\n\t".join(all_files))
        _ = tqdm.write("-------------------------------------------------------")

    for exp_folder in tqdm(experiment_names):
        code_folder_exists = osp.exists(osp.join(exp_folder, "code"))
        code_tar_exists = osp.exists(osp.join(exp_folder, "code.tar"))

        if not code_folder_exists and not code_tar_exists:
            _ = twrite(f"[INFO] exp_folder={exp_folder} has no code.tar or code folder -> skip")
            continue

        commands = []
        if code_folder_exists:
            commands.append(f"rsync -r {' '.join(all_files)} {osp.join(exp_folder, 'code')}/")
        if code_tar_exists:
            commands.append(f"tar -rf {osp.join(exp_folder, 'code.tar')} {' '.join(all_files)}")

        if args.verbose:
            commands_pretty = "\n\t".join(commands)
            _ = tqdm.write(f"[INFO] exp_folder={exp_folder} -> update with commands:\n\t{commands_pretty}")
        
        if args.dry_run:
            _ = tqdm.write("[INFO] Dry run, not executing.")
        else:
            os.system(" ; ".join(commands))