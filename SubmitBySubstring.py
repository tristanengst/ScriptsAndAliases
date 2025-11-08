"""Given a list of identifying substrings, submits the SLURM scripts of the corresponding jobs."""
import argparse
import os
import os.path as osp
import subprocess
import sys

import FileFinding
import Utils
import UtilsBase
from UtilsBase import twrite

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-s", "--substrs", type=str, nargs="+", required=True,
        help="List of substrings to identify jobs to resubmit.")
    P.add_argument("--dry_run", action="store_true",
        help="If set, only prints the commands that would be run, without actually running them.")
    P.add_argument("--verbose", "-v", type=int, default=1,
        help="Verbosity level.")
    P.add_argument("--dependency", type=str, default=None,
        help="If set, adds this dependency to the sbatch command.")
    P.add_argument("--time", type=str, default=None,
        help="If set, adds this time limit to the sbatch command.")
    args = P.parse_args()

    for substr in UtilsBase.tqdm(args.substrs):
        slurm_script = FileFinding.str_to_file(substr,
            slurm_or_result="slurm",
            verbose=False,
            matches=None,
            resolve="half_then_user")

        dependency_str = f"--dependency={args.dependency} " if args.dependency else ""
        time_str = f"--time={args.time} " if args.time else ""
        cmd = f"sbatch {dependency_str} {time_str} {slurm_script}"

        if args.dry_run:
            twrite(f"[DRY RUN] Would run command: {cmd}", verbose=args.verbose)
        else:
            twrite(f"[INFO] Running command: {cmd}", verbose=args.verbose)
            sbatch_output = subprocess.getoutput(cmd)
            twrite(f"[INFO] sbatch output: {sbatch_output}", verbose=args.verbose)