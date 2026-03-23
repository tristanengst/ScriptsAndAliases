"""Allows viewing job results and/or a SLURM submission script without necessarily
being in the right folder.
"""

import argparse
import os
import os.path as osp
import subprocess
import sys

import Utils
import UtilsBase
from UtilsBase import twrite
import FileFinding

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-r", "--result", action="store_true",
        help="Print job results")
    P.add_argument("-s", "--slurm", action="store_true",
        help="Print job submission script")
    P.add_argument("-e", "--error", action="store_true",
        help="Print job error log")
    P.add_argument("--substr", required=True,
        help="Substring that identifies job")
    P.add_argument("--search_dirs", nargs="+", default="default",
        help="Directories to search in")

    # Ensure that the argument to --substr isn't misinterpreted as a flag even if it
    # has a leading dash
    argv = [(idx,a) for idx,a in enumerate(sys.argv[1:])]
    argv = [UtilsBase.strip_left(a, "-") if argv[idx-1][1] == "--substr" else a for idx,a in argv]
    args = P.parse_args(argv)

    # If not on a cluster, the log file will be stored in the experiment directory
    if not Utils.is_slurm() and not args.slurm:
        twrite(f"Not on cluster -> find output under experiment directory")
        args.search_dirs = FileFinding.exp_search_dirs if args.search_dirs == "default" else args.search_dirs
        exp_name = FileFinding.str_to_exp_folder(args.substr, search_dirs=args.search_dirs, resolve="half_then_user")

        possible_result_exts = [".txt", ".out", ".log", ".json", ".err"]
        non_result_files = ["heartbeat.txt", "config.json", "wandb_attempt.txt"]

        possible_result_files = [osp.join(exp_name, f) for f in os.listdir(exp_name) if any([f.endswith(ext) for ext in possible_result_exts]) and f not in non_result_files]

        if len(possible_result_files) == 0:
            twrite(f"No result files found for {args.substr} in {exp_name}")
            sys.exit(0)
        elif len(possible_result_files) == 1:
            fname = possible_result_files[0]
        else:
            fname = UtilsBase.query_among_list(prompt=f"Multiple possible result files found for {args.substr}, please choose:", options=possible_result_files)
    else:
        # This functionality could be useful but isn't really needed at present
        if args.search_dirs == "default":
            pass
        
        search_dirs = [s for s in args.search_dirs if not s == "default"]
        if args.result:
            file_type = "result"
        elif args.slurm:
            file_type = "slurm"
        elif args.error:
            file_type = "error"
        else:
            raise ValueError("One of --result, --slurm, or --error must be specified")

        fname = FileFinding.str_to_file(args.substr,
            search_dirs=search_dirs,
            file_type=file_type,
            resolve="half_then_user")

    subprocess.run(f"cat {fname}", shell=True)
    print("")
        
        