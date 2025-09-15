import argparse
import glob
import os
import os.path as osp
import pty
import subprocess

import FileFinding
import UtilsBase
from UtilsBase import twrite

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-m", "--experiment", required=True, help="Experiment name")
    P.add_argument("--search_dirs", nargs="+", default=FileFinding.exp_search_dirs,
        help="Directories to search for experiments in")
    P.add_argument("--debug", action="store_true", help="If set, print debug info")

    # Arguements to pass to ls command
    P.add_argument("-l", action="store_true", help="Like -l for ls")
    P.add_argument("-t", action="store_true", help="Like -t for ls")
    P.add_argument("-r", action="store_true", help="Like -r for ls")
    P.add_argument("-a", action="store_true", help="Like -a for ls")
    P.add_argument("-d", action="store_true", help="Like -d for ls")
    P.add_argument("-s", action="store_true", help="Like -s for ls")
    args = P.parse_args()

    experiment = FileFinding.str_to_exp_folder(args.experiment, resolve="half_then_user")

    ls_args_str = ""
    ls_args_str += " -l" if args.l else ""
    ls_args_str += " -t" if args.t else ""
    ls_args_str += " -r" if args.r else ""
    ls_args_str += " -a" if args.a else ""
    ls_args_str += " -d" if args.d else ""
    ls_args_str += " -s" if args.s else ""

    ls_command = f"ls --color=always {ls_args_str} {experiment}"
    if args.debug:
        _ = twrite(f"Running command: {ls_command}")
    ls_output = subprocess.run(ls_command, shell=True, capture_output=True).stdout.decode("utf-8")
    _ = print(experiment)
    _ = print(ls_output)
