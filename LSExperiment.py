import argparse
import glob
import os
import os.path as osp
import subprocess

import UtilsBase
from UtilsBase import twrite

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-m", "--experiment", required=True, help="Experiment name")
    P.add_argument("--search_dirs", nargs="+", default=[
        osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
        osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
        osp.expanduser("~/scratch/IMLE-SSL/finetunes"),
        osp.expanduser("~/scratch/IMLE-SSL/models_stop")],
        help="Directories to search for experiments in")

    # Arguements to pass to ls command
    P.add_argument("-l", action="store_true", help="Like -l for ls")
    P.add_argument("-t", action="store_true", help="Like -t for ls")
    P.add_argument("-r", action="store_true", help="Like -r for ls")
    P.add_argument("-a", action="store_true", help="Like -a for ls")
    P.add_argument("-d", action="store_true", help="Like -d for ls")
    P.add_argument("-s", action="store_true", help="Like -s for ls")
    args = P.parse_args()

    found_exp_folders = []
    for s in args.search_dirs:
        if not osp.exists(s):
            continue
        experiments = [osp.join(s, f) for f in os.listdir(s) if args.experiment in f]

    if len(experiments) == 0:
        _ = twrite(f"No files found matching {args.experiment} in directories={args.search_dirs}")
    elif len(experiments) == 1:
        experiment = experiments[0]

        ls_args_str = ""
        ls_args_str += " -l" if args.l else ""
        ls_args_str += " -t" if args.t else ""
        ls_args_str += " -r" if args.r else ""
        ls_args_str += " -a" if args.a else ""
        ls_args_str += " -d" if args.d else ""
        ls_args_str += " -s" if args.s else ""

        files = subproces.getoutput(f"ls {ls_args_str} {experiments}", shell=True)
        _ = print(f"Found experiment={osp.join(osp.basename(osp.dirname(experiment)), osp.basename(experiment))}")
        _ = print(files)
    else:
        experiment_list = "\n".join([f"{idx+1}. {osp.join(osp.basename(osp.dirname(exp)), osp.basename(exp))}" for idx,exp in enumerate(experiments)])
        _ = twrite(f"Multiple experiments found matching {args.experiment}. Select one:\n{experiment_list}")

        user_select = input()
        while not user_select.isdigit() or int(user_select) < len(experiments):
            user_select = input(f"Select experiment by number:")
        experiment = experiments[int(user_select) - 1]

        ls_args_str = ""
        ls_args_str += " -l" if args.l else ""
        ls_args_str += " -t" if args.t else ""
        ls_args_str += " -r" if args.r else ""
        ls_args_str += " -a" if args.a else ""
        ls_args_str += " -d" if args.d else ""
        ls_args_str += " -s" if args.s else ""

        files = subproces.getoutput(f"ls {ls_args_str} {experiments}", shell=True)
        _ = print(f"Found experiment={osp.join(osp.basename(osp.dirname(experiment)), osp.basename(experiment))}")
        _ = print(files)