"""Python script supporting functionality to list experiment folder contents in a way
similar to the ls command, but without needing to know the full path to the experiment
folder—just an identifying substring will do.

The functionality to find experiment folders comes from FileFinding.py, which in turn
relies on configuration in UserConfig.py.

MOTIVATION AND EXAMPLE USAGE:
Suppose that we have checkpoints stored under ~/scratch/IMLE-SSL/models_imle, and we
know that an experiment we're interested in has 'abc123' as a unique string in its
name, with possibly other parts of the name tacked on at either end.

Now, you could in principle run ls on this experiment folder via:
ls -various_ls_options ~/scratch/IMLE-SSL/models_imle/*abc123*

But obviously this sucks because you have to type the whole path. Try doing this for a
few dozen experiments on a deadline to appreciate just *how much* this sucks.

Solution (if you have the aliases set up):
lse abc123 -various_ls_options

"""

import argparse
import glob
import os
import os.path as osp
import pty
import subprocess
import sys

import FileFinding
import UtilsBase
from UtilsBase import twrite

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="List contents of experiment folders matching a given name or glob pattern, with ls-like options though somewhat different/smaller semantics",
        add_help=False,
        allow_abbrev=False)
    P.add_argument("--experiment", type=str, nargs="*", dest="experiment",
        help="Experiment name, or glob pattern to match multiple experiment folders. Only one level (ie. that containing experiment folders) of the file hierarchy is searched.")

    P.add_argument("experiment", type=str, nargs="*")
    P.add_argument("--search_dirs", nargs="+", default=FileFinding.exp_search_dirs,
        help="Directories to search for experiments in")
    P.add_argument("--debug", action="store_true", help="If set, print debug info")

    # Arguments to pass to ls command. Most aren't supported because parsing arguments
    # in (at least a simple way) with the ability for experiments to start with a dash
    # could be an issue. Extra ls keyword arguments can be added by putting them after
    # '--' in the lse command, eg.
    # lse abc123 -l -h -- -o

    ls_keyword_args = ["a", "d", "l", "r", "R", "s", "S", "t"]
    for ls_arg in ls_keyword_args:
        P.add_argument(f"-{ls_arg}", action="store_true", help=f"Like -{ls_arg} for ls")
    ls_keyword_args += ["h"]
    P.add_argument("-h", action="store_true", help="Like -h for ls (human-readable sizes). Enabled by default if -l is provided")
    
    P.add_argument("--help", action="help", help="Show this help message and exit")
    args, unparsed_args = P.parse_known_args()

    # If -l is provided, -h is enabled by default. Non-human-readable sizes are not
    # very useful.
    args.h = True if args.l else args.h

    ##################################################################################
    # Hacky way to correctly parse ls arguments that come after --
    ##################################################################################
    if "--" in sys.argv:
        # In this case, assume everything after -- is some kind of ls argument if
        # either (a) it's in [unparsed_args] or (b) it's in [args.experiment]
        if not "--experiment" in sys.argv:
            dash_dash_idx = sys.argv.index("--")
            unparsed_args += [e for e in args.experiment if e in sys.argv[dash_dash_idx+1:]]
        args.experiment = [e for e in args.experiment if not e in unparsed_args]
    ##################################################################################
    ##################################################################################
    ##################################################################################

    found_experiments = []
    for experiment in args.experiment:
        # This will return a list of all matches if * is included in --experiment, and
        # will otherwise attempt to resolve to a single experiment folder (string)
        experiment = FileFinding.str_to_exp_folder(experiment,
            resolve="all" if "*" in experiment else "half_then_user",)
        found_experiments += [experiment] if isinstance(experiment, str) else experiment

    all_ls_args = [f"-{k}" for k in ls_keyword_args if vars(args)[k]] + unparsed_args

    ls_args_str = " ".join(all_ls_args)
    ls_exeriments_str = " ".join(found_experiments)
    ls_command = f"ls --color=always {ls_args_str} {ls_exeriments_str}"
    if args.debug:
        _ = twrite(f"Running command: {ls_command}")
    ls_output = subprocess.run(ls_command, shell=True, capture_output=True).stdout.decode("utf-8")
    _ = print(experiment)
    _ = print(ls_output)

    # for experiment in experiments:
    #     ls_command = f"ls --color=always {ls_args_str} {experiment}"
    #     if args.debug:
    #         _ = twrite(f"Running command: {ls_command}")
    #     ls_output = subprocess.run(ls_command, shell=True, capture_output=True).stdout.decode("utf-8")
    #     _ = print(experiment)
    #     _ = print(ls_output)
