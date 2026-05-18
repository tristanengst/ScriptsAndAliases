"""Makes switching Python environments less of a hassle, especially when needing to
module load stuff.

USAGE (with aliases):
pythonact ENV_NAME

Setup in UserConfig.py
"""
import argparse
import os
import os.path as osp
import subprocess
import sys

import Utils
import UserConfig
from UtilsBase import twrite

def get_conda_envs(verbose=0):
    """Returns a (conda available, conda_envs) tuple."""
    conda_available = (subprocess.run("conda --version", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode == 0)
    if not conda_available:
        _ = twrite(f"get_conda_envs(): Found conda not avaialble", quiet=not verbose)
        return False, []

    else:
        conda_envs = subprocess.run("conda env list", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        conda_envs = conda_envs.stdout.decode().strip().splitlines()
        conda_envs = [" ".join([l for l in line.split() if l != "*"]) for line in conda_envs]
        conda_envs = conda_envs[2:] if conda_envs[0].startswith("#") else conda_envs
        conda_envs = [line.split()[0] for line in conda_envs if len(line.split()) == 2]

        _ = twrite(f"Found conda environments: {conda_envs}", quiet=not verbose)
        return True, conda_envs

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--env_name", default=None, type=str, help="Name of environment to activate or None to choose")
    P.add_argument("--verbose", action="count", default=0,)
    args = P.parse_args()

    # For now, assume environments are either all-conda or all-pip.
    conda_available, conda_envs = get_conda_envs(verbose=args.verbose)
    env2activate_cmd = dict()

    if conda_available:
        env2activate_cmd |= {env: f"conda activate {env}" for env in conda_envs}
    else:
        env2activate_cmd = dict()
        for env_name, system2activate_cmd in UserConfig.env2system2activate_cmd.items():
            for system_name, activate_cmd in system2activate_cmd.items():
                # Add default first so that we can overwrite it later if there's a
                # system-specific command.
                if system_name == "default":
                    env2activate_cmd[env_name] = activate_cmd
                elif system_name == "cc" and Utils.is_cc():
                    env2activate_cmd[env_name] = activate_cmd
                else:
                    raise NotImplementedError()

    # Check that path-based virtualenvs (ie. source /path/to/env/bin/activate)
    # actually have existing paths.
    for env,cmd in env2activate_cmd.items():
        if cmd.startswith("source "):
            fpath = cmd[len("source "):].split()[0].strip()
            if not osp.exists(fpath):
                _ = twrite(f"[WARNING]: activation command for environment {env} is '{cmd}', but path {fpath} does not exist. Skipping this environment.", quiet=not args.verbose)
                continue
                del env2activate_cmd[env]

    if not args.env_name:
        _ = twrite(f"Available environments: {list(env2activate_cmd.keys())}")
        sys.exit(0)
    elif args.env_name not in env2activate_cmd:
        _ = twrite(f"Environment {args.env_name} not found. Available environments: {list(env2activate_cmd.keys())}")
        sys.exit(1)
    else:
        activate_cmd = env2activate_cmd[args.env_name]
        print(f"EVAL {activate_cmd}")
        sys.exit(0)




    



