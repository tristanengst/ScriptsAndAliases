"""Hacky wrapper for using APEX servers until we can put SLURM on them. The basic
idea is to specify specific GPU indices with --gpus like normal, figure out which CPUs
these should use, and then set only the specified GPUs to be visible and replace them
with --gpus starting at zero, using the specified ones.

Should be more portable, but a hacky, horrible thing to undo once possibe.
"""
import argparse
from collections import defaultdict
import os
import os.path as osp
import sys
import uuid

import MachineInfo

shell2rc = dict(zsh="~/.zshrc", bash="~/.bashrc")

def get_gpu2cpu(*, num_cpus, num_gpus, last_gpu_gets_remaining_cpus=True):
    """Returns a GPU index -> (min CPU core index, max CPU core index) map."""
    cpus_per_gpu = num_cpus // num_gpus
    gpu2cpu = {gpu_idx: [gpu_idx * cpus_per_gpu, (gpu_idx+1) * cpus_per_gpu-1] for gpu_idx in range(num_gpus)}

    if last_gpu_gets_remaining_cpus:
        gpu2cpu[num_gpus - 1][1] = num_cpus - 1
    
    return gpu2cpu

def get_cpus_from_gpus(*, gpus):
    """Returns the string of CPU indices to feed to taskset for the specified GPUs."""
    host_info = MachineInfo.get_updated_machine_info(os.uname().nodename)
    gpu2cpu = get_gpu2cpu(num_cpus=host_info.total_cpus, num_gpus=host_info.total_gpus)
    return ",".join([f"{gpu2cpu[gpu][0]}-{gpu2cpu[gpu][1]}" for gpu in gpus])

def inset_arg_into_arg_list(*, arg_list, k, v):
    """Returns [arg_list] with argument --k inserted with values [v] before the first
    keyword argument that is alphabetically after [k].

    Args:
    arg_list    -- list of arguments
    k           -- key to insert (does not start with '--')
    v           -- value to insert (as a list)
    """
    new_arg_list = []
    already_inserted = False
    for a in arg_list:
        if a.startswith(f"--") and a.lstrip("--") > k and not already_inserted:
            new_arg_list.append(f"--{k}")
            new_arg_list += v
            new_arg_list.append(a)
            already_inserted = True
        else:
            new_arg_list.append(a)
    return new_arg_list

def get_script_from_alias(alias):
    if alias == "python":
        return "python"
    elif alias.startswith("python_ddp") and alias.replace("python_ddp", "").isdigit():
        gpu_spec = alias.replace("python_ddp", "")
        return f"torchrun --standalone --nnodes=1 --nproc-per-node {gpu_spec}"
    else:
        print(f"Unknown alias {alias}, returning alias={alias}")
        return alias

P = argparse.ArgumentParser()
P.add_argument("-c", default="parse_gpus",
    help="CPU specification")
P.add_argument("--gpus", nargs="+", type=int, required=True,
    help="GPU specification")
P.add_argument("--shell", default="bash", choices=["bash", "zsh"],
    help="Shell type")
P.add_argument("--taskset_scripts_dir", default=osp.expanduser("~/.taskset_scripts"),
    help="Directory to store taskset scripts")
P.add_argument("--taskset_debug", choices=[0, 1], default=0, type=int,
    help="Print the taskset script instead of running it")
P.add_argument("--time", type=str, default=None,
    help="Time limit for the script passed to timeout command")
P.add_argument("--strip_gpus", choices=[0, 1], default=0, type=int,
    help="Remove --gpus from the command being run")
args, unparsed_args = P.parse_known_args()

args.c = get_cpus_from_gpus(gpus=args.gpus) if args.c == "parse_gpus" else args.c

if not args.strip_gpus:
    unparsed_args = inset_arg_into_arg_list(arg_list=unparsed_args, k="gpus", v=[str(gpu) for gpu in range(len(args.gpus))])

unparsed_args[0] = get_script_from_alias(unparsed_args[0])

# Really useful for submitting things in the early morning and letting them run till
# before someone else could reasonably wake up!
unparsed_args = unparsed_args if args.time is None else (["timeout", args.time] + unparsed_args)
unparsed_args = " ".join(unparsed_args)

script_file = osp.join(args.taskset_scripts_dir, f"{uuid.uuid4()}.sh")
script = f"source {shell2rc[args.shell]}\nCUDA_VISIBLE_DEVICES={','.join([str(g) for g in args.gpus])} taskset -c {args.c} {unparsed_args}\n"

if args.taskset_debug:
    print(script)
else:
    os.makedirs(osp.dirname(script_file), exist_ok=True)
    with open(script_file, "w+") as f:
        f.write(script)
    os.system(f"taskset -c {args.c} {args.shell} {script_file}")
