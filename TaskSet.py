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

server_gpu2cpu = {
    0: list(range(0, 6)) + list(range(64, 70)),
    1: list(range(6, 12)) + list(range(70, 76)),
    2: list(range(12, 18)) + list(range(76, 82)),
    3: list(range(18, 24)) + list(range(82, 88)),
    4: list(range(24, 32)) + list(range(88, 96)), # Gets extra CPUs: 30, 31, 94, 95
    5: list(range(32, 38)) + list(range(96, 102)),
    6: list(range(38, 44)) + list(range(102, 108)),
    7: list(range(44, 50)) + list(range(108, 114)),
    8: list(range(50, 56)) + list(range(114, 120)),
    9: list(range(56, 64)) + list(range(120, 128))} # Gets extra CPUs: 62, 63, 126, 127
server_gpu2cpu = {gpu: sorted(cpus) for gpu, cpus in server_gpu2cpu.items()}

def get_cpus_from_gpus(*, gpus):
    """Returns the string of CPU indices to feed to taskset for the specified GPUs."""
    host_info = MachineInfo.get_updated_machine_info(os.uname().nodename)
    machine_name = MachineInfo.hostname_to_machine(os.uname().nodename)

    # Servers use a specific map because we enable hyperthreading. Although it can be
    # computed, it's easier to just state it plainly for quick reference
    if machine_name in ["S1", "S2", "S3"]:
        gpu2cpu = server_gpu2cpu
    else:
        cpus_per_gpu =  host_info.total_cpus // host_info.total_gpus
        gpu2cpu = {gpu_idx: list(range(gpu_idx * cpus_per_gpu, (gpu_idx+1) * cpus_per_gpu-1)) for gpu_idx in range(host_info.total_gpus)}

    cpu_range = sorted(set([c for gpu in gpus for c in gpu2cpu[gpu]]))
    cpu_ranges = []
    cur_cpu = cpu_range[0]
    for c in cpu_range:
        if c == cur_cpu + 1:
            cpu_ranges[-1].append(c)
        else:
            cpu_ranges.append([c])
        cur_cpu = c
        
    return ",".join(f"{cr[0]}-{cr[-1]}" if len(cr) > 1 else f"{cr[0]}" for cr in cpu_ranges)

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

def get_random_port(max_port_address=65535, min_port_address=3456):
    """Returns a port between [min_port_address] and [max_port_address]."""
    import random
    return random.randint(min_port_address, max_port_address)

def get_script_from_alias(alias):
    if alias == "python":
        return "python"
    elif alias.startswith("python_ddp") and alias.replace("python_ddp", "").isdigit():
        from importlib.metadata import version as package_version
        if package_version("torch") < "1.9.0":
            gpu_spec = alias.replace("python_ddp", "")
            return f"python -m torch.distributed.launch --nproc_per_node={gpu_spec} --master_port='{get_random_port()}'"
        else:
            gpu_spec = alias.replace("python_ddp", "")
            return f"torchrun --standalone --nnodes=1 --nproc-per-node {gpu_spec}"
    else:
        print(f"Unknown alias {alias}, returning alias={alias}")
        return alias

def try_add_wandb(script_args):
    if script in ["TrainSSL2.py", "EvalFinetune.py"]:
        import subprocess
        script_args_to_get_wandb = argparse.Namespace(**vars(script_args) | dict(gpus=[args.gpus[0]], wandb="online"))
        script_args_to_get_wandb = args_to_unparsed_args(before_script="python", script=script, args=script_args_to_get_wandb)
        script_args_to_get_wandb = " ".join(script_args_to_get_wandb)  # Ensure it's a string
        out = subprocess.getoutput(script_args_to_get_wandb).split()[-1]
        print(f"Experiment name: {out}")
        # return osp.join(experiment_name, "log.txt")
        assert 0
    else:
        return script_args

def get_log_file(script, script_args):
    """Returns the file to which the file should write results to."""
    if script in ["TrainSSL2.py"] and "save_iter" in script_args and int(script_args.save_iter) > 0:
        import subprocess
        script_args_to_get_name = argparse.Namespace(**vars(script_args) | dict(gpus=[args.gpus[0]], print_experiment_name=1))
        script_args_to_get_name = args_to_unparsed_args(before_script="python", script=script, args=script_args_to_get_name)
        script_args_to_get_name = " ".join(script_args_to_get_name)  # Ensure it's a string
        experiment_name = subprocess.getoutput(script_args_to_get_name).split()[-1]
        return osp.join(experiment_name, "log.txt")
    else:
        return None
    
def unparsed_args_to_args(unparsed_args):
    """Returns an argpare Namespace of [unparsed_args]. It requires that the Python
    script in it takes only keyword arguments with one or more values.
    """
    result = dict()
    begin_parsing = False
    cur_key = None
    cur_values = []
    script = None
    before_script = []

    unparsed_args = unparsed_args if isinstance(unparsed_args, list) else unparsed_args.split()

    for idx,a in enumerate(unparsed_args):
        if not begin_parsing and not a.endswith(".py"):
            before_script.append(a)
        elif not begin_parsing and a.endswith(".py"):
            begin_parsing = True
            script = a
        elif begin_parsing and a.startswith("--"):
            if not cur_key is None:
                result[cur_key] = " ".join(cur_values)
            cur_key, cur_values = a[2:], []
        elif begin_parsing and not cur_key is None:
            cur_values.append(a)
            if idx == len(unparsed_args) - 1 and not cur_values is None:
                result[cur_key] = " ".join(cur_values)
        else:
            raise NotImplementedError()

    before_script = " ".join(before_script)
    return before_script, script, argparse.Namespace(**result)

def args_to_unparsed_args(*, args, script=None, before_script=None, sort=True):
    """Returns a list of unparsed arguments from the argparse Namespace [args]."""
    unparsed_args = []
    if before_script is not None:
        unparsed_args.append(before_script)
    if script is not None:
        unparsed_args.append(script)

    args = sorted(vars(args).items(), key=lambda x: x[0]) if sort else vars(args).items()

    for k,v in args:
        unparsed_args.append(f"--{k}")
        unparsed_args.append(" ".join([str(x) for x in v]) if isinstance(v, list) else str(v))
    return unparsed_args

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-c", default="parse_gpus",
        help="CPU specification")
    P.add_argument("--gpus", nargs="*", type=int, default=None,
        help="GPU specification")
    P.add_argument("--shell", default="bash", choices=["bash", "zsh"],
        help="Shell type")
    P.add_argument("--taskset_scripts_dir", default=osp.expanduser("~/.taskset_scripts"),
        help="Directory to store taskset scripts")
    P.add_argument("--taskset_debug", choices=[0, 1], default=0, type=int,
        help="Print the taskset script instead of running it")
    P.add_argument("--time", type=str, default=None,
        help="Has no effect, but can resolve bugs where we accidentally use this.")
    P.add_argument("--strip_gpus", nargs="*", type=int, default=None,
        help="Like 'gpus' but they are removed from the command being run")
    P.add_argument("--try_add_wandb", choices=[0, 1], default=1, type=int,
        help="Tries to add --wandb online if not specified")
    args, unparsed_args = P.parse_known_args()

    if Utils.is_cc():
        print("tpython_ddpX not for use on ComputeCanada.")
        sys.exit(1)

    if args.gpus is None and not args.strip_gpus is None:
        args.gpus = args.strip_gpus
        args.strip_gpus = True
    elif not args.gpus is None and not args.strip_gpus is None:
        raise ValueError("Cannot specify both --gpus and --strip_gpus")
    elif args.gpus is None and args.strip_gpus is None:
        raise ValueError("Must specify either --gpus or --strip_gpus")

    # Get the CPU string for taskset
    args.c = get_cpus_from_gpus(gpus=args.gpus) if args.c == "parse_gpus" else args.c
    taskset_str = f"taskset -c {args.c}"

    before_script, script, script_args = unparsed_args_to_args(unparsed_args=unparsed_args)

    # Map the tpython_ddpX or other prefix to the script to what it should actually be
    before_script = get_script_from_alias(before_script)

    # Add --gpus to [args] unless they were set with --strip_gpus
    if not args.strip_gpus:
        script_args.gpus = [int(g) for g in args.gpus]
    cuda_visible_devices_str = f"CUDA_VISIBLE_DEVICES={','.join([str(g) for g in args.gpus])}"

    # Try and get the experiment name so we can get a log file
    log_file = get_log_file(script, script_args)
    _ = None if log_file is None else os.makedirs(osp.dirname(log_file), exist_ok=True)

    unparsed_args = args_to_unparsed_args(before_script=before_script, script=script, args=script_args)
    unparsed_args = " ".join(unparsed_args)  # Ensure it's a string

    command = f"command=\"{cuda_visible_devices_str} {taskset_str} {unparsed_args}\""
    full_command = f"full_command=\"$command $@ 2>&1 | tee -a '{log_file}'\"" if log_file else f"full_command=\"$command\""

    script_file = osp.join(args.taskset_scripts_dir, f"{uuid.uuid4()}.sh")
    script = f"source {shell2rc[args.shell]}\n{command}\n{full_command}\necho \"Running: $full_command\"\neval \"$full_command\""

    if args.taskset_debug:
        print(script)
    else:
        print(f"=============================================================================")
        print(f"Writing taskset script to {script_file}")
        print(f"Logs write to: stdout " + (f"and {log_file}" if log_file else ""))
        print(f"=============================================================================")

        os.makedirs(osp.dirname(script_file), exist_ok=True)
        with open(script_file, "w+") as f:
            f.write(script)
        os.system(f"taskset -c {args.c} {args.shell} {script_file}")


