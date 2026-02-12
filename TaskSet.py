"""Hacky wrapper for using APEX servers until we can put SLURM on them. The basic
idea is to specify specific GPU indices with --gpus like normal, figure out which CPUs
these should use, and then set only the specified GPUs to be visible and replace them
with --gpus starting at zero, using the specified ones.

Should be more portable, but a hacky, horrible thing to undo once possibe.
"""
import argparse
from collections import defaultdict
import json
import os
import os.path as osp
import subprocess
import sys
import uuid

import MachineInfo
import Utils
import UtilsBase
from UtilsBase import twrite


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

def insert_arg_into_arg_list(*, arg_list, k, v):
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
    if script in ["TrainSSL2.py", "TrainNorMAE.py", "EvalLinear.py", "EvalFinetune.py"] and not "wandb" in script_args:
        script_args_to_get_wandb = argparse.Namespace(**vars(script_args) | dict(gpus=[args.gpus[0]], wandb="online"))
        script_args_to_get_wandb = args_to_unparsed_args(before_script="python", script=script, args=script_args_to_get_wandb)
        script_args_to_get_wandb = " ".join(script_args_to_get_wandb)  # Ensure it's a string
        out = subprocess.getoutput(script_args_to_get_wandb).split()[-1]
        print(f"Experiment name: {out}")
        # return osp.join(experiment_name, "log.txt")
        assert 0
    else:
        return script_args

def get_experiment_metadata(*, script, script_args):
    """Returns an argparse Namespace of metadata for the experiment that will be run.
    This can be used to configure it properly.
    """
    if script in ["TrainSSL2.py", "TrainNorMAE.py", "EvalLinear.py", "EvalFinetune.py"]:
        script_args_to_get_name = argparse.Namespace(**vars(script_args) | dict(gpus=[args.gpus[0]], write_metas=1))
        script_args_to_get_name = args_to_unparsed_args(before_script="python", script=script, args=script_args_to_get_name)
        script_args_to_get_name = " ".join(script_args_to_get_name)  # Ensure it's a string
        output = subprocess.getoutput(script_args_to_get_name)
        
        try:
            return argparse.Namespace(**UtilsBase.load_meta(output))
        except Exception as e:
            twrite(f"[ERROR] failed to load meta. Used:\nscript_args_to_get_name={script_args_to_get_name}\noutput={output}")
            raise e
    else:
        twrite(f"No metadata for script {script}")
        return argparse.Namespace()
    
def unparsed_args_to_args(unparsed_args):
    """Returns a (before_script, script, args) triple from the list of string of
    unparsed argument [unparsed_args], where:
    - [before_script] is a string of command line arguments prior the script being run
    - [script] is the Python script being run
    - [args] is a Namespace containing arguments to the script, parsed heuristically
    
    Note: the argparse arguments must be all keyword arguments!
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


def get_new_directory_strs(*, exp_name, args, script_args):
    """Returns (start_command, end_command) tuple. The first sets up and changes to a
    temporary directory unique to an experiment being run. The second removes the
    directory.

    This is basically a SLURM_TMPDIR. However, we won't actually set that environment
    variable as that could confuse things. Instead, we will (literally) clone the
    current working directory. If there exists saved state for the experiment, we will
    then load from it into the current working directory, replacing things.
    
    Args:
    exp_name    -- abspath to name of the experiment, used to create a new directory
    args        -- taskset argparse Namespace
    script_args -- argparse Namespace of the script being run, used to determine what
    """
    def get_rel_root(path):
        rels = []
        for r in osp.relpath(path).split(osp.sep):
            rels.append(r)
            if not r in [".", ".."]:
                break
        return osp.sep.join(rels)

    def get_symlink_to_rel_root(*, temp_dir, path):
        """Returns a command to create a symlink to the top-level directory in [path]
        as seen in the current working directory. If the file is itself a symlink,
        then symlink to where it actually points to.
        """
        if path.startswith("~") or osp.isabs(path):
            return f"# no need to symlink {path}"
        else:
            relpath = osp.relpath(get_rel_root(path))
            abspath = osp.abspath(osp.realpath(relpath))
            return f"ln -s {abspath} {osp.join(temp_dir, relpath)}"
        
    if not args.new_dir:
        return "", ""

    temp_dir = osp.join(osp.expanduser("~/.taskset_dirs"), osp.basename(exp_name))

    symlinked_files = []
    commands = [
        f"echo \"Current working directory: $PWD\"",
        f"echo \"Using temporary directory {temp_dir}\"",
        f"mkdir -p {temp_dir}"]
    
    # Ensure that data and weights are symlinked. We can assume this code is being run
    # from a directory capable of executing the script, ie. paths will be right.
    # Because these paths are being copied explicitly and with symlinks, we will
    # assume they should not be copied again, sym- or hardlinked.
    # if "data_tr" in script_args and osp.exists(script_args.data_tr):
    #     symlinked_files.append(get_rel_root(script_args.data_tr))
    #     cmd = get_symlink_to_rel_root(temp_dir=temp_dir, path=script_args.data_tr)
    #     if not cmd in commands:
    #         commands.append(cmd)
    # if "data_val" in script_args and osp.exists(script_args.data_val):
    #     symlinked_files.append(get_rel_root(script_args.data_val))
    #     cmd = get_symlink_to_rel_root(temp_dir=temp_dir, path=script_args.data_val)
    #     if not cmd in commands:
    #         commands.append(cmd)
    # if "colormae_file" in script_args and osp.exists(script_args.colormae_file):
    #     symlinked_files.append(get_rel_root(script_args.colormae_file))
    #     cmd = get_symlink_to_rel_root(temp_dir=temp_dir, path=script_args.colormae_file)
    #     if not cmd in commands:
    #         commands.append(cmd)

    # Ensure that any file contained in the command line arguments is symlinked.
    for k,v in vars(script_args).items():
        if isinstance(v, str) and osp.exists(v) and not v.startswith("/") and not v.startswith("~"):
            symlinked_files.append(get_rel_root(v))
            cmd = get_symlink_to_rel_root(temp_dir=temp_dir, path=v)
            if not cmd in commands:
                commands.append(cmd)

    # Copy everything to the save directory. We will use heuristics to ignore some
    # files that we almost certainly don't need/want to copy. Note that this will copy
    # the .git directory from whenever the script is being started.
    ignore_prefixes = ["__pycache__", "fgvc-aircraft", "fgvcaircraft", "aircraft", "flowers"]
    ignore_suffixes = [".PYC", ".PNG", ".JPEG", ".JPG"]
    ignore_suffixes += [s.lower() for s in ignore_suffixes]
    ignore_files = [f"{p}*" for p in ignore_prefixes] + [f"*{s}" for s in ignore_suffixes] + symlinked_files
    ignore_str = " ".join([f"--exclude='{f}'" for f in ignore_files])
    cp_everything_cmd = f"rsync -a {ignore_str} ./ {temp_dir}/"
    commands.append(cp_everything_cmd)

    # Now, if there is saved state, overwrite whatever we just copied with it. In each
    # case, there should either be a cur_git_sha.txt file or a .git subdirectory 
    # the former case, we need to remove the .git directory that was just copied as it
    # wouldn't match the code.
    if osp.exists(exp_name) and osp.exists(osp.join(exp_name, "code")):
        commands.append(f"cp -r --remove-destination {osp.join(exp_name, 'code')}/* {temp_dir}")
    elif osp.exists(exp_name) and osp.exists(osp.join(exp_name, "code.tar")):
        commands.append(f"tar -xf {osp.join(exp_name, 'code.tar')} -C {temp_dir}")
    elif osp.exists(exp_name) and osp.exists(osp.join(exp_name, "cur_git_sha.txt")):
        commands.append(f"rm -rf {osp.join(temp_dir, '.git')}")
        commands.append(f"cp {osp.join(exp_name, 'cur_git_sha.txt')} {temp_dir}")
    else:
        sha = subprocess.run("git rev-parse HEAD", capture_output=True, shell=True, text=True).stdout.strip()
        commands.append(f"echo {sha} > {osp.join(exp_name, 'cur_git_sha.txt')}")
        
    commands.append(f"cd {temp_dir}")
    # commands.append(f"echo \"Changed to directory $PWD -> found files are:\"")
    # commands.append(f"ls -lh")

    submit_dir = osp.abspath(os.getcwd())
    at_end_cmds = [f"echo \"Ran in {temp_dir}\"",
        f"echo \"Moving back to {submit_dir}\"",
        f"cd {osp.abspath(os.getcwd())}"]

    if args.remove_temp_dir:
        at_end_cmds += [f"echo \"Removing {temp_dir}\"", f"rm -rf {temp_dir}"]

    return "\n" + "\n".join(commands) + "\n", "\n" + "\n".join(at_end_cmds) + "\n"

    



if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-c", "--cpu_range", default="parse_gpus",
        help="CPU specification")
    gpu_parser = P.add_mutually_exclusive_group(required=True)
    gpu_parser.add_argument("--gpus", nargs="*", type=int, default=None,
        help="GPU specification")
    gpu_parser.add_argument("--strip_gpus", nargs="*", type=int, default=None,
        help="Like --gpus but not added to the arguments of the script being run")

    # P.add_argument("--gpus", nargs="*", type=int, default=None,
    #     help="GPU specification")
    # P.add_argument("--strip_gpus", nargs="*", type=int, default=None,
    #     help="Like --gpus but not added to the arguments of the script being run")
    P.add_argument("--shell", default="bash", choices=["bash", "zsh"],
        help="Shell type")
    
    P.add_argument("--taskset_scripts_dir", default=osp.expanduser("~/.taskset_scripts"),
        help="Directory to store taskset scripts")
    P.add_argument("-n", "--new_dir", default=1, type=int, choices=[0, 1],
        help="Runs in an isolated directory if possible")
    P.add_argument("--remove_temp_dir", default=1, type=int, choices=[0, 1],
        help="Removes the temporary directory at the end if --new_dir is set")

    P.add_argument("--allow_on_slurm", default=0, type=int, choices=[0, 1],
        help="Allows running on SLURM clusters. Usually this is not desired.")
    P.add_argument("--log_file", type=str, default=None,
        help="If specified, logs to this file in addition to stdout.")
    P.add_argument("--query_metas", default=1, type=int, choices=[0, 1],
        help="Queries the script being run for metadata about the experiment")

    P.add_argument("--try_add_wandb", choices=[0, 1], default=1, type=int,
        help="Tries to add --wandb online if not specified")
    P.add_argument("--time", type=str, default=None,
        help="Has no effect, but can resolve bugs where we accidentally use this.")
    P.add_argument("--taskset_debug", action="store_true",
        help="Print the taskset script instead of running it")

    P.add_argument("--basic", action="store_true",
        help="Print something morally equivalent to the command being run: CUDA_VISIBLE_DEVICES=... taskset -c ... python script.py --args ...")
    args, unparsed_args = P.parse_known_args()

    if Utils.is_slurm() and not args.allow_on_slurm:
        print("tpython_ddpX not for use on ComputeCanada.")
        sys.exit(1)

    # Parse the GPUs to use and whether or not to add a --gpus argument with them back
    # into the script that actually gets run
    specified_gpus = args.gpus if not args.gpus is None else args.strip_gpus
    add_back_gpus = args.strip_gpus is None

    # Get the CPU string for taskset based on --cpu_range and --gpus
    if args.cpu_range == "parse_gpus":
        args.cpu_range = get_cpus_from_gpus(gpus=specified_gpus)
        taskset_str = f"taskset -c {args.cpu_range}"
    elif args.cpu_range == "none":
        taskset_str = ""
    else:
        taskset_str = f"taskset -c {args.cpu_range}"

    cuda_visible_devices_str = f"CUDA_VISIBLE_DEVICES={','.join([str(g) for g in specified_gpus])}" 
    
    if args.basic:
        remaining_args = " ".join(unparsed_args)
        print(f"{cuda_visible_devices_str} {taskset_str} {remaining_args}")
        sys.exit(0)

    # Parse remaining arguments to those before the script being run, the script, and
    # a Namespace of arguments to the script
    before_script, script, script_args = unparsed_args_to_args(unparsed_args=unparsed_args)
    script_args = UtilsBase.updated_namespace(script_args, gpus=specified_gpus) if add_back_gpus else script_args

    # Map the tpython_ddpX or other prefix to the script to what it should actually be
    before_script = get_script_from_alias(before_script)

    # Query the script that we are running for metadata about the run. In particular,
    # this includes the experiment name, which is the folder stuff will save to, and
    # any arguments that should be included in the script args that were
    # auto-generated when it ran initially and thus informed the experiment name
    if args.query_metas:
        exp_metas = get_experiment_metadata(script=script, script_args=script_args)
        if "exp_name" in exp_metas:
            _ = os.makedirs(exp_metas.exp_name, exist_ok=True)
            log_file = osp.join(exp_metas.exp_name, "log.txt")
            start_cmd, end_cmd = get_new_directory_strs(exp_name=exp_metas.exp_name, args=args, script_args=script_args)
        else:
            new_directory_str = ""
            log_file = None
            start_cmd, end_cmd = "", ""
    else:
        exp_metas = argparse.Namespace()
        log_file = args.log_file
        start_cmd, end_cmd = "", ""

    # If the script being run said to add stuff to its arguments, then do so now
    if "include_in_args" in exp_metas:
        script_args = argparse.Namespace(**vars(script_args) | exp_metas.include_in_args)

    # If --log_file was specified, then it overrides any log_file we might have found
    # based on querying metadata from the script being run
    if not args.log_file is None and log_file is None:
        print(f"[INFO] Setting log_file={args.log_file} from --log_file")
        log_file = args.log_file
    elif not args.log_file is None and not log_file is None:
        print(f"[WARNING] --log_file was set, so changing log_file={log_file} -> {args.log_file}")
        log_file = args.log_file
    else:
        pass

    unparsed_args = args_to_unparsed_args(before_script=before_script, script=script, args=script_args)
    unparsed_args = " ".join(unparsed_args)  # Ensure it's a string

    command = f"command=\"{cuda_visible_devices_str} {taskset_str} {unparsed_args}\""
    full_command = f"full_command=\"$command $@ 2>&1 | tee -a '{log_file}'\"" if log_file else f"full_command=\"$command\""

    script_file = osp.join(args.taskset_scripts_dir, f"{uuid.uuid4()}.sh")
    script = f"source {shell2rc[args.shell]}\n{start_cmd}\n{command}\n{full_command}\necho \"Running: $full_command\"\neval \"$full_command\"\n{end_cmd}"

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
        os.system(f"{taskset_str} {args.shell} {script_file}")


