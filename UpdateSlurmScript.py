"""Wrapper for easily updating SLURM scripts."""
import argparse
import os
import os.path as osp
import subprocess
import sys

import FileFinding
import Utils
import UtilsBase
from UtilsBase import twrite, tqdm

plain_sbatch_keys = ["time", "account", "nodes", "gpus-per-node", "mem",
    "cpus-per-task", "ntasks-per-node"]
append_sbatch_keys = ["exclude", "constraint", "partition"]


def update_slurm_script(*, fname, args):
    if not osp.exists(fname):
        twrite(f"[ERROR] SLURM script {fname} does not exist!")
        return
    extant_script = UtilsBase.load_file_lite(fname)
    extant_script_lines = [l.strip() for l in extant_script.splitlines()]


    update_k2v = args.update_k2v
    update_k_append2v = {k: v for k,v in update_k2v.items() if k.endswith("_append")}
    update_k2v = {k: v for k,v in update_k2v.items() if not k.endswith("_append")}

    update_k2start_line = {k: idx for k in update_k2v.keys() for idx,l in enumerate(extant_script_lines) if l.startswith(f"#SBATCH --{k}=")}
    update_k2start_line_missing = set(update_k2v.keys()) - set(update_k2start_line.keys())
    update_k_append2start_line = {k: idx for k in update_k_append2v.keys() for idx,l in enumerate(extant_script_lines) if l.startswith(f"#SBATCH --{UtilsBase.strip_right(k, '_append')}=")}
    update_k_append2start_line_missing = set(update_k_append2v.keys()) - set(update_k_append2start_line.keys())

    replacement_map = dict()

    for k,start_line in update_k2start_line.items():
        l = extant_script_lines[start_line]
        old_value = l.split("=", 1)[1].strip()
        new_value = update_k2v[k]
        new_line = f"#SBATCH --{k}={new_value}"
        extant_script = extant_script.replace(l, new_line)
        replacement_map[k] = dict(v_old=old_value, v_new=new_value)
        twrite(f"[INFO] Updating SLURM script {fname}: {k}={old_value} -> {k}={new_value}", quiet=args.verbose < 1)

        # This can appear in other places, and would need to be updated there as well
        if k == "nodes":
            extant_script = extant_script.replace(f"--nodes {old_value}", f"--nodes {new_value}")
        elif k == "gpus-per-node":
            new_gpu_list = " ".join([str(i) for i in range(int(new_value))])
            extant_script = extant_script.replace(f"--gpus {old_value}", f"--gpus {new_gpu_list}")
            replacement_map[k] = dict(v_old=old_value, v_new=new_gpu_list)
    
    for k,start_line in update_k_append2start_line.items():
        l = extant_script_lines[start_line]
        old_value_list = l.split("=", 1)[1].strip().split(",")
        new_value = set(old_value_list) | set(update_k_append2v[k].split(","))
        new_value_str = ",".join(new_value)
        new_line = f"#SBATCH --{UtilsBase.strip_right(k, '_append')}={new_value_str}"
        extant_script = extant_script.replace(l, new_line)
        twrite(f"[INFO] Updating SLURM script {fname}: {k}={old_value_list} -> {k}={new_value_str}", quiet=args.verbose < 1)
        replacement_map[k] = dict(v_old=old_value, v_new=new_value)

    if len(update_k2start_line_missing) > 0 or len(update_k_append2start_line_missing) > 0:
        raise NotImplementedError(f"Currently, all SBATCH keys to update must already be present in the SLURM script. Missing keys: {update_k2start_line_missing} {update_k_append2start_line_missing}")

    dry_run_str = " (dry run)" if args.dry_run else ""
    replace_infos = [f"{k}={v['v_old']} -> {v['v_new']}" for k,v in replacement_map.items()]
    replace_infos_str = " ".join(replace_infos)
    one_line_print = f"[INFO] Updating SLURM script {fname}{dry_run_str}: {replace_infos_str}"
    if len(one_line_print) <= os.get_terminal_size().columns:
        twrite(one_line_print)
    else:
        multi_line_print = f"[INFO] Updating SLURM script {fname}{dry_run_str}:\n" + "\n\t".join(replace_infos)
        twrite(multi_line_print)

    if not args.dry_run:
        _ = UtilsBase.atomic_save_lite(data=extant_script, fpath=fname)
    
    



def get_args():
    def str_or_list_to_comma_sep_str(s):
        if isinstance(s, str):
            return s
        elif isinstance(s, list):
            return ",".join(s)
        else:
            raise argparse.ArgumentTypeError(f"Expected a string or a list of strings, got {type(s)}")

    P = argparse.ArgumentParser(prefix_chars="-+", allow_abbrev=False)
    P.add_argument("substrs", nargs="+",
		help="List of identifying substrings for scripts to update")
    # P.add_argument("-s", "--substrs", "--uids", nargs="+", default=[],
    #     help="Identifying substrings of things to update. These can be substrings of SLURM script filenames or jobids")
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("--search_dirs", nargs="+", default=FileFinding.slurm_script_search_dirs)
    P.add_argument("-v", "--verbose", action="count", default=0)

    # Additional things that can be updated. It's nice to hardcode common ones as it's
    # nicer to specify them in a canonical command line way.
    
    for sk in [sb for sb in plain_sbatch_keys if sb not in ["partition"]]:
        P.add_argument(f"--{sk}", type=str, default=None,
            help=f"If set, updates the --{sk} SBATCH argument to this value.")

    # For some SBATCH keys, we might want to update them in an append kind of way.
    # So for these, we will also have '+' style versions of the keys.
    
    for ak in append_sbatch_keys:
        group = P.add_mutually_exclusive_group()
        group.add_argument(f"--{ak}", nargs="?", const="", default=None, dest=f"{ak}",
            help=f"Updates --{ak} SBATCH argument to this value. If passed without a value, sets it to an empty string.")
        group.add_argument(f"+{ak}", nargs="+", default=None, dest=f"{ak}_append", type=str_or_list_to_comma_sep_str,
            help=f"If set, appends the value to the existing --{ak} SBATCH argument instead of replacing it. Can be a comma separated list or a list of strings.")

    P.add_argument("--update", nargs="+", default=[], type=str,
        help="List of key-value pairs passed as 'key=value' ")

    args = P.parse_args()

    ##################################################################################
    # Check the arguments for correctness and set them up
    ##################################################################################
    if not all(["=" in kv for kv in args.update]):
        raise argparse.ArgumentError(f"[ERROR] All --update arguments must be key-value pairs of the form key=value. Got {args.update}")
    update_k2v = {k: v for k,v in [kv.split("=", 1) for kv in args.update]}

    all_keys = plain_sbatch_keys + append_sbatch_keys
    vars_args_to_check = {k: v for k,v in vars(args).items() if k in all_keys and not v is None}
    for k,v in vars_args_to_check.items():
        if k in update_k2v:
            raise argparse.ArgumentError(f"[ERROR] Both --{k} and --update {k}=... were set. Specify only one.")
        else:
            update_k2v[k] = v

    vars_args_to_check = {f"{ak}_append": v for ak in append_sbatch_keys for k,v in vars(args).items() if k == f"{ak}_append" and not v is None}
    for k,v in vars_args_to_check.items():
        if UtilsBase.strip_right(k, "_append") in update_k2v:
            raise argparse.ArgumentError(f"[ERROR] Both +{ak} and --update {ak}=... were set. Specify only one.")
        else:
            update_k2v[k] = v

    return UtilsBase.updated_namespace(args, update_k2v=update_k2v)

if __name__ == "__main__":
    args = get_args()

    ##################################################################################
    # Map everything in [substrs] to a SLURM script if possible
    ##################################################################################
    substr2slurm_script = dict()
    
    # Heuristic: If a passed in identifier for something to update is numeric, first
    # assume it's a jobid and try to get the SLURM script from that.
    numeric_substrs = [s for s in args.substrs if s.isnumeric()]
    for s in tqdm(numeric_substrs, desc="Finding SLURM scripts from numeric substrs/jobids"):
        scontrol_cmd = f"scontrol show job {s} | grep Command="
        output = subprocess.getoutput(scontrol_cmd).strip()
        _ = twrite(f"[INFO] Running command: {scontrol_cmd}", quiet=args.verbose < 2)
        _ = twrite(f"[INFO] Got output:\n{output}", quiet=args.verbose < 2)
        
        if output.startswith("Command="):
            slurm_script = output.split("Command=")[1].strip()

            # Ensure that the SLURM script actually exists and that we have write
            # permissions for it
            if not osp.exists(slurm_script):
                twrite(f"[WARNING] Found SLURM script {slurm_script} for assumed-jobid {s} -> will try again later assuming it uniquely identifies a SLURM script")
                pass
            elif not os.access(slurm_script, os.W_OK):
                twrite(f"[WARNING] Found SLURM script {slurm_script} for jobid {s}, but it is not writable -> will try again later assuming it uniquely identifies a SLURM script")
                pass
            else:
                substr2slurm_script[s] = slurm_script
                twrite(f"[INFO] assuming {s} is a jobid -> found SLURM script {substr2slurm_script[s]}")
        else:
            twrite(f"[WARNING] Could not find SLURM script for assumed-jobid {s} -> will try again later assuming it uniquely identifies a SLURM script")
            continue

    remaining_substrs = [s for s in args.substrs if s not in substr2slurm_script]
    for s in tqdm(remaining_substrs, desc="Finding SLURM scripts from substrings"):
        substr2slurm_script[s] = FileFinding.str_to_file(s,
            search_dirs=args.search_dirs,
            file_type="slurm",
            verbose=args.verbose,
            resolve="half_then_user")

    additional_print_info = "\n\t".join(list(substr2slurm_script.values()))
    additional_print_info = f":\n{additional_print_info}" if args.verbose else ""
    twrite(f"[INFO] Found {len(substr2slurm_script.values())} SLURM scripts to update{additional_print_info}")
    
    for slurm_script in substr2slurm_script.values():
        _ = update_slurm_script(
            fname=slurm_script,
            args=args,)
