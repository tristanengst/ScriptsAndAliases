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

def update_slurm_script(*, fname, sbatch_kv_pairs, replace_substr_pairs, args):
    """Updates the SLURM script at [fname] by changing the SBATCH key-value pairs
    in [sbatch_kv_pairs] and replacing substrings according to [replace_substr_pairs].
    """
    if not osp.exists(fname):
        twrite(f"[ERROR] SLURM script {fname} does not exist!")
        return

    sbatch_kv_pairs = set(sbatch_kv_pairs)
    sbatch_kv = [skv.split("=") for skv in sbatch_kv_pairs if skv.count("=") == 1]
    sbatch_k2v = dict(sbatch_kv)
    # TODO: Handle cases where there isn't exactly one '=' in the sbatch_kv_pair  
    
    # Want to make sure all the keys are updated
    replacement_map = dict()
    
    extant_script = UtilsBase.load_file_lite(fname)
    extant_script_lines = [l.strip() for l in extant_script.split("\n")]
    for line in extant_script_lines:
        for k,v in sbatch_k2v.items():
            if line.startswith(f"#SBATCH --{k}="):
                old_value = line.split(f"#SBATCH --{k}=")[1].strip()
                replacement_map[k] = dict(line=line,
                    kv_old=f"#SBATCH --{k}={old_value}", kv_new=f"#SBATCH --{k}={v}",
                    v_old=old_value, v_new=v)

    # TODO: There is a more complete way to handle this
    if not len(replacement_map) == len(sbatch_k2v):
        missing_keys = set(sbatch_k2v.keys()) - set(replacement_map.keys())
        twrite(f"[WARNING] Could not find SBATCH keys {missing_keys} in SLURM script {fname}. These will not be updated.")
    for k,v in replacement_map.items():
        extant_script = extant_script.replace(v["kv_old"], v["kv_new"])
    
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
        _ = UtilsBase.atomic_save_lite(data=extant_script, fname=fname)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-s", "--substrs", "--uids", nargs="+", default=[],
        help="Identifying substrings of things to update. These can be substrings of SLURM script filenames or jobids")
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("--search_dirs", nargs="+", default=FileFinding.file_search_dirs)
    P.add_argument("-v", "--verbose", action="store_true")

    # Additional things that can be updated. It's nice to hardcode common ones as it's
    # nicer to specify them in a canonical command line way.
    sbatch_keys = ["time"]
    P.add_argument("--update", nargs="+", default=[],
        help="List of key-value pairs such that #SBATCH --key=value can be updated in the SLURM scripts.")
    P.add_argument("--time", type=str, default=None,
        help="If set, updates the --time SBATCH argument to this value.")
    args = P.parse_args()

    # Place SLURM sbatch key-value pairs specified from the main argparse into the
    # --update list.
    for sk in sbatch_keys:
        if not vars(args)[sk] is None and not any([skv.startswith(f"{sk}=") for skv in args.update]):
            args.update.append(f"{sk}={vars(args)[sk]}")
        elif not vars(args)[sk] is None and any([skv.startswith(f"{sk}=") for skv in args.update]):
            raise argparse.ArgumentError(f"[ERROR] Both --{sk} and --update {sk}=... were set. Specify only one.")
        else:
            pass

    substr2slurm_script = dict()
    
    # Heuristic: If a passed in identifier for something to update is numeric, first
    # assume it's a jobid and try to get the SLURM script from that.
    numeric_substrs = [s for s in args.substrs if s.isnumeric()]
    job2info = Utils.get_slurm_status(cur_user=True) if len(numeric_substrs) else None
    for s in numeric_substrs:
        if s in job2info:
            scontrol_cmd = f"scontrol show job {s} | grep Command="
            output = subprocess.getoutput(scontrol_cmd).strip()
            if args.verbose:
                twrite(f"[INFO] Running command: {scontrol_cmd}")
                twrite(f"[INFO] Got output:\n{output}")
            if output.startswith("Command="):
                substr2slurm_script[s] = output.split("Command=")[1].strip()
                twrite(f"[INFO] assuming {s} is a jobid -> found SLURM script {substr2slurm_script[s]}")
                continue
    
    non_numeric_substrs = [s for s in args.substrs if not s.isnumeric()]
    file_finding_kwargs = dict(search_dirs=args.search_dirs, slurm_or_result="slurm", verbose=args.verbose, resolve="half_then_user")
    for s in tqdm(non_numeric_substrs, desc="Finding SLURM scripts from substrs/uids"):
        substr2slurm_script[s] = FileFinding.str_to_file(s, **file_finding_kwargs)
    
    additional_print_info = "\n\t".join(list(substr2slurm_script.values()))
    additional_print_info = f":\n{additional_print_info}" if args.verbose else ""
    twrite(f"[INFO] Found {len(substr2slurm_script.values())} SLURM scripts to update{additional_print_info}")
    
    for slurm_script in substr2slurm_script.values():
        _ = update_slurm_script(
            fname=slurm_script,
            sbatch_kv_pairs=args.update,
            replace_substr_pairs=[],
            args=args,)
