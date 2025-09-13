"""Run ls on a bunch of files and figure out the status of the files they go with."""
import argparse
from datetime import datetime
from collections import defaultdict
import glob
import json
import os
import os.path as osp
import subprocess
import shutil

import Utils
import UtilsBase
from UtilsBase import twrite

slurm_state2sort_key = {"COMPLETING": 4, "RUNNING": 3, "PENDING": 2, "not in squeue": 1, "finished": 0, "N/A": -1}
slurm_state2sort_key = defaultdict(lambda: -1, slurm_state2sort_key)

def possible_exp_folder_to_uid(folder):
    """Heuristic to determine if a folder is an experiment folder. Either returns the
    UID of the experiment, or False.
    """
    if not osp.exists(folder) or not osp.isdir(folder):
        return False

    if osp.exists(osp.join(folder, "config.json")):
        config = UtilsBase.load_file_lite(osp.join(folder, "config.json"))
        if "uid" in config:
            return config["uid"]
        else:
            twrite(f"[WARNING] Found config.json in {folder} but no uid field.")

    if osp.exists(osp.join(folder, "wandb_data.pt")):
        assert 0, f"[DEBUG] needed to look in wandb_data.pt in {folder}. TODO: find the UID"

    return False

def fname_to_exp_folder(*, fname, args):
    """Given a file, try to figure out the experiment folder(s) it goes with."""
    def find_exp_folder_in_search_dirs(exp_name):
        found_exp_folders = []
        for s in args.exp_search_dirs:
            candidate = osp.join(s, exp_name)
            if possible_exp_folder_to_uid(candidate):
                found_exp_folders.append(candidate)
        
        if len(found_exp_folders) == 0:
            return None
        if len(found_exp_folders) > 1:
            twrite(f"[WARNING] Found multiple experiment folders for {exp_name}: {found_exp_folders}. Using the first one.")
        return found_exp_folders[0]
    
    
    if not osp.exists(fname):
        return False

    if osp.isdir(fname) and possible_exp_folder_to_uid(fname):
        return fname

    content = UtilsBase.load_file_lite(fname)
    if "--exp " in content and fname.endswith(".sh"):
        exp_folder = content.split("--exp ")[1].split(" ")[0].strip()
        if heuristic_is_exp_folder(exp_folder):
            return exp_folder

    if "#SBATCH --comment=\"" in content and fname.endswith(".sh"):
        comment = content.split("#SBATCH --comment=\"")[1].split("\"")[0]
        comment = json.loads(comment.strip())
        exp_name = comment.get("exp_name", comment.get("exp", None))
        exp_folder = find_exp_folder_in_search_dirs(exp_name)
        if exp_folder:
            return exp_folder

    if "EXPERIMENT:" in content and fname.endswith(".txt"):
        exp_name = content.split("EXPERIMENT:")[1].split("\n")[0].strip()
        exp_folder = find_exp_folder_in_search_dirs(exp_name)
        if exp_folder:
            return exp_folder

    # Possibly the file is inside the experiment folder?
    if possible_exp_folder_to_uid(osp.dirname(fname)):
        return osp.dirname(fname)

    return None


def exp_folder_to_status_dict(*, exp_folder, uid, args, slurm_status="check"):
    """Given an experiment folder, return a status line for it. This should include:
    1. UID of the experiment
    2. If there are any '..._latest.pt' files, and if so, their modification time, or if there is 'finished.txt' and its modification time
    3. The name and dirname
    4. Whether the job is running or queued in SLURM
    """
    latest_or_finished_files = [f for f in os.listdir(exp_folder) if f.endswith("_latest.pt") or f == "finished.txt"]
    if latest_or_finished_files:
        latest_and_time = [(f, osp.getmtime(osp.join(exp_folder, f))) for f in latest_or_finished_files]
        latest_and_time = sorted(latest_and_time, key=lambda x: x[1])
        latest_file = latest_and_time[-1][0]
        latest_time = latest_and_time[-1][1]
        latest_time_str = datetime.fromtimestamp(latest_time).strftime("%Y-%m-%d-%H:%M")
    else:
        latest_file, latest_time_str = "", ""

    # If possible, see if we can get the SLURM status of the job
    if (slurm_status == "check" and int(subprocess.getoutput("sinfo >/dev/null 2>&1 && echo 1 || echo 0"))) or isinstance(slurm_status, dict):
        job2info = UtilsBase.get_slurm_status(cur_user=True) if slurm_status == "check" else slurm_status
        slurm_states = {info.state for info in job2info.values() if info.uid == uid}

        if slurm_states:
            slurm_state = f"{', '.join(sorted(list(slurm_states)))}"
        else:
            slurm_state = "finished" if "finished.txt" in os.listdir(exp_folder) else "not in squeue"
    else:
        slurm_state = "N/A"

    if args.abspath:
        exp_folder = osp.abspath(exp_folder)
    else:
        exp_folder = osp.join(osp.basename(osp.dirname(exp_folder)), osp.basename(exp_folder))
    return dict(uid=uid, latest_file=latest_file, latest_time=latest_time_str, slurm_state=slurm_state, exp_folder=exp_folder)


def heuristic_find_file(fname, args):
    if fname.endswith(".sh") or fname.endswith(".txt"):
        for ff in args.file_search_dirs:
            if osp.exists(osp.join(ff, fname)):
                return osp.join(ff, fname)
    for s in args.exp_search_dirs:
        if osp.exists(osp.join(s, fname)):
            return osp.join(s, fname)
    return False



if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--files", nargs="+")
    P.add_argument("--exp_search_dirs", nargs="+", default=[
        osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
        osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
        osp.expanduser("~/scratch/IMLE-SSL/models_finetunes")])
    P.add_argument("--file_search_dirs", nargs="+", default=[
        osp.expanduser("~/Development/IMLE-SSL-2/pretrain_results"),
        osp.expanduser("~/Development/IMLE-SSL-2/finetune_results"),
        osp.expanduser("~/Development/IMLE-SSL-2/slurm")]),

    P.add_argument("-s", "--sort", choices=["uid", "latest_time", "latest_file", "state"], default="latest_time",
        help="How to sort the output")
    P.add_argument("-a", "--abspath", action="store_true",
        help="If set, will print absolute paths instead of shorter paths")
    args = P.parse_args()

    args.files = UtilsBase.flatten([ff.split()[-1] for f in args.files for ff in f.split("\n")])
    
    all_files = []
    for f in args.files:
        if "*" in f:
            all_files += glob.glob(f)
        elif osp.exists(f) and osp.isdir(f):
            all_files += [osp.join(f, ff) for ff in os.listdir(f)]
        else:
            all_files.append(f)

    all_files = [heuristic_find_file(f, args) or f for f in all_files]
    all_files = [f for f in all_files if f]  # Remove any that were not found

    exists_files = [f for f in all_files if osp.exists(f)]
    not_exists_files = [f for f in all_files if not osp.exists(f)]
    if not_exists_files:
        _ = twrite(f"Found {len(exists_files)} existing files and {len(not_exists_files)} non-existing files.")
    all_files = exists_files

    if not all_files:
        _ = twrite("No files matched")
        sys.exit(0)

    file2exp_folder = {f: fname_to_exp_folder(fname=f, args=args) for f in all_files}

    # For each file, we need to try and figure out if either it's an experiment
    # folder, or how to get to the experiment folder it goes with.
    file2exp_folder_uid = dict()
    for f,exp_folder in file2exp_folder.items():
        if exp_folder:
            possible_uid = possible_exp_folder_to_uid(exp_folder)
            if possible_uid:
                file2exp_folder_uid[f] = (exp_folder, possible_uid)
                continue
            else:
                print(f"[DEBUG] Found exp folder {exp_folder} for {f} but could not find UID")
        else:
            pass

    # See if we're on SLURM
    if int(subprocess.getoutput("sinfo >/dev/null 2>&1 && echo 1 || echo 0")):
        job2info = Utils.get_slurm_status(cur_user=True, keys=["jobid", "name", "state", "uid"])
    else:
        _ = twrite(f"[INFO] Not on a SLURM system -> not checking job status")
        job2info = None

    columns = ["uid", "latest_file", "latest_time", "exp_folder", "slurm_state"]
    status_dicts = [exp_folder_to_status_dict(exp_folder=exp_folder, uid=uid, args=args, slurm_status=job2info) for exp_folder,uid in file2exp_folder_uid.values()]

    col2max_chars = {c: len(c) for c in columns}
    for status in status_dicts:
        for c in columns:
            col2max_chars[c] = max(col2max_chars[c], 0 if (not c in status or not isinstance(status[c], str)) else len(status[c]))

    print_width = min(shutil.get_terminal_size().columns, sum(col2max_chars.values()))
    header = " CHECK RESULTS ".center(print_width, "=")
    header = f"\n\n\n{header}\n"
    header += " | ".join([c.ljust(col2max_chars[c]) for c in columns])
    print(header.upper())

    def dict_to_sort_key(d, sort_key=None):
        if args.sort == "latest_time":
            v = d.get("latest_time", None)
            return datetime.strptime(v, "%Y-%m-%d-%H:%M") if v else float("-inf")
        elif args.sort == "latest_file":
            v = d.get("latest_file", None)
            return int(UtilsBase.remove_nonnumeric(v)) if v else float("-inf")
        elif args.sort == "state":
            return slurm_state2sort_key[d.get("slurm_state", "N/A")]
        elif sort_key in d:
            return d.get(sort_key, "")
        else:
            return ""

    status_dicts = sorted(status_dicts, key=lambda d: dict_to_sort_key(d, sort_key=args.sort))
        
    for status in status_dicts:
        line = " | ".join([str(status.get(c, "")).ljust(col2max_chars[c]) for c in columns])
        print(line)
    







