import argparse
from collections import defaultdict
import glob
import os
import os.path as osp
import subprocess

import UtilsBase
from UtilsBase import twrite

def get_cluster_type():
    """Returns a string for special host types, or None if they are not recognized."""
    h = os.uname()[1]
    if "nibi" in h:
        return "nibi"
    elif "trig" in h or "trillium" in h:
        return "trillium"
    elif h.startswith("fc") or h.startswith("login"):
        return "fir"
    elif h.startswith("rorqual") or h.startswith("rq") or h.startswith("rg") or h.startswith("rl"):
        return "rorqual"
    elif h.startswith("narval") or h.startswith("ng"):
        return "narval"
    elif h.startswith("cedar") or h.startswith("cdr"):
        return "cedar"
    elif h.startswith("beluga") or h.startswith("bg"):
        return "beluga"
    elif h.startswith("gra-") or h.startswith("gra") or h.startswith("gr"):
        return "graham"
    elif h.startswith("cs-s") or h.startswith("cs-v"):
        return "solar"
    else:
        h_ = "-".join(h.split("-")[:2])
        return os.environ.get("CLUSTER_TYPE", "cs-apex")

def is_solar(): return get_cluster_type() == "solar"
def is_cc(): return get_cluster_type() in ["nibi", "narval", "cedar", "beluga", "graham", "rorqual", "trillium", "fir"]
def is_slurm(): return is_solar() or is_cc()
def is_workstation(): return not is_solar() and not is_cc()

def get_slurm_status(cur_user=False, account=None, verbose=False,
    keys=["jobid", "user", "state", "start_time", "time_left", "time_limit", "gres",
        "nodes", "name", "reason", "account", "partition", "host", "exclude",
        "comment", "submit_time", "eligible_time", "stderr", "stdout", "uid"],
    key2sq_format=dict()
    ):
    """Returns a dictionary describing the entire state what's running. Strings are
    not processed or reformatted in any way.

    Args:
    cur_user        -- whether to get results for only the current user's jobs
    account         -- if not None, only show jobs for this account
    keys            -- which keys to include in the output. See
                    https://slurm.schedmd.com/squeue.html for details, but some
                    'uid' is custom
    key2sq_format   -- Additional key2sq_format entries to use with -o formatting in
                        addition to those used in [keys]. Keys whose values have
                        colons in them (eg. 'SubmitTime:64') get -O formatting

    Notes:
    - Some keys must generally be included for sensible results, eg. 'jobid'
    - Some keys require others. In particular, 'uid' requires 'comment'
    """
    import json
    def try_parse_comment(c):
        """Tries to parse the comment [c]."""
        c = c.strip()
        if c.startswith("{") and c.endswith("}"):
            c = c.replace("'", "\"")  # Replace single quotes with double quotes
            try:
                return json.loads(c) | dict(comment=c)
            except json.JSONDecodeError:
                return dict(comment=c)
        else:
            return dict(comment=c)

    # On ComputeCanada, get jobs by account
    if account is None and is_cc():
        accounts = ["rrg-keli_gpu", "rrg-keli_cpu", "def-keli_gpu", "def-keli_cpu"]
        result = dict()
        for account in accounts:
            result |= get_slurm_status(cur_user=cur_user, account=account)
        return result

    key2required_keys = defaultdict(list, dict(
        uid=["comment"],
    ))

    def find_required_keys(k):
        """Recursively finds all required keys for key [k]."""
        req_keys = key2required_keys[k]
        extra = [find_required_keys(kr) for kr in req_keys if not kr == k and not kr in keys]
        extra = UtilsBase.flatten(extra) if extra else extra
        return req_keys + extra

    extra_keys = list(set(UtilsBase.flatten([find_required_keys(k) for k in keys])))
    keys += extra_keys

    user_str = "-u $USER" if cur_user else ""
    account_str = f"-A {account}" if account else ""
    sep = "  |_||_|||_|||||  "

    key2sq_format_o = dict(
        jobid=f"%i",
        user=f"%u",
        state=f"%T",
        start_time=f"%S",
        time_left=f"%L",
        time_limit=f"%l",
        gres=f"%b",
        nodes=f"%D",
        name=f"%j",
        reason=f"%r",
        account=f"%a",
        partition=f"%P",
        host=f"%N",
        exclude=f"%x",
        comment=f"%k"
    )

    key2sq_format_O = dict(
        jobid="JOBID:10",
        submit_time="SubmitTime:64",
        eligible_time="EligibleTime:64",
        stderr="StdErr:1024",
        stdout="StdOut:1024"
    )

    key2sq_format_o = {k: v for k,v in key2sq_format_o.items() if k in keys} | {k: v for k,v in key2sq_format.items() if not ":" in v}
    key2sq_format_O = {k: v for k,v in key2sq_format_O.items() if k in keys} | {k: v for k,v in key2sq_format.items() if ":" in v}
    
    
    sq_format_str = sep.join(key2sq_format_o.values())
    sq_cmd = f"squeue {user_str} {account_str} -h -o \"{sq_format_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()

    if verbose:
        print(f"Running command: {sq_cmd}")
        print(f"Output:\n{sq}")
    
    if sq == "":
        twrite(f"[INFO] No jobs found for cur_user={cur_user}, account={account}", cur_user=cur_user, account=account, verbose=verbose)
        return dict()

    jobs = sq.split("\n")
    jobs = [j.strip().split(sep) for j in jobs]
    infos = [dict(list(zip(key2sq_format_o.keys(), j))) for j in jobs]
    job2info = {info["jobid"]: info for info in infos}
    job2info = {j: info | dict(comment=try_parse_comment(info["comment"])) for j,info in job2info.items()} if "comment" in keys else job2info
    job2info = {j: info | dict(uid=info["comment"].get("uid", None)) for j,info in job2info.items()} if "uid" in keys else job2info

    # Possible early exit if no -O formatting keys are needed
    if len(key2sq_format_O) == 0 or list(key2sq_format_O.keys()) == ["jobid"]:
        result = {j: argparse.Namespace(**info) for j,info in job2info.items()}
    
    sq_key_str = f"{sep},".join(key2sq_format_O.values())
    sq_cmd = f"squeue {user_str} {account_str} -h -O \"{sq_key_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()
    if verbose:
        print(f"Running command: {sq_cmd}")
        print(f"Output:\n{sq}")

    
    jobs = sq.split("\n")

    # Hack because Solar's SLURM is different?
    if is_solar():
        jobs = [j.strip().split()[:len(key2sq_format_O.values())] for j in jobs]
    else:
        jobs = [j.strip().split(sep) for j in jobs]

    jobs = [[j1.strip() for j1 in j] for j in jobs]
    infos = [dict(list(zip(key2sq_format_O.keys(), j))) for j in jobs]
    job2info_ = {info["jobid"]: info for info in infos}

    job2info = {j: info1 | job2info_[j] for j,info1 in job2info.items() if j in job2info_}
    job2info = {j: argparse.Namespace(**info) for j,info in job2info.items()}
    return job2info


def jobid2info_to_uid2jobids(job2info=None):
    """Converts a job2info dictionary to a uid2jobids dictionary."""
    from collections import defaultdict
    job2info = job2info if job2info else get_slurm_status(cur_user=True)

    uid2jobids = defaultdict(list)
    for jobid,info in job2info.items():
        if not info.uid is None:
            uid2jobids[info.uid].append(jobid)
    return dict(uid2jobids)






def query_yes_no(msg="Proceed? (y/n): "):
    """Queries the user to proceed. Returns True if the user wants to proceed, False otherwise."""
    print(msg)
    while True:
        choice = input("")
        if choice.lower() in ["y", "yes"]:
            return True
        elif choice.lower() in ["n", "no"]:
            return False
        else:
            print(f"[WARNING] Invalid choice: {choice} -> try again")



























def compress_user(path):
    """Inverse to osp.expanduser(). Tries to follow symlinks wherever possible."""
    return osp.relpath(osp.abspath(osp.expanduser(path)), osp.expanduser("~"))

exp_search_dirs = [osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
    osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
    osp.expanduser("~/scratch/IMLE-SSL/finetunes")]

file_search_dirs = [osp.expanduser("~/Development/IMLE-SSL-2/pretrain_results"),
    osp.expanduser("~/Development/IMLE-SSL-2/finetune_results"),
    osp.expanduser("~/Development/IMLE-SSL-2/slurm")]

def str_to_slurm_info(s, job2info=None, verbose=False):
    """Tries to find the slurm info for a given string [s]."""
    return get_slurm_info_by_key(s, key="name", job2info=job2info, verbose=verbose)
def uid_to_slurm_info(s, job2info=None, verbose=False):
    """Tries to find the slurm info for a given string [s]."""
    return get_slurm_info_by_key(s, key="uid", job2info=job2info, verbose=verbose)
def get_slurm_info_by_key(s, key, job2info=None, verbose=False, resolve="pos", search_dirs=exp_search_dirs):
    """Tries to find the slurm info for a given string [s]."""
    job2info = job2info if job2info else get_slurm_status(cur_user=True, verbose=verbose)
    job2info = {j: info for j,info in job2info.items() if s in vars(info).get(key, "")}

    if len(job2info) == 0:
        _ = twrite(f"[INFO] No jobs found for str={s}", verbose=verbose)
        return None
    elif len(job2info) == 1:
        return list(job2info.values())[0]
    elif len(job2info) > 1 and resolve in ["pos"]:
        job2info = sorted(job2info.values(), key=lambda info: len(info) - info.name.rfind(s))
        return job2info[0]
    else:
        raise NotImplementedError(f"[ERROR] get_slurm_info_by_key(): Multiple jobs found for str={s} with key={key}")

def uid_to_exp_folder(uid, search_dirs=exp_search_dirs, verbose=False, resolve="pos"):
    """Tries to find the experiment folder for a given UID."""
    return str_to_exp_folder(uid, search_dirs=search_dirs, resolve=resolve, verbose=verbose)

def exp_folder_to_uid(exp_folder, verbose=False):
    """Tries to find the UID for an experiment folder."""
    if not osp.exists(exp_folder) or not osp.isdir(exp_folder):
        _ = twrite(f"[WARNING] exp_folder={exp_folder} does not exist or is not a folder", verbose=verbose)
        return None
    
    if osp.exists(osp.join(exp_folder, "config.json")):
        content = UtilsBase.load_file_lite(osp.join(exp_folder, "config.json"))
        if "uid" in content:
            return content["uid"]
    
    _ = twrite(f"[WARNING] Could not find UID for exp_folder={exp_folder}", verbose=verbose)
    return None



def str_to_exp_folder(s, search_dirs=exp_search_dirs, resolve="half", verbose=False, matches=None):
    """Returns the experiment folder that matches the string [s]. If there are multiple possible matches, then one of several strategies can be used to resolve them.

    Args:
    s           -- string to match. Does not need pre-globbing
    search_dirs -- directories to search in if [s] does not exist directly
    resolve     -- how to resolve multiple matches. One of:
                    ps -- the one where the match ends nearest to the end of the string is chosen
                    user -- the user is prompted to choose
                    half_then_user -- the one where the match is in the second half of the basename is chosen; if multiple, the user is prompted to choose
                    latest -- the one with the most recent modification time is chosen
    matches     -- if provided, use this list of matches instead of searching
    verbose     -- whether to print verbose messages
    """
    s = s.strip()
    if osp.exists(s) and osp.isdir(s):
        return s
    
    matches = str_to_all_exp_folders(s, search_dirs=search_dirs, verbose=verbose) if matches is None else matches

    if len(matches) == 0:
        raise FileNotFoundError(f"str_to_exp_folder(): No experiment folders found matching {s} in {search_dirs}")
    elif len(matches) == 1:
        return matches[0]
    elif resolve == "pos":
        matches = sorted(matches, key=lambda m: len(s) - m.rfind(s))
        return matches[0]
    elif resolve == "user":
        print(f"[INFO] Found multiple matches for {s}:")
        for i,m in enumerate(matches):
            print(f"\t{idx+1}: {m}")
        
        while True:
            choice = input(f"Enter the number of the experiment folder to choose (1-{len(matches)}), or 0 to cancel: ")
            if choice.isdigit() and int(choice) == 0:
                raise KeyboardInterrupt()
            elif choice.isdigit() and 1 <= int(choice) <= len(matches):
                return [matches[int(choice)-1]]
            else:
                print(f"[WARNING] Invalid choice: {choice} -> try again")

    # Valid matches are those where the match ends in the second half of the basename.
    # This tends to be the most unique part of the name.
    elif resolve == "half":
        matches2match_idxs = {m: (osp.basename(m).rfind(s), osp.basename(m).rfind(s) + len(s)) for m in matches if s in m}
        new_matches = [m for m, (start_idx, end_idx) in matches2match_idxs.items() if start_idx >= len(osp.basename(m)) // 2]
        if len(new_matches) == 0:
            raise ValueError(f"[ERROR] str_to_exp_folder(): zero matches for {s} with resolve='{resolve}', but there were multiple original matches:\n\t{UtilsBase.list_to_pretty_str(matches)}")
        elif len(new_matches) > 1:
            raise ValueError(f"[ERROR] str_to_exp_folder(): multiple matches for {s} with resolve='{resolve}':\n\t{UtilsBase.list_to_pretty_str(new_matches)}")
        else:
            return new_matches[0]

    # Try first using resolve='half', and if this fails, fall back to the user.
    elif resolve == "half_then_user":
        matches2match_idxs = {m: (osp.basename(m).rfind(s), osp.basename(m).rfind(s) + len(s)) for m in matches if s in m}
        matches = [m for m, (start_idx, end_idx) in matches2match_idxs.items() if start_idx >= len(osp.basename(m)) // 2]

        if len(matches) == 0:
            raise ValueError(f"[ERROR] str_to_exp_folder(): zero matches for {s} with resolve='{resolve}', but there were multiple original matches:\n\t{UtilsBase.list_to_pretty_str(matches)}")
        else:
            return str_to_exp_folder(s, search_dirs=search_dirs, resolve="user", verbose=verbose, matches=matches)
    
    # Return the most-recently modified match. Need to check all files in the folder,
    # but assume we don't need to do so recursively.
    elif resolve == "latest":
        matches2mtime = {m: max([osp.getmtime(osp.join(m, f)) for f in os.listdir(m)]) for m in matches}
        matches = sorted(matches, key=lambda m: matches2mtime[m])
        return matches[-1]
    
    elif resolve == "all":
        return matches
    else:
        raise ValueError(f"str_to_exp_folder(): Unknown resolve method {resolve}")


