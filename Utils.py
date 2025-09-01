import argparse
import os
import os.path as osp
import subprocess
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
def is_workstation(): return not is_solar() and not is_cc()


def format_time_delta(td, days=False, seconds=True):
    """Returns timde delta [td] formatted as per the arguments.
    
    Args:
    td          -- time delta string formatted as HH:MM:SS or DD-HH:MM:SS. This is
                    what Solar and ComputeCanada SLURM seems to output.
    days        -- if the time delta is more than 24H, express the days separately
    seconds     -- if True, include seconds in the output, otherwise remove them
    """
    if "-" in td:
        dd, hhmmss = td.split("-")
        hh, mm, ss = hhmmss.split(":")
        dd, hh, mm, ss = int(dd), int(hh), int(mm), int(ss)
    elif td.count(":") == 2:
        dd, hhmmss = "0", td
        hh, mm, ss = hhmmss.split(":")
        dd, hh, mm, ss = int(dd), int(hh), int(mm), int(ss)
    elif td.count(":") == 1:
        dd, hhmmss = "0", td
        hh = "0"
        mm, ss = hhmmss.split(":")
        dd, hh, mm, ss = int(dd), int(hh), int(mm), int(ss)

    total_seconds = (dd * 24 * 3600) + (hh * 3600) + (mm * 60) + ss

    if days:
        dd = total_seconds // (24 * 3600)
        hh = (total_seconds - (dd * 24 * 3600)) // 3600
    else:
        dd = 0
        hh = total_seconds // 3600

    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60

    dd_str = f"{dd}D" if dd > 0 else ""
    hh_str = f"{hh:03}:"
    mm_str = f"{mm:02}:" if seconds else f"{mm:02}"
    ss_str = f"{ss:02}" if seconds else ""
    return f"{dd_str}{hh_str}{mm_str}{ss_str}"

def get_slurm_status(cur_user=False, account=None, verbose=False):
    """Returns a dictionary describing the entire state what's running. Strings are
    not processed or reformatted in any way.

    Args:
    cur_user        -- if True, only show jobs for the current user
    account         -- if not None, only show jobs for this account
    submit_time     -- if True, include the submit time for all jobs
    eligible_time   -- include the eligible time for all jobs
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

    user_str = "-u $USER" if cur_user else ""
    account_str = f"-A {account}" if account else ""
    

    # First, get everything we can with -o formatting. TODO: could possibly make all of this happen with -O formatting?
    key2sq_format = dict(
        jobid=f"%i",
        user=f"%u",
        state=f"%T",
        start_time=f"%S",
        time_left=f"%L",
        gres=f"%b",
        nodes=f"%D",
        name=f"%j",
        reason=f"%r",
        account=f"%a",
        partition=f"%P",
        host=f"%N",
        exc_nodes=f"%x",
        comment=f"%k"
    )
    sep = "  |_||_|||_|||||  "
    sq_format_str = sep.join(key2sq_format.values())
    sq_cmd = f"squeue {user_str} {account_str} -h -o \"{sq_format_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()


    # verbose = True
    if verbose:
        print(f"Running command: {sq_cmd}")
        print(f"Output:\n{sq}")
    
    if sq == "":
        print(f"[INFO] No jobs found for cur_user={cur_user}, account={account}", cur_user=cur_user, account=account, verbose=verbose)
        return dict()
    
    jobs = sq.split("\n")
    jobs = [j.strip().split(sep) for j in jobs]
    infos = [dict(list(zip(key2sq_format.keys(), j))) for j in jobs]
    job2info = {info["jobid"]: info for info in infos}
    job2info = {j: info | dict(comment=try_parse_comment(info["comment"])) for j,info in job2info.items()}
    job2info = {j: info | dict(uid=info["comment"].get("uid", None)) for j,info in job2info.items()}

    # Now, use -O formatting to get other things
    # Need to make sure the allotted space is enough for the longest possible output
    key2sq_format = dict(
        jobid="JOBID:10",
        submit_time="SubmitTime:64",
        eligible_time="EligibleTime:64",
        stderr="StdErr:1024",
        stdout="StdOut:1024"
    )
    sq_key_str = f"{sep},".join(key2sq_format.values())
    sq_cmd = f"squeue {user_str} {account_str} -h -O \"{sq_key_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()
    if verbose:
        print(f"Running command: {sq_cmd}")
        print(f"Output:\n{sq}")
    
    jobs = sq.split("\n")
    jobs = [j.strip().split(sep) for j in jobs]
    jobs = [[j1.strip() for j1 in j] for j in jobs]
    infos = [dict(list(zip(key2sq_format.keys(), j))) for j in jobs]
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