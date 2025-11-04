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
    elif h.startswith("vulcan") or h.startswith("rack"):
        return "vulcan"
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
    elif h == ("cs-star") or h.startswith("cs-v"):
        return "solar"
    elif h == "cs-star1":
        return "solar1"
    else:
        h_ = "-".join(h.split("-")[:2])
        return os.environ.get("CLUSTER_TYPE", "cs-apex")

def is_solar(): return get_cluster_type() in ["solar", "solar1"]
def is_cc(): return get_cluster_type() in ["nibi", "narval", "cedar", "beluga", "graham", "rorqual", "trillium", "fir", "vulcan"]
def is_slurm(): return is_solar() or is_cc()
def is_workstation(): return not is_solar() and not is_cc()

cluster2info = dict(
    nibi=dict(accounts=["rrg-keli_gpu", "def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    trillium=dict(accounts=["def-keli"], conf_file=None, partitions_startswith=["compute", "compute_full_node"]),
    fir=dict(accounts=["def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    rorqual=dict(accounts=["def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    narval=dict(accounts=["def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    cedar=dict(accounts=["def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    beluga=dict(accounts=["def-keli_gpu"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    vulcan=dict(accounts=["aip-keli"], conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
    solar=dict(accounts=None, conf_file="/etc/slurm/slurm.conf", partitions_startswith=["cs-gpu-research"]),
    solar1=dict(accounts=None, conf_file="/etc/slurm/slurm.conf", partitions_startswith=["cs-gpu-research"]),
)
cluster2info = {k: argparse.Namespace(**v) for k,v in cluster2info.items()}

def get_slurm_status(cur_user=False, account=None, verbose=False,
    keys=["jobid", "user", "state", "start_time", "time_left", "time_limit", "gres",
        "nodes", "name", "reason", "account", "partition", "host", "exclude",
        "comment", "submit_time", "eligible_time", "stderr", "stdout", "uid",
        "partition", "dependency"],
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
        accounts = cluster2info[get_cluster_type()].accounts
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
        comment=f"%k",
        dependency=f"%E",
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
    
    sep = " , " if get_cluster_type() == "solar1" else sep # TODO: Why is this needed on Solar1 but not other clusters?
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

