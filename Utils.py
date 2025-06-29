import os
import os.path as osp
import subprocess

def get_cluster_type():
    """Returns a string for special host types, or None if they are not recognized."""
    h = os.uname()[1]
    if "nibi" in h:
        return "nibi"
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
def is_cc(): return get_cluster_type() in ["nibi", "narval", "cedar", "beluga", "graham", "rorqual"]
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

def get_slurm_status(cur_user=False, account=None):
    """Returns a dictionary describing the entire state what's running. Strings are
    not processed or reformatted in any way.
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
    keys = ["JOBID", "USER", "STATE", "START_TIME", "TIME_LEFT", "Gres", "NODES", "NAME", "REASON", "ACCOUNT", "PARTITION", "HOST"]
    
    squeue = f"squeue {user_str} {account_str} -h -o \"%i|%u|%T|%S|%L|%b|%D|%j|%r|%a|%P|%N\""
    squeue = subprocess.getoutput(squeue).strip()

    if squeue == "":
        return dict()

    jobs = squeue.split("\n")
    jobs = [j.strip().split("|") for j in jobs]
    jobid_info = [list(zip(keys, j)) for j in jobs]
    job2info = [dict(j) for j in jobid_info]
    job2info = {j["JOBID"]: j for j in job2info}

    # Now get comments
    squeue = f"squeue {user_str} {account_str} -h -O JobID:10,Comment:.300"
    squeue = subprocess.getoutput(squeue).strip()
    jobs = squeue.split("\n")
    jobs = [j.strip().split() for j in jobs]
    job2comment = {j[0]: " ".join(j[1:]) for j in jobs}
    job2comment = {j: try_parse_comment(c) for j,c in job2comment.items()}
    job2info = {j: info | job2comment.get(j, {}) for j,info in job2info.items()}

    # Now, ensure UIDs are recorded at a top level for convenience
    job2info = {j: info | dict(UID=info.get("uid", None)) for j,info in job2info.items()}
    return job2info