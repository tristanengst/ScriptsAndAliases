import os
import os.path as osp
import subprocess

def get_cluster_type():
    """Returns a string for special host types, or None if they are not recognized."""
    h = os.uname()[1]
    if "nibi" in h:
        return "nibi"
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
def is_cc(): return get_cluster_type() in ["nibi", "narval", "cedar", "beluga", "graham"]
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











