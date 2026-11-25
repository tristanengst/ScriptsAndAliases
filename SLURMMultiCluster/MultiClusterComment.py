"""The rules for multicluster job comments (see README.md), shared by the controller and the
in-job check so the two can't disagree. Comments themselves are parsed and written with
UtilsBase.job_comment_to_dict() and dict_to_job_comment().

Uses only the standard library, so the in-job check can import it under whichever python is on
PATH.
"""
REQUIRED_KEYS = ["uid", "exp_folder", "mc_cont", "mc_time"]

RUN = "run"
REQUEUE = "requeue"             # a sibling is running: stand down with `scontrol requeue`
REQUEUEHOLD = "requeuehold"     # the experiment finished: stand down with `scontrol requeuehold`
STAND_DOWN_VERDICTS = [REQUEUE, REQUEUEHOLD] # Each is also the scontrol command that does it

def is_flag_set(d, key):
    """Returns if the flag [key] of comment dict [d] is set (1 or true)."""
    return d.get(key, 0) in [1, True]

def is_multicluster(d):
    """Returns if comment dict [d] is a multicluster job's: it has every REQUIRED_KEYS, and its
    exp_folder starts with ~/, so it resolves on any cluster's login nodes.
    """
    return (all(k in d for k in REQUIRED_KEYS)
        and isinstance(d["exp_folder"], str) and d["exp_folder"].startswith("~/"))

def is_ignored(d):
    """Returns if the multicluster job with comment dict [d] is left out entirely: it's not a
    sibling of anything, and nothing is done to it.
    """
    return is_flag_set(d, "mc_ignore")

def is_protected(d):
    """Returns if the multicluster job with comment dict [d] is protected: it counts as a sibling
    (it can make others stand down, and finish their experiment), but it's never stood down or
    held itself.
    """
    return is_flag_set(d, "mc_protected")
