#!/usr/bin/env python3
"""Run at the start of a SLURM job: if the job is a multicluster job (see README.md) whose
comment carries a fresh verdict from the controller to stand down, carries it out on the job
itself. Otherwise does nothing, and the job carries on.

Put this as the first command of the job script:
    python ~/.ScriptsAndAliases/SLURMMultiCluster/SLURMMultiClusterCheck.py || true

The job ($SLURM_JOB_ID) stands down only if its comment (JSON) is a multicluster job's (see
MultiClusterComment.is_multicluster()), neither 'mc_ignore' nor 'mc_protected' is set, 'mc_cont'
is 'requeue' (a sibling is running: `scontrol requeue`) or
'requeuehold' (a sibling finished: `scontrol requeuehold`), and 'mc_time' (Unix seconds) is
less than --mc_time_stale_after_seconds old (the controller renews it whenever it reads the
cluster, so a verdict this old means the controller has stopped, and the job runs). After
standing down it waits --kill_after_seconds for SLURM to end the job, so the job script doesn't reach its next line first; if the job is still
alive then, it exits and the job runs. Every other case, including any error, exits at once
and the job runs: the bias is toward running. `|| true` keeps job scripts using `set -e`
running when this fails.

Uses only the standard library, UtilsBase and MultiClusterComment, so it runs under the system
python on PATH, before any environment is set up.
"""
import argparse
import os
import os.path as osp
import subprocess
import sys
import time

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import UtilsBase
from UtilsBase import twrite
import MultiClusterComment as MCC

def get_comment(*, job):
    """Returns the comment of job [job], as squeue prints it."""
    result = subprocess.run(["squeue", "-h", "-j", job, "-o", "%k"], capture_output=True, text=True, check=True)
    return result.stdout.strip()

def get_stand_down_action(*, comment, mc_time_stale_after_seconds, now):
    """Returns the scontrol command ('requeue' or 'requeuehold') the job with comment
    [comment] should run on itself at Unix time [now], or None if it should run, with the
    reason.
    """
    try:
        d = UtilsBase.job_comment_to_dict(comment)
    except ValueError:
        return None, "not a multicluster job (comment isn't JSON)"
    if not MCC.is_multicluster(d):
        return None, "not a multicluster job"
    elif MCC.is_ignored(d):
        return None, "mc_ignore is set"
    elif MCC.is_protected(d):
        return None, "mc_protected is set"
    elif d["mc_cont"] == MCC.RUN:
        return None, "verdict is 'run'"
    elif not d["mc_cont"] in MCC.STAND_DOWN_VERDICTS:
        return None, f"unknown verdict mc_cont={d['mc_cont']!r}"
    elif not isinstance(d["mc_time"], (int, float)) or isinstance(d["mc_time"], bool):
        return None, f"verdict {d['mc_cont']!r} has no time (mc_time={d['mc_time']!r})"
    elif not abs(now - d["mc_time"]) < mc_time_stale_after_seconds:
        return None, f"verdict {d['mc_cont']!r} is stale: mc_time={d['mc_time']} is {now - d['mc_time']:.0f} s from now, stale after {mc_time_stale_after_seconds} s"
    else:
        return d["mc_cont"], f"verdict {d['mc_cont']!r} from {now - d['mc_time']:.0f} s ago"

def get_args():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--mc_time_stale_after_seconds", type=float, default=1200,
        help="Seconds after which mc_time is stale, so a verdict to stand down is ignored and the job runs (the controller renews mc_time every poll)")
    P.add_argument("--kill_after_seconds", type=float, default=120,
        help="After standing down, seconds to wait for SLURM to end the job before letting it run")
    return P.parse_args()

if __name__ == "__main__":
    args = get_args()
    job = os.environ["SLURM_JOB_ID"]
    action, reason = get_stand_down_action(comment=get_comment(job=job),
        mc_time_stale_after_seconds=args.mc_time_stale_after_seconds, now=time.time())
    if action is None:
        twrite(f"[SLURMMultiClusterCheck] Job {job} runs: {reason}")
        sys.exit(0)

    twrite(f"[SLURMMultiClusterCheck] Job {job} stands down ({reason}): scontrol {action} {job}")
    sys.stdout.flush() # Output to a file is buffered, and the job may be ended before it's written
    result = subprocess.run(["scontrol", action, job], capture_output=True, text=True)
    if result.returncode != 0:
        twrite(f"[SLURMMultiClusterCheck] scontrol {action} failed, so job {job} runs: {(result.stdout + result.stderr).strip()}")
        sys.exit(1)
    time.sleep(args.kill_after_seconds)
    twrite(f"[SLURMMultiClusterCheck] Job {job} wasn't ended within {args.kill_after_seconds} s, so it runs")
    sys.exit(1)
