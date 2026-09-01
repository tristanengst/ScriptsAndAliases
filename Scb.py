import argparse
import os
import subprocess

from Utils import is_jobid, expand_jobid

default_get_all_jobs_cmd = "squeue -A rrg-keli_gpu -O 'JobArrayID:11,UserName:6,State:9,tres-per-node:17,TimeLeft:12,Reason:20,Name:.160'; squeue -A def-keli_gpu  -O 'JobArrayID:11,UserName:6,State:9,tres-per-node:17,TimeLeft:12,Reason:20,Name:.160'"

def lindex(s, substr):
    """Returns the leftmost index of [substr] in [s] or len(s) if not [substr] in [s]."""
    for idx in range(len(s)):
        if s[idx:].startswith(substr):
            return idx
    return len(s)

def split_command(s):
    """Returns string [s] as a list where each element is one bash command."""
    return [c.strip() for l in s.split("\n") for c in l.split(";")]

def get_all_jobs(args):
    """Returns a list of all jobs queued or running using --get_all_jobs_cmd.""" 
    cmd_list = split_command(args.get_all_jobs_cmd)
    all_jobs = ""
    for cmd in cmd_list:
        all_jobs += subprocess.run(cmd, capture_output=True, text=True, shell=True).stdout + "\n"

    all_jobs = [a for a in all_jobs.split("\n")]
    all_jobs = [a for a in all_jobs if len(a.split()) > 0 and a.split()[0].replace("_", "").isnumeric()]
    all_jobs = [a.split() for a in all_jobs]
    return all_jobs

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--job")
    P.add_argument("--get_all_jobs_cmd", default=default_get_all_jobs_cmd)
    args = P.parse_args()

    if is_jobid(args.job):
        # Expand bracketed array notation ("20942478_[3-5]") into individual task IDs so
        # each `scontrol show job` reports only the task(s) that were asked for.
        for jid in expand_jobid(args.job):
            os.system(f"scontrol show job {jid}")
    else:
        jobs = get_all_jobs(args)
        jobs = [j for j in jobs if args.job in j[-1]]
        for j in jobs:
            print(f"--- scontrol info for job {j[-1]}")
            os.system(f"scontrol show job {j[0][:lindex(j[0], '_')]}")