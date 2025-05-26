"""Extracts all UIDs from a selection of text or from the entire squeue output."""

import argparse
import json
import os
import os.path as osp
import subprocess    

def print_found_uids(*, args, uids):
    uids = [f"*{u}*" for u in uids] if args.globs else uids
    print(" ".join(uids))

def extract_line_contents(*, s, key):
    lines = s.split("\n")
    for l in lines:
        if l.strip().startswith(key):
            return l.strip().replace(key, "")
    return None


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--jobs", default=None,
        help="Squeue output containing jobs. If None, computes for the current user")
    P.add_argument("--globs", default=1, choices=[0, 1], type=int,
        help="Print extracted job IDs with jobs")
    args = P.parse_args()

    if args.jobs is None:
        args.jobs = subprocess.getoutput(f"squeue -u $USER").split("\n")[1:]
    else:
        args.jobs = args.jobs.split("\n")
    
    line2uid = {l: None for l in args.jobs}

    # First, see if jobs have the UID recorded in their COMMENT
    for l in line2uid:
        scontrol_output = subprocess.getoutput(f"scontrol show job {l.split()[0]}")

        if "Comment=" in scontrol_output:
            comment = extract_line_contents(s=scontrol_output, key="Comment=")
            try:
                comment = json.loads(comment)
            except:
                comment = comment.replace("'", "\"")
                comment = json.loads(comment)
            line2uid[l] = comment["uid"]
            continue

        if "Command=" in scontrol_output:
            command = extract_line_contents(s=scontrol_output, key="Command=")
            with open(command, "r") as f:
                slurm_script = f.read()
            slurm_script = slurm_script.split()
            possible_uids = [slurm_script[idx+1] for idx,s in enumerate(slurm_script) if s == "--uid"]
            if len(possible_uids) == 0:
                print(f"Found zero UIDs for line {l}")
            elif len(possible_uids) == 1:
                line2uid[l] = possible_uids[0]
                continue
            else:
                print(f"Found multiple UIDs for line {l}: {possible_uids}")

    print_found_uids(args=args, uids=[u for u in line2uid.values()])