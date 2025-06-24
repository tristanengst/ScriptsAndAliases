"""Extracts all UIDs from a selection of text or from the entire squeue output."""

import argparse
import json
import os
import os.path as osp
import subprocess    

def print_found_uids(*, args, uids):
    uids = [f"*{u}*" for u in uids] if args.globs else uids
    print(" ".join(uids))

def extract_line_contents(*, s, key, remove_parenthetical=True):
    key = key.replace("=", "")
    one_line_keys = ["Command", "Comment", ] # Non-exhaustive list of keys that are expected to be on one line
    lines = s.split("\n")
    for l in lines:
        l = l.strip()
        if any([l.startswith(o) for o in one_line_keys]):
            k,v = l.split("=")
            if k == key:
                return (v[:v.index("(")].strip() if remove_parenthetical and "(" in v else v.strip())
        else:
            kvs_in_line = l.split()
            kvs = [kv.split("=") for kv in kvs_in_line if "=" in kv]

        # Hack
        for kv in kvs:
            if len(kv) == 2:
                k,v = kv
                if k == key:
                    if remove_parenthetical and "(" in v:
                        return v[:v.index("(")].strip()
                    else:
                        return v.strip()
    return ""

def jobid_to_uid(jobid, default=None, cur_user_only=True):
    """Returns the UID of the job with [jobid].
    
    Args:
    jobid                   -- job ID to find the UID for
    default                 -- if not job ID is found, return this
    other_users_quiet_fail  -- for other users' jobs for which no UID is found, quiet
    """
    def print_message(s):
        user = extract_line_contents(s=scontrol_output, key="UserId=")
        user = user[:user.index("(")] if "(" in user else user
        if user == os.environ["USER"] or not cur_user_only:
            print(s)

    scontrol_output = subprocess.getoutput(f"scontrol show job {jobid}")
    
    # This is the most deliberate way to store and return job UIDs
    if "Comment=" in scontrol_output:
            comment = extract_line_contents(s=scontrol_output, key="Comment=")
            try:
                comment = json.loads(comment)
            except:
                comment = comment.replace("'", "\"")
                try:
                    comment = json.loads(comment)
                except:
                    print_message(f"Failed to parse comment for job {jobid}: {comment}")
                    return default
            return comment["uid"]
    
    elif "Command=" in scontrol_output:
        command = extract_line_contents(s=scontrol_output, key="Command=")
        if osp.exists(command) and not command == "/bin/sh":
            with open(command, "r") as f:
                slurm_script = f.read()
            slurm_script = slurm_script.split()
            possible_uids = [slurm_script[idx+1] for idx,s in enumerate(slurm_script) if s == "--uid"]
            if len(possible_uids) == 0:
                print_message(f"Found zero UIDs for line {slurm_script}")
            elif len(possible_uids) == 1:
                line2uid[l] = possible_uids[0]
                return possible_uids[0]
            else:
                print_message(f"Found multiple UIDs for line {l}: {possible_uids}")
        else:
            print_message(f"Command file {command} does not exist for job {jobid}")
            return default
    else:
        print_message(f"Job {jobid} does not have a Comment or Command field in scontrol output")
        return default


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
        line2uid[l] = jobid_to_uid(jobid=l.split()[0], default=None, cur_user_only=True)

    print_found_uids(args=args, uids=[u for u in line2uid.values()])
