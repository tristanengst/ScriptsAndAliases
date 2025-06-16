"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
import shutil
import subprocess

import ExtractUIDs
import Utils

def jobs_data_solar(cur_user=False):
    user_str = "-u $USER" if cur_user else ""
    s = f"squeue {user_str} -O 'NodeList:20,JobArrayID:.6,State:.9,tres-per-node:.12,Account:.100,Partition:.16,Name:.250,TimeLeft:.12,NumNodes:.4,StartTime:.20,Reason:.15' --sort N --noheader"

    jobs = subprocess.getoutput(s).strip()
    job_datas = []
    if not len(jobs) == 0:
        jobs = jobs.split("\n")
        for j in jobs:
            job_datas.append(dict(
                NODES=j.strip().split()[0],
                JOBID=j.strip().split()[1],
                UID=ExtractUIDs.jobid_to_uid(j.strip().split()[1], default=None, cur_user_only=False),
                STATE=j.strip().split()[2],
                START_TIME=j.strip().split()[9],
                GPUS=str(int(j.strip().split()[3].replace("gres/gpu:", "").replace("gres:gpu:", "").split(":")[-1]) * int(j.strip().split()[8])),
                ACCOUNT=j.strip().split()[4],
                PARTITION=j.strip().split()[5],
                NAME=j.strip().split()[6],
                TIME_LEFT=j.strip().split()[7],
                REASON=" ".join(j.strip().split()[10:]),
            ))
    if cur_user:
        colnames = ["NODES", "JOBID", "UID", "STATE", "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON"]
    else:
        colnames = ["NODES", "JOBID", "UID", "STATE", "START_TIME", "GPUS", "ACCOUNT", "PARTITION", "NAME", "TIME_LEFT", "REASON"]

    return job_datas, colnames 

def jobs_data_cc(*, account, cur_user=False):
    user_str = "-u $USER" if cur_user else ""
    account_str = f"-A {account}"

    s = f"squeue {user_str} {account_str} -O 'JobArrayID:11,UserName:6,State:9,tres-per-node:17,TimeLeft:.12,NumNodes:.4,Name:.250,StartTime:.15,Reason:.15,' --noheader"
    jobs = subprocess.getoutput(s).strip()

    job_datas = []
    if not len(jobs) == 0:
        jobs = jobs.split("\n")
        for j in jobs:
            job_datas.append(dict(
                JOBID=j.strip().split()[0],
                UID=ExtractUIDs.jobid_to_uid(j.strip().split()[0], default=None, cur_user_only=False),
                USER=j.strip().split()[1],
                STATE=j.strip().split()[2],
                GPUS=str(int(j.strip().split()[3].replace("gres/gpu:", "").replace("gres:gpu:", "").split(":")[-1]) * int(j.strip().split()[5])),
                TIME_LEFT=j.strip().split()[4],
                NAME=j.strip().split()[6],
                START_TIME=j.strip().split()[7],
                REASON=" ".join(j.strip().split()[8:]),
            ))

    if cur_user:
        col_names = ["JOBID", "UID", "STATE", "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON"]
    else:
        col_names = ["JOBID", "UID", "USER", "STATE", "START_TIME", "GPUS", "TIME_LEFT", "NAME", "REASON"]
    
    return job_datas, col_names

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--cur_user", choices=[0, 1], type=int, default=1,
        help="Show only jobs for the current user")
    args = P.parse_args()

    if Utils.is_solar():
        job_datas, colnames = jobs_data_solar()
        job_datas = colnames + {c: c for c in colnames}
    else:
        job_datas_rrg, colnames = jobs_data_cc(account="rrg-keli_gpu", cur_user=args.cur_user)
        job_datas_rrg = job_datas_rrg + [{c: c for c in colnames}]
        job_datas_def, colnames = jobs_data_cc(account="def-keli_gpu", cur_user=args.cur_user)
        job_datas_def = job_datas_def + [{c: c for c in colnames}]

        job_datas = job_datas_rrg + job_datas_def
        
        
    col2max_chars = {c: len(c) for c in colnames}
    for job_data in job_datas:
        for c in colnames:
            col2max_chars[c] = max(col2max_chars[c], len(str(job_data[c])))
    col2max_chars = {c: mc+1 for c,mc in col2max_chars.items()}

    # If including the full name would put the output over one line, then move it to a new line below the rest of the output
    other_width = sum([col2max_chars[c] for c in col2max_chars if not c == "NAME"])
    terminal_width = shutil.get_terminal_size().columns
    possible_name_width = terminal_width - other_width - 1
    if col2max_chars["NAME"] > possible_name_width:
        col_names = [c for c in colnames if not c == "NAME"] + ["NAME"]
        for j in job_datas:
            j["NAME"] = "\n\t" + j["NAME"]

    lines = []
    for j in job_datas:
        s = []
        for c in colnames:
            s.append(f"{j[c]:<{col2max_chars[c]}}")

        lines.append("  ".join(s))
    
    lines = "\n".join(lines)
    print(lines)