"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
import shutil
import subprocess

import ExtractUIDs
import Utils

def extract_gpu_str(gres_gpu, nodes):
    s = gres_gpu.replace("gres/gpu:", "").replace("gres:gpu:", "").split(":")[-1]
    if s.isnumeric():
        return str(int(s) * int(nodes))
    else:
        return gres_gpu

def job_datas_with_to_prints(*, job_datas, col2max_chars):
    """Returns [job_datas] with a new key "to_print" that contains the string that
    should be printed to the terminal added for each job data.
    """
    all_time_left_prefixed_zero = all([jd["TIME_LEFT"].startswith("0") or jd["TIME_LEFT"] == "TIME_LEFT" for jd in job_datas])
    if all_time_left_prefixed_zero:
        for jd in job_datas:
            jd["TIME_LEFT"] = jd["TIME_LEFT"][1:] if jd["TIME_LEFT"].startswith("0") else jd["TIME_LEFT"]

    all_start_times_prefixed_zero = all([jd["START_TIME"].startswith("0") or jd["START_TIME"] in ["START_TIME", "N/A"] for jd in job_datas])
    if all_start_times_prefixed_zero:
        for jd in job_datas:
            jd["START_TIME"] = jd["START_TIME"][1:] if jd["START_TIME"].startswith("0") else jd["START_TIME"]

    for j in job_datas:
        s = []
        for c in col2max_chars:
            to_print = str(j[c])
            s.append(f"{to_print:<{col2max_chars[c]}}")
        s = "  ".join(s)
        j["to_print"] = s
    return job_datas

def job_name_without_gpu_time_spec(jn):
    """Returns job name [jn] without the GPU and time specification if it exists."""
    import re
    pattern = re.compile(r"^(.*?)(-?gpus\d+-\d{3}H\d{2}M)$")
    match = pattern.match(jn)
    jn = match.group(1) if match else jn
    jn = jn.replace("preempt_me_", "") if jn.startswith("preempt_me_") else jn # On Solar, this prefix is for only other users
    return jn

def format_start_time_from_slurm(start_time):
    """Returns the start time as given by squeue better formatted."""
    if start_time == "N/A":
        return "N/A"
    else:
        return start_time[5:-3].replace("T", "-")  # Remove the year and seconds

def format_gpu_str(gres_gpu, num_nodes=1):
    """Returns the GPU string formatted from the SLURM output."""
    if gres_gpu == "N/A":
        return "N/A"
    gpus = gres_gpu.replace("gres/gpu:", "").replace("gres:gpu:", "").replace("gpu:", "").split(":")
    num_gpus = int(gpus[-1])
    gpu_type = None if len(gpus) < 2 else gpus[-2]

    if not isinstance(num_nodes, int):
        num_nodes = int(num_nodes)
    return f"{num_gpus * num_nodes}" # We could do something fancier, but right now there's no ambiguity it could resolve

def format_reason_from_slurm(reason):
    """Returns the reason for the job in a more readable format."""
    reason = " ".join(reason) if isinstance(reason, list) else reason
    if reason == "Nodes required for job are DOWN, DRAINED or reserved for jobs in higher priority partitions":
        return "Nodes required"
    else:
        return reason.strip()

def jobs_data_solar(cur_user=False):
    user_str = "-u $USER" if cur_user else ""
    s = f"squeue {user_str} -O 'NodeList:100,JobArrayID:.100,State:.100,tres-per-node:.100,Account:.100,Partition:.100,Name:.250,TimeLeft:.30,NumNodes:.100,StartTime:.20,Reason:.15' --sort N --noheader"

    jobs = subprocess.getoutput(s).strip()
    job_datas = []
    if not len(jobs) == 0:

        jobs = jobs.split("\n")

        for j in jobs:
            j = j.strip()
            j_list = [j.strip() for j in j.split()]

            job_datas.append(dict(
                NODES=j_list[0],
                JOBID=j_list[1],
                UID=ExtractUIDs.jobid_to_uid(j_list[1], default=None, cur_user_only=True),
                STATE=j_list[2],
                START_TIME=format_start_time_from_slurm(j_list[9]),
                GPUS=format_gpu_str(j_list[3], num_nodes=j_list[8]),
                ACCOUNT=j_list[4],
                PARTITION=j_list[5],
                NAME=j_list[6],
                TIME_LEFT=j_list[7],
                REASON=format_reason_from_slurm(j_list[10:]),
            ))
    if cur_user:
        colnames = ["NODES", "JOBID", "UID", "STATE", "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON"]
    else:
        colnames = ["NODES", "JOBID", "UID", "STATE", "START_TIME", "GPUS", "ACCOUNT", "PARTITION", "NAME", "TIME_LEFT", "REASON"]

    return job_datas, colnames 

def jobs_data_cc(*, account, cur_user=False):
    user_str = "-u $USER" if cur_user else ""
    account_str = f"-A {account}"

    s = f"squeue {user_str} {account_str} -O 'JobArrayID:.100,UserName:.100,State:.100,tres-per-node:.100,TimeLeft:.100,NumNodes:.10,Name:.250,StartTime:.100,Reason:.100,' --noheader"
    jobs = subprocess.getoutput(s).strip()
    job_datas = []
    if not len(jobs) == 0:
        jobs = jobs.split("\n")
        for j in jobs:
            j = j.strip()
            j_list = [j.strip() for j in j.split()]

            job_datas.append(dict(
                JOBID=j_list[0],
                UID=ExtractUIDs.jobid_to_uid(j_list[0], default=None, cur_user_only=True),
                USER=j_list[1],
                STATE=j_list[2],
                GPUS=format_gpu_str(j_list[3], num_nodes=j_list[5]),
                TIME_LEFT=Utils.format_time_delta(j_list[4]),
                NAME=j_list[6],
                START_TIME=format_start_time_from_slurm(j_list[7]),
                REASON=format_reason_from_slurm(j_list[8])
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
        job_datas, colnames = jobs_data_solar(cur_user=args.cur_user)
        job_datas = [{c: c for c in colnames}] + job_datas
    else:
        job_datas_rrg, colnames = jobs_data_cc(account="rrg-keli_gpu", cur_user=args.cur_user)
        job_datas_rrg = [{c: c for c in colnames}] + job_datas_rrg
        job_datas_rrg[0]["JOBID"] = f"JOBID"
        job_datas_def, colnames = jobs_data_cc(account="def-keli_gpu", cur_user=args.cur_user)
        job_datas_def = [{c: c for c in colnames}] + job_datas_def
        job_datas_def[0]["JOBID"] = f"JOBID"

        job_datas = job_datas_rrg + job_datas_def
        
        
    col2max_chars = {c: len(c) for c in colnames}
    for job_data in job_datas:
        for c in colnames:
            col2max_chars[c] = max(col2max_chars[c], len(str(job_data[c])))
    col2max_chars = {c: mc for c,mc in col2max_chars.items()}

    # Try building the string representation for each job data.
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)

    # If including the full name would put the output over one line, first try
    # removing GPU and time specifications
    all_on_one_line = max([len(j["to_print"]) for j in job_datas]) <= shutil.get_terminal_size().columns
    if not all_on_one_line:
        col2max_chars["NAME"] = 0
        for job_data in job_datas:
            job_data["NAME"] = job_name_without_gpu_time_spec(job_data["NAME"])
            col2max_chars["NAME"] = max(col2max_chars["NAME"], len(job_data["NAME"]))

    # If any job name is still too long, re-order the output so the job name comes
    # last, and then make offending job names print on a line below the rest
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)
    all_on_one_line = max([len(j["to_print"]) for j in job_datas]) < shutil.get_terminal_size().columns
    if not all_on_one_line:
        col_names = [c for c in colnames if not c == "NAME"] + ["NAME"]
        col2max_chars = {c: col2max_chars[c] for c in col_names}
        
        # Have to work with explicitly the name column, since it will have been padded
        # so that short job names have many characters
        other_chars = sum([col2max_chars[c] for c in col_names if not c == "NAME"])
        max_name_chars = shutil.get_terminal_size().columns - other_chars - 2 * (len(col_names)-1)  # 2 for the spaces between columns
        for j in job_datas:
            if len(j["NAME"].strip()) > max_name_chars:
                j["NAME"] = "\n\t\t" + j["NAME"] + "\n"
        
        # Exclude jobs whose names are on a new line from the length calculation
        col2max_chars["NAME"] = 0
        for job_data in job_datas:
            job_name_ = "" if job_data["NAME"].startswith("\n\t\t") else job_data["NAME"].strip()
            col2max_chars["NAME"] = max(col2max_chars["NAME"], len(job_name_))
        
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)
    lines = "\n".join([j["to_print"] for j in job_datas])
    print(lines)
