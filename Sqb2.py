"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
import shutil
import subprocess

from ShowCluster import Node
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
    if start_time[0].isnumeric():
        # Extract up to the first non-numeric, non-dash, non-color character
        start_time_chars = []
        for c in start_time:
            if c.isnumeric() or c in "-:T":
                start_time_chars.append(c)
            else:
                break
        start_time = "".join(start_time_chars)
        return start_time[5:-3].replace("T", "-")
    elif start_time.startswith("N/A"):
        return "N/A"
    else:
        assert 0, f"Unexpected start time format: {start_time}"
          # Remove the year and seconds

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
    reason = reason.split(":")[0].split(" ")[0].strip()  # Remove the first word and any colons
    return reason

def jobs_data(*, account=None, cur_user=False, next_chunks=False):
    job2info = Utils.get_slurm_status(cur_user=cur_user, account=account)
    job2info = {k: v | dict(GPUS=format_gpu_str(v["Gres"], num_nodes=v.get("NODES", 1))) for k,v in job2info.items()}
    job2info = {k: v | dict(START_TIME=format_start_time_from_slurm(v["START_TIME"])) for k,v in job2info.items()}
    job2info = {k: v | dict(REASON=format_reason_from_slurm(v["REASON"])) for k,v in job2info.items()}
    job2info = {k: v | dict(NAME=v["NAME"].replace("preempt_me_", "") if v["NAME"].startswith("preempt_me_") else v["NAME"]) for k,v in job2info.items()}


    if cur_user:
        col_names = ["JOBID", "UID", "STATE", "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON"]
    else:
        col_names = ["JOBID", "UID", "USER", "STATE", "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON"]
    
    # On Solar, sort all the running jobs by the node name
    if Utils.is_solar():
        running_jobs = [k for k,v in job2info.items() if v["STATE"] == "RUNNING"]
        other_jobs = [k for k,v in job2info.items() if not v["STATE"] == "RUNNING"]
        running_jobs.sort(key=lambda k: job2info[k]["HOST"])
        job2info = {j: job2info[j] for j in running_jobs + other_jobs}
        col_names = ["HOST"] + col_names  # Add HOST to the beginning of the columns
    
    # On ComputeCanada, there may be duplicate UIDs as jobs pre-submit their next job
    # chunk. So, we will sort all of the duplicates below the rest. The smallest JobID
    # should be in the regular position in this case.
    elif Utils.is_cc():
        uid2jobids = defaultdict(list)
        for idx,(jobid,info) in enumerate(job2info.items()):
            # Use indices so jobs without UIDs aren't impacted
            uid = info["UID"] if not info["UID"] is None else str(idx)
            uid2jobids[uid].append(jobid)
        
        least_job_ids = set([min(jobids) for _,jobids in uid2jobids.items()])
        if next_chunks:
            job2info_main = {j: info for j,info in job2info.items() if j in least_job_ids}
            job2info_with_duplicates = {j: info for j,info in job2info.items() if not j in least_job_ids}
            job2info = job2info_main | job2info_with_duplicates
        else:
            job2info = {j: info for j,info in job2info.items() if j in least_job_ids}

    return list(job2info.values()), col_names

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-u", "--cur_user", action="store_true", default=True,
        help="Show only jobs for the current user")
    P.add_argument("-a", "--all", action="store_true",
        help="Show next chunk jobs too")
    args = P.parse_args()

    Node.print_cluster_stats(Node.get_node_list())

    if Utils.is_solar():
        job_datas, colnames = jobs_data(cur_user=args.cur_user, account=None)
        job_datas = [{c: c for c in colnames}] + job_datas
    else:
        accounts = ["rrg-keli_cpu", "def-keli_cpu", "rrg-keli_gpu", "def-keli_gpu"]
        job_datas = []
        for account in accounts:
            job_datas_account, colnames = jobs_data(account=account, cur_user=args.cur_user, next_chunks=args.all)
            if len(job_datas_account) > 0:
                job_datas += [{c: c for c in colnames}] + job_datas_account
        
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
