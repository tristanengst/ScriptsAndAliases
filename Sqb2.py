"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import shutil
import subprocess
import sys

from ShowCluster import Node
import Utils

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
        job_id = "" if j["JOBID"].startswith("next chunks") else str(j["JOBID"])

        s = []
        for c in col2max_chars:
            if j["JOBID"].startswith("__next chunks") and not c == "NAME":
                s.append(" " * col2max_chars[c])  # Empty space for next chunks
            elif j["JOBID"].startswith("__account") and c == "JOBID":
                s.append(f"{'JOBID':<{col2max_chars[c]}}")
            

            elif c == "NAME" and j["NAME"].startswith("next chunks"):
                to_print = f"------ {j['NAME']} ------"
                to_print = f"{to_print:^{col2max_chars[c]}}"
                s.append(to_print)

            elif c == "NAME" and j["NAME"] == "NAME" and Utils.is_cc():
                account = j["JOBID"].replace("__account ", "")
                to_print = f"------ {account} ------"
                to_print = f"{to_print:^{col2max_chars[c] - len(c) * 2}}"
                to_print = "NAME" + to_print
                s.append(to_print)
            
            else:
                s.append(f"{str(j[c]):<{col2max_chars[c]}}")
            
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

def start_time_to_comparable(start_time):
    """Returns [start_time] a comparable string."""
    if start_time[0].isnumeric():
        month = int(start_time[:start_time.index("-")])
        month = f"{month:02d}"  # Ensure month is two digits
        return month + start_time[start_time.index("-"):]
    else:
        return start_time

def job_dict_with_queue_time(jd):
    """Returns job dict [jd] with the queue time added."""
    if "ELIGIBLE_TIME" in jd and jd["ELIGIBLE_TIME"] in ["Unknown", "N/A"]:
        return jd | dict(QUEUE_TIME="N/A")
    elif "ELIGIBLE_TIME" in jd and jd["ELIGIBLE_TIME"]:
        eligible_time = datetime.strptime(jd["ELIGIBLE_TIME"], "%Y-%m-%dT%H:%M:%S")
    elif not "ELIGIBLE_TIME" in jd:
        raise NotImplementedError(f"job_dict={jd} missing 'ELIGIBLE_TIME' key")
    else:
        raise NotImplementedError(f"job_dict={jd} ")

    if jd["STATE"] in ["RUNNING", "COMPLETING"]:
        start_time = datetime.strptime(jd["START_TIME"], "%Y-%m-%dT%H:%M:%S")
    else:
        start_time = datetime.now()
    
    queue_time = start_time - eligible_time
    queue_time = f"{queue_time.total_seconds() / 3600:.2f}H"
    return jd | dict(QUEUE_TIME=queue_time) 

def job_dict_with_formatted_date_time(jd, *, key):
    """Returns job dict [jd] with the date/time key [key] formatted. This function should be run last!"""
    date_time = jd[key]

    # If [start_time] starts with four numbers, these are the year and are removed.
    if len(date_time) > 4 and date_time[0:4].isnumeric():
        date_time = date_time[5:]  # Remove the year and the dash after it
    
    if date_time[0].isnumeric():
        date_time_chars = []
        for c in date_time:
            if c.isnumeric() or c in "-:T":
                date_time_chars.append(c)
            else:
                print(f"Unexpected character in start time: {c}")
                break
        date_time = "".join(date_time_chars)
        date_time = date_time[:-3].replace("T", "-") # Exclude seconds
    elif date_time.startswith("N/A"):
        date_time = "N/A"
    else:
        assert 0, f"Unexpected start time format: {date_time}"

    return jd | {key: date_time}

def job_dict_with_formatted_time_delta(jd, key="TIME_LEFT"):
    """Returns job dict [jd] with the time delta key [key] formatted."""
    delta = jd[key]
    if "-" in delta:
        days, hhmmss = delta.split("-")
        hh, mm, ss = hhmmss.split(":")
        days, hh, mm, ss = int(days), int(hh), int(mm), int(ss)
        hh = days * 24 + hh
    else:
        if delta.count(":") == 2:
            hh, mm, ss = delta.split(":")
            hh, mm, ss = int(hh), int(mm), int(ss)
        elif delta.count(":") == 1:
            mm, ss = delta.split(":")
            hh, mm, ss = 0, int(mm), int(ss)
        else:
            raise ValueError(f"Unexpected time left format: {delta}")
    
    result = f"{mm:02}:{ss:02}" if hh == 0 else f"{hh}:{mm:02}:{ss:02}"
    result = " " * (9 - len(result)) + result

    return jd | {key: result}

def job_dict_with_formatted_resources(jd, num_nodes=1):
    """Returns job dict [jd] with the resources formatted."""
    known_gpu = ["h100", "a100", "l40s", "a40", "a5000", "v100"]
    
    gres_gpu = jd.get("Gres", "N/A")
    num_nodes = jd.get("NODES", num_nodes)
    
    if gres_gpu == "N/A":
        gpus = "N/A"
    else:
        gpus = gres_gpu.replace("gres/gpu:", "").replace("gres:gpu:", "").replace("gpu:", "").split(":")
        
        # If the last element is a GPU, no GPU was requested
        num_gpus = 0 if any([gpus[-1].startswith(g) for g in known_gpu]) else int(gpus[-1])
        gpu_type = None if len(gpus) < 2 else gpus[-2]
        gpus =  f"{num_gpus * int(num_nodes)}"

    return jd | dict(GPUS=gpus,)

def job_dict_with_formatted_reason(jd):
    """Returns job dict [jd] with the reason formatted."""
    reason = " ".join(jd["REASON"]) if isinstance(jd["REASON"], list) else jd["REASON"]
    reason = reason.split(":")[0].split(" ")[0].strip()  # Remove the first word and any colons
    return jd | dict(REASON=reason)

def job_dict_without_preempt_me_name(jd):
    """Returns job dict [jd] with the name without the 'preempt_me_' prefix."""
    name = jd["NAME"].replace("preempt_me_", "") if jd["NAME"].startswith("preempt_me_") else jd["NAME"]
    return jd | dict(NAME=name)

def jobs_data(*, account=None, cur_user=False, next_chunks=False, nodes=False, submit_time=False, eligible_time=False, queue_time=False, verbose=False):
    """Returns a (job2info, col_names) tuple where job2info is a dictionary mapping
    job IDs to info about their SLURM whatnot, and col_names is a list of column names
    indexing each value of [job2info].

    Args:
    account     -- SLURM account to get jobs for, or none for all applicable accounts
    cur_user    -- if True, only show jobs for the current user
    next_chunks -- if True, include next chunk jobs too
    nodes       -- if True, include the node list for all jobs
    submit_time -- if True, include the submit time for all jobs
    """
    job2info = Utils.get_slurm_status(cur_user=cur_user, account=account, verbose=(verbose > 1))

    if submit_time:
        job2info = {j: job_dict_with_formatted_date_time(v, key="SUBMIT_TIME") for j,v in job2info.items()}
    if eligible_time:
        job2info = {j: job_dict_with_formatted_date_time(v, key="ELIGIBLE_TIME") for j,v in job2info.items()}
    if queue_time:
        job2info = {j: job_dict_with_queue_time(v) for j,v in job2info.items()}

    job2info = {j: job_dict_with_formatted_resources(info) for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_date_time(info, key="START_TIME") for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_time_delta(info, key="TIME_LEFT") for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_reason(info) for j,info in job2info.items()}
    job2info = {j: job_dict_without_preempt_me_name(info) for j,info in job2info.items()}

    # Combine an abbreviated partition name with the user name on Solar
    if Utils.is_solar() and not cur_user:
        for jobid,info in job2info.items():
            partition = info["PARTITION"].replace("-short", "").replace("-long", "").replace("-lab", "").replace("cs-gpu-research", "cs-gpu-")
            job2info[jobid]["USER"] = f"{info['USER']}/{partition}"


    col_names = ["HOST" if Utils.is_solar() or nodes else None,
        "JOBID", "UID",
        "USER" if not cur_user else None,
        "STATE",
        "SUBMIT_TIME" if submit_time else None,
        "ELIGIBLE_TIME" if eligible_time else None,
        "QUEUE_TIME" if queue_time else None,
        "START_TIME", "GPUS", "NAME", "TIME_LEFT", "REASON",]
    col_names = [c for c in col_names if not c is None]        
    
    # On Solar, sort all the running jobs by the node name. The node name is printed
    # on the far left.
    if Utils.is_solar():
        running_jobs = [k for k,v in job2info.items() if v["STATE"] == "RUNNING"]
        other_jobs = [k for k,v in job2info.items() if not v["STATE"] == "RUNNING"]
        running_jobs.sort(key=lambda k: job2info[k]["HOST"])
        job2info = {j: job2info[j] for j in running_jobs + other_jobs}

    # On ComputeCanada, there may be duplicate UIDs as jobs pre-submit their next job
    # chunk. So, we will sort all of the duplicates below the rest. In this case, we
    # will sort the jobs so matching UIDs are grouped together and ordered by the
    # start time of the least-job ID of the next chunks.
    elif Utils.is_cc():
        uid2jobids = defaultdict(list)
        for idx,(jobid,info) in enumerate(job2info.items()):
            # Use indices so jobs without UIDs aren't impacted
            uid = info["UID"] if not info["UID"] is None else str(idx)
            uid2jobids[uid].append(jobid)
        
        least_job_ids = set([min(jobids) for _,jobids in uid2jobids.items()])
        if next_chunks:
            duplicate_job_ids = [j for j in job2info if not j in least_job_ids]
            duplicate_job_ids = sorted(duplicate_job_ids, key=lambda j: start_time_to_comparable(job2info[j]["START_TIME"]))
            duplicate_job_ids = sorted(duplicate_job_ids, key=lambda j: job2info[j]["UID"])

            job2info_main = {j: info for j,info in job2info.items() if j in least_job_ids}
            job2info_with_duplicates = {j: job2info[j] for j in duplicate_job_ids}

            # Insert an indicator into [job2info] giving where the next chunks start
            if len(job2info_with_duplicates) > 0:
                account_str = "" if account is None else f" ({account})"
                next_chunk = {f"__next chunks{account_str}": {c: f"next chunks{account_str}" if c == "NAME" else (f"__next chunks{account_str}" if c == "JOBID" else c) for c in col_names}}
                job2info = job2info_main | next_chunk | job2info_with_duplicates
        else:
            job2info = {j: info for j,info in job2info.items() if j in least_job_ids}

    return list(job2info.values()), col_names

def account_to_levelfs_record(account):
    """Returns a dictionary giving the group and user LevelFS for [account]."""
    s = subprocess.getoutput(f"sshare -l -A {account} --noheader")
    if len(s) > 0:
        group, user = s.split("\n")
        group = float(group.split()[6])
        user = float(user.split()[8])
        return dict(group=group, user=user)
        # return f"{account}={group:.2f} (user={user:.2f})"
    else:
        return dict(group=None, user=None)

def build_record(*, job_datas, account2lfs):
    """Returns a JSON record giving what was going on with the cluster when run."""
    import os, time # Imported here to be faster when not 
    return dict(date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        time=time.time(), # Maybe useful for easy sorting? Idk.
        account2lfs=account2lfs,
        job_data=job_datas,
        user=os.environ["USER"],  # Maybe useful if multiple people run this and end up with different LevelFS user fields?
        )


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-u", "--users", action="store_true", default=False,
        help="Show only jobs for all users")
    P.add_argument("-a", "-c", "--next_chunks", action="store_true", default=False,
        help="Show next chunk jobs too")
    P.add_argument("-n", "--nodes", action="store_true", default=False,
        help="Show show the node list for all jobs")
    P.add_argument("-s", "--submit_time", action="store_true", default=False,
        help="Show show the submit time for all jobs")
    P.add_argument("-e", "--eligible_time", action="store_true", default=False,
        help="Show show the submit time for all jobs")
    P.add_argument("-q", "--queue_time", action="store_true", default=False,
        help="Show show the submit time for all jobs")
    P.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2], type=int,
        help="Verbosity: 0=no output, 1=default output, 2=default+commands being run")
    P.add_argument("-r", "--record", default=False,
        help="Save outputs to this file for recording. 'default' saves to ~/.ClusterData/SqbOutputs/sqb_output_TIMESTR.json")
    args = P.parse_args()

    if Utils.is_solar():
        job_datas, colnames = jobs_data(cur_user=not args.users, account=None,
            next_chunks=args.next_chunks,
            nodes=args.nodes,
            submit_time=args.submit_time,
            eligible_time=args.eligible_time,
            queue_time=args.queue_time,
            verbose=args.verbose)
        job_datas = [{c: c for c in colnames}] + job_datas
    elif Utils.is_cc():
        accounts = ["rrg-keli_cpu", "def-keli_cpu", "rrg-keli_gpu", "def-keli_gpu"]
        job_datas = []
        for account in accounts:
            job_datas_account, colnames = jobs_data(account=account, cur_user=not args.users,
                next_chunks=args.next_chunks,
                nodes=args.nodes,
                submit_time=args.submit_time,
                eligible_time=args.eligible_time,
                queue_time=args.queue_time,
                verbose=args.verbose)
            if len(job_datas_account) > 0:
                colnames_job_data = {c: f"__account {account}" if c == "JOBID" else c for c in colnames}
                job_datas += [colnames_job_data] + job_datas_account
    else:
        # On workstations, the obvious equivalent is finding free GPUs.
        print(subprocess.getoutput("python ~/.ScriptsAndAliases/FindFreeGPUs.py --solar 0"))
        time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(time_str)
        sys.exit(0)
        
    col2max_chars = {c: len(c) for c in colnames}
    for job_data in job_datas:
        for c in colnames:
            col2max_chars[c] = max(col2max_chars[c], 0 if str(job_data["JOBID"]).startswith("__") else len(str(job_data[c])))
    col2max_chars = {c: mc for c,mc in col2max_chars.items()}

    # Try building the string representation for each job data.
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)

    # If including the full name would put the output over one line, first try
    # removing GPU and time specifications
    all_on_one_line = len(job_datas) == 0 or max([len(j["to_print"]) for j in job_datas]) <= shutil.get_terminal_size().columns
    if not all_on_one_line:
        col2max_chars["NAME"] = 0
        for job_data in job_datas:
            job_data["NAME"] = job_name_without_gpu_time_spec(job_data["NAME"])
            col2max_chars["NAME"] = max(col2max_chars["NAME"], len(job_data["NAME"]))

    # If any job name is still too long, re-order the output so the job name comes
    # last, and then make offending job names print on a line below the rest
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)
    all_on_one_line = len(job_datas) == 0 or max([len(j["to_print"]) for j in job_datas]) < shutil.get_terminal_size().columns
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
    
    if args.verbose:
        print(lines)

    # Now describe the overall cluster status or roughly how allocated it is
    time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    meta_str = f"--- Overall Cluster Status ({time_str}) ---\n"
    if Utils.is_cc():
        accounts = ["rrg-keli_gpu", "def-keli_gpu"]
        account2lfs = {a: account_to_levelfs_record(a) for a in accounts} # This is a record with saving
        account2lfs_str = {a: {k: f"{l:.2f}" if isinstance(l, float) else str(l) for k,l in lfs.items()} for a,lfs in account2lfs.items() if not lfs["group"] is None}
        level_fs_strs = [f"{a}={lfs['group']:} (user={lfs['user']})" for a,lfs in account2lfs_str.items()]
        level_fs_str = "\t|\tLevelFS: " + "\t".join(level_fs_strs)
        level_fs_str = level_fs_str.replace("_gpu", "")
        meta_str += level_fs_str
    
    meta_str +=  "\t|\t" + Node.cluster_stats_to_str()
    if args.verbose:
        print(meta_str)

    # If on ComputeCanada and --record is set, save the data that was computed so we
    # can reference it later.
    if Utils.is_cc() and args.record == "default":
        import os.path as osp # Imported here to be faster when not needed
        time_str = time_str.replace(":", "-")
        args.record = osp.join(osp.expanduser(f"~/.ClusterData"), "SqbOutputs", f"sqb_output_{time_str}.json")
    if Utils.is_cc() and args.record:
        import UtilsBase # Imported here to be faster when not needed
        job_datas = [jd for jd in job_datas if not jd["JOBID"].startswith("__")]
        record = build_record(job_datas=job_datas, account2lfs=account2lfs)
        _ = UtilsBase.atomic_save_lite(data=record, fname=args.record)

   




