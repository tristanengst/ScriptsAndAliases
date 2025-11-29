"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
import copy
from datetime import datetime
from functools import partial
import glob
import io
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time

# import ClusterInfo
import ClusterInfo2
import FileFinding
import MachineInfo
from ShowCluster import Node
import Utils
import UtilsBase
from UtilsBase import twrite, colorize, decolorize, get_color_scale

##### Miscellaneous ##################################################################


######################################################################################
######################################################################################
######################################################################################
def colorize_submit_times(job_infos):
    """Returns each job info in [job_infos] with the time left colorized."""
    cutoff_values = [0.25, 0.5, 1, 2, 3, 5, 7, 12, 24, 48] # In hours
    color_scale = get_color_scale(
        start="blue",
        mid="purple",
        end="red",
        light_bias=2,
        num_colors=len(cutoff_values)+1)

    def colorize_submit_time(ji):
        if ji.time_left is None or not any([vc.isdigit() for vc in ji.submit_time]):
            return ji
        else:
            time_ago = UtilsBase.hours_since_time(ji.submit_time)
            idxs = [idx for idx,c in enumerate(cutoff_values) if time_ago <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            submit_time = colorize(ji.submit_time, color=color_scale[min_valid_idx])
            return UtilsBase.updated_namespace(ji, submit_time_color=submit_time)

    return [colorize_submit_time(ji) for ji in job_infos]

def colorize_time_lefts(job_infos, cur_user=True):
    """Returns each job info in [job_infos] with the time left colorized."""
    cutoff_values = [0.25, 0.5, 1, 2, 3, 5, 7, 12, 24, 48] # In hours
    color_scale = get_color_scale(
        start="red",
        mid="purple",
        end="blue",
        light_bias=2,
        num_colors=len(cutoff_values)+1)

    def colorize_time_left(ji):
        if (ji.time_left in ["N/A", None]
            or not "user" in ji
            or (cur_user and not ji.user == os.environ["USER"])
            or not decolorize(ji.state) in ["RUNNING", "COMPLETING"]):
            return ji
        else:
            time_left_hours = UtilsBase.time_to_hours(ji.time_left)
            if time_left_hours <= 1:
                idxs = [idx for idx,c in enumerate(cutoff_values) if time_left_hours <= c]
                min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            else:
                frac_remaining = time_left_hours / UtilsBase.time_to_hours(ji.time_limit)
                min_valid_idx = min(int(frac_remaining * 8 + 3), len(cutoff_values))

            time_left = colorize(ji.time_left, color=color_scale[min_valid_idx])
            return UtilsBase.updated_namespace(ji, time_left_color=time_left)

    return [colorize_time_left(ji) for ji in job_infos]

def colorize_start_times(job_infos):
    """Returns each job info in [job_infos] with the start time colorized."""
    cutoff_values = [0.25, 1, 3, 6, 12, 18, 24, 36, 48, 72] # In hours
    color_scale = get_color_scale(
        start="blue",
        mid="purple",
        end="red",
        light_bias=2,
        num_colors=len(cutoff_values)+1)

    def colorize_start_time(ji):
        if ji.name == "HeldToProvideLevelFSEstimate":
            return ji
        elif ji.start_time == "N/A":
            start_time = colorize(ji.start_time, color=color_scale[-1])
            return UtilsBase.updated_namespace(ji, start_time_color=start_time)
        elif ji.start_time is None or not any([vc.isdigit() for vc in ji.start_time]):
            return ji
        else:
            time_in_future = (UtilsBase.time_stamp_to_datetime(ji.start_time) - datetime.now()).total_seconds() / 3600
            idxs = [idx for idx,c in enumerate(cutoff_values) if time_in_future <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            start_time = colorize(ji.start_time, color=color_scale[min_valid_idx])
            return UtilsBase.updated_namespace(ji, start_time_color=start_time)

    return [colorize_start_time(ji) for ji in job_infos]

def colorize_reasons(job_infos):
    """Returns each job info in [job_infos] with the reason colorized."""
    def colorize_reason(ji):
        if ji.jobid.startswith("__"):
            return ji
        elif ji.state in ["RUNNING", "COMPLETING"]:
            reason = colorize(ji.reason, color="green")
            return UtilsBase.updated_namespace(ji, reason_color=reason)
        elif ji.state == "PENDING" and (ji.reason.startswith("Priority")
            or ji.reason.startswith("ReqNodeNotAvail")
            or ji.reason.startswith("Resources")
            or ji.reason.startswith("Nodes")
            or ji.reason.startswith("None")):
            reason = colorize(ji.reason, color="orange")
            return UtilsBase.updated_namespace(ji, reason_color=reason)
        elif ji.state == "PENDING" and (ji.reason.startswith("Dependency") or ji.reason.startswith("After")):
            reason = colorize(ji.reason, color="red1")
            return UtilsBase.updated_namespace(ji, reason_color=reason)
        elif ji.state == "PENDING" and ji.reason.startswith("JobHeld"):
            reason = colorize(ji.reason, color="red1")
            return UtilsBase.updated_namespace(ji, reason_color=reason)
        else:
            reason = colorize(ji.reason, color="red")
            return UtilsBase.updated_namespace(ji, reason_color=reason)
    return [colorize_reason(ji) for ji in job_infos]

def colorize_queues(job_infos):
    """Returns each job info in [job_infos] with the queue time colorized."""
    cutoff_values = [0.25, 0.5, 1, 3, 6, 12, 18, 24, 36, 72] # In hours
    color_scale = get_color_scale(
        start="green",
        mid="yellow",
        end="red",
        num_colors=len(cutoff_values)+1)

    def colorize_queue(ji):
        if ji.queue is None or not any([vc.isdigit() for vc in ji.queue]):
            return ji
        else:
            time_in_past = UtilsBase.time_to_minutes(ji.queue)
            idxs = [idx for idx,c in enumerate(cutoff_values) if time_in_past <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            queue = colorize(ji.queue, color=color_scale[min_valid_idx])
            return UtilsBase.updated_namespace(ji, queue_color=queue)

    return [colorize_queue(ji) for ji in job_infos]

def colorize_states(job_infos):
    """Returns each job info in [job_infos] with the state colorized.

    The first half of the state value reflects a potential heartbeat key, while the
    rest reflects the time since the job's output was written to.
    """
    cutoff_values = [1, 2, 5, 10, 20, 30, 40, 50, 60, 90] # In minutes
    color_scale = get_color_scale(
        start="green",
        mid="yellow",
        end="red",
        num_colors=len(cutoff_values)+1)

    def colorize_state(ji):
        if ji.jobid.startswith("__") or not any([c.isdigit() for c in ji.jobid]):
            return ji

        job_running = ji.state in ["RUNNING", "COMPLETING"]
        if not "user" in ji:
            return ji
        elif ji.user == os.environ["USER"] and (not "heartbeat" in ji or not decolorize(ji.heartbeat[0]).isnumeric()):
            color1 = color_scale[-1] if job_running else "no_change"
        elif not ji.user == os.environ["USER"]:
            # Here there's no 
            color1 = 142 if job_running else 174
        else:
            # Last possible time a heartbeat could've been written is the submit time for
            # pending jobs (ie. the completion time of a prior chunk) or the current time.
            heartbeat_time = UtilsBase.time_stamp_to_datetime(decolorize(ji.heartbeat))
            if job_running:
                last_possible_heartbeat = datetime.now()
            else:
                last_possible_heartbeat = UtilsBase.time_stamp_to_datetime(decolorize(ji.submit_time))
            elapsed1 = (last_possible_heartbeat - heartbeat_time).total_seconds() / 60
            idxs = [idx for idx,c in enumerate(cutoff_values) if elapsed1 <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            color1 = color_scale[min_valid_idx]

        if job_running:
            now = time.time()
            output_files = [ji.stderr, ji.stdout]
            output_files = [f for f in output_files if osp.exists(f)]
            output_file2seconds_elapsed = {f: now - osp.getmtime(f) for f in output_files}
            elapsed2 = (min(output_file2seconds_elapsed.values()) / 60) if output_file2seconds_elapsed else None
        
        elif ji.queue is None or not any([vc.isdigit() for vc in ji.queue]):
            submit_time = UtilsBase.time_stamp_to_datetime(decolorize(ji.submit_time))
            elapsed2 = (datetime.now() - submit_time).total_seconds() / 60
        else:
            elapsed2 = UtilsBase.time_to_minutes(decolorize(ji.queue))

        if elapsed2 is None:
            color2 = color1
        else:
            idxs = [idx for idx,c in enumerate(cutoff_values) if elapsed2 <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            color2 = color_scale[min_valid_idx]
        
        ji_state_len = len(ji.state)
        state_part1 = ji.state[:ji_state_len//2]
        state_part1 = colorize(state_part1, color=color1)
        state_part2 = ji.state[ji_state_len //2:]
        state_part2 = colorize(state_part2, color=color2)
        return UtilsBase.updated_namespace(ji, state_color=state_part1 + state_part2)

    return [colorize_state(ji) for ji in job_infos]
    
######################################################################################
######################################################################################
######################################################################################


def job_datas_with_to_prints(*, job_datas, col2max_chars):
    """Returns [job_datas] with a new key "to_print" that contains the string that
    should be printed to the terminal added for each job data.
    """
    try:
        all_time_left_prefixed_zero = all([jd.time_left.startswith("0") or jd.time_left == "time_left" for jd in job_datas])
        if all_time_left_prefixed_zero:
            for jd in job_datas:
                jd.time_left = jd.time_left[1:] if jd.time_left.startswith("0") else jd.time_left

        all_start_times_prefixed_zero = all([jd.start_time.startswith("0") or jd.start_time in ["start_time", "N/A"] for jd in job_datas])
        if all_start_times_prefixed_zero:
            for jd in job_datas:
                jd.start_time = jd.start_time[1:] if jd.start_time.startswith("0") else jd.start_time

        for j in job_datas:
            append_upper = j.jobid.startswith("__account")
            job_id = "" if j.jobid.startswith("next chunks") else str(j.jobid)
            data_dict = vars(j)
            s = []
            for c in col2max_chars:
                if j.jobid.startswith("__next chunks") and not c == "name":
                    s.append(" " * col2max_chars[c])  # Empty space for next chunks
                elif j.jobid.startswith("__account") and c == "jobid":
                    to_append = f"{'jobid':<{col2max_chars[c]}}"
                    s.append(to_append.upper() if append_upper else to_append)
                
                elif c == "name" and j.name.startswith("next chunks"):
                    to_append = f"------ {j.name} ------"
                    to_append = to_append.center(col2max_chars[c])
                    s.append(to_append)

                elif c == "name" and j.name == "name" and Utils.is_cc():
                    account = j.jobid.replace("__account ", "")
                    to_append = f"------ {account} ------"
                    to_append = to_append.center(col2max_chars[c] - len("NAME"))
                    to_append = "NAME" + to_append
                    s.append(to_append)
                
                else:
                    # Handles colorization better
                    value = data_dict[f"{c}_color"] if f"{c}_color" in data_dict else data_dict[c]
                    value = str(value)
                    value_len = len(decolorize(value)) if f"{c}_color" else len(value)
                    
                    chars_to_append = col2max_chars[c] - value_len
                    chars_to_append = max(chars_to_append, 0)
                    to_append = f"{value}{' ' * chars_to_append}"
                    s.append(to_append.upper() if append_upper else to_append)
        
            j.to_print = "  ".join(s)
        
        return job_datas
    except ValueError as e:
        if str(e) == "Sign not allowed in string format specifier":
            print("Terminal width too small. Resize the window and try again.")
            sys.exit(1)
        else:
            raise e

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

def job_info_with_queue(jd):
    """Returns job info [jd] with the queue time added."""
    if "eligible_time" in jd and jd.eligible_time in ["Unknown", "N/A"]:
        return UtilsBase.updated_namespace(jd, queue="N/A")
    elif "eligible_time" in jd and jd.eligible_time:
        eligible_time = datetime.strptime(jd.eligible_time, "%Y-%m-%dT%H:%M:%S")
    elif not "eligible_time" in jd:
        raise NotImplementedError(f"job_dict={jd} missing 'eligible_time' key")
    else:
        raise NotImplementedError(f"job_dict={jd} ")

    if jd.state in ["RUNNING", "COMPLETING"]:
        start_time = datetime.strptime(jd.start_time, "%Y-%m-%dT%H:%M:%S")
    else:
        start_time = datetime.now()
    
    queue = start_time - eligible_time
    queue = f"{queue.total_seconds() / 3600:.2f}H"
    return UtilsBase.updated_namespace(jd, queue=queue)

def job_info_with_formatted_date_time(jd, *, key, tz="America/Vancouver"):
    """Returns job info [jd] with the date/time key [key] formatted. This function should be run last!"""
    date_time = vars(jd)[key]
    valid_date_time = False

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
        valid_date_time = True
    elif date_time.startswith("N/A"):
        date_time = "N/A"
    else:
        assert 0, f"Unexpected start time format: {date_time}"

    if valid_date_time and tz:
        from zoneinfo import ZoneInfo
        target_tz = ZoneInfo("America/Vancouver")
        dt = datetime.strptime(date_time, "%m-%d-%H:%M")
        dt = dt.astimezone(target_tz)  # Convert to local timezone
        date_time = dt.strftime("%m-%d-%H:%M")

    return argparse.Namespace(**vars(jd) | {key: date_time})

def job_info_with_formatted_time_delta(jd, key="time_left"):
    """Returns job info [jd] with the time delta key [key] formatted."""
    delta = vars(jd)[key]
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
            twrite(f"[INFO] jobid={jd.jobid} got unexpected time_left={delta}")
            return argparse.Namespace(**vars(jd) | {key: delta})
    
    result = f"{mm:02}:{ss:02}" if hh == 0 else f"{hh}:{mm:02}:{ss:02}"
    result = " " * (9 - len(result)) + result

    return argparse.Namespace(**vars(jd) | {key: result})

def job_info_with_formatted_resources(jd, num_nodes=1):
    """Returns job info [jd] with the resources formatted."""
    def jd_with_unspecified_gpu_to_gpu_alias(jd):
        """Returns the GPU alias for the GPU used by job data [jd]."""
        node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
        if Utils.is_solar():
            gpu_name = node2config[jd.node]["gpu_name"]
            return MachineInfo.gpu_name2alias[gpu_name]
        elif Utils.is_cc():
            node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
            avail_gpu_alias2vram = {ga: MachineInfo.gpu2vram[ga] for ga in node2config if not ga == "default"}
            return min(avail_gpu_alias2vram, key=lambda x: avail_gpu_alias2vram[x])
        else:
            raise NotImplementedError(f"Unexpected cluster type for gres_gpu={gres_gpu} parsed as gpus={gpus}")
    
    gres_gpu = "N/A" if not jd.gres else jd.gres
    num_nodes = num_nodes if not jd.nodes else jd.nodes
    
    if gres_gpu == "N/A":
        gpus = "N/A"
    else:
        gpus = gres_gpu.replace("gres/gpu:", "").replace("gres:gpu:", "").replace("gpu:", "").split(":")
        
        if gpus[-1].isdigit() and len(gpus) == 1:
            num_gpus = int(gpus[-1])
            gpu_alias = jd_with_unspecified_gpu_to_gpu_alias(jd)
        elif gpus[-1] in MachineInfo.gpu_alias2name.values() and len(gpus) == 1:
            num_gpus = 1
            gpu_alias = MachineInfo.gpu_name2alias[gpus[-1]]
        elif gpus[-1].isdigit() and len(gpus) == 2 and gpus[0] in MachineInfo.gpu_alias2name.values():
            num_gpus = int(gpus[-1])
            gpu_alias = MachineInfo.gpu_name2alias[gpus[0]]
        else:
            raise NotImplementedError(f"Unexpected gres_gpu={gres_gpu} parsed as gpus={gpus} for jobid={jd.jobid}")
                
        num_gpus = MachineInfo.gpu2info[gpu_alias]["gpu_frac"] * num_gpus
        gpus =  f"{num_gpus * int(num_nodes)}"

    return UtilsBase.updated_namespace(jd, gpus=gpus)

def job_info_with_formatted_reason(jd):
    """Returns job info [jd] with the reason formatted."""
    reason = " ".join(jd.reason) if isinstance(jd.reason, list) else jd.reason
    if reason.startswith("Dependency") and not jd.dependency in ["N/A", "None", ""]:
        dependency_type,depends_on_jobid = jd.dependency.split(":")
        depends_on_jobid,_ = (depends_on_jobid.split("(")[0] if "(" in depends_on_jobid else depends_on_jobid).strip(), None
        
        # CamelCase the dependency type
        dependency_type = dependency_type.replace("after", "After").replace("ok", "Ok").replace("any", "Any").replace("not", "Not")
        reason = f"{dependency_type}:{depends_on_jobid}"
    else:
        reason = reason.split(":")[0].split(" ")[0].strip()  # Remove the first word and any colons
    return UtilsBase.updated_namespace(jd, reason=reason)

def job_info_without_preempt_me_name(jd):
    """Returns job info [jd] with the name without the 'preempt_me_' prefix."""
    name = jd.name.replace("preempt_me_", "") if jd.name.startswith("preempt_me_") else jd.name
    return UtilsBase.updated_namespace(jd, name=name)

def job_info_with_heartbeat(jd):
    """Returns job info [jd] with the heartbeat time added if possible."""
    if "comment" in jd:
        if "exp_name" in jd.comment:
            exp_name = jd.comment["exp_name"]
        elif "exp_name_trunc" in jd.comment and "uid" in jd.comment:
            exp_name = f"{jd.comment['exp_name_trunc']}*{jd.comment['uid']}*"
        else:
            return UtilsBase.updated_namespace(jd, heartbeat="no name")

        found_exp_folders = UtilsBase.flatten([glob.glob(osp.join(s, exp_name)) for s in args.exp_search_dirs])
        if len(found_exp_folders) == 0:
            return UtilsBase.updated_namespace(jd, heartbeat="not found")
        elif len(found_exp_folders) > 1:
            twrite(f"Multiple experiment folders found for exp_name={exp_name} with search_dirs={args.exp_search_dirs}: {found_exp_folders}")
            return UtilsBase.updated_namespace(jd, heartbeat="multiple exp names")
        elif osp.exists(osp.join(found_exp_folders[0], "heartbeat.txt")):
            with open(osp.join(found_exp_folders[0], "heartbeat.txt"), "r") as f:
                heartbeat = f.read().strip().split()
                heartbeat = f"{heartbeat[0]}T{heartbeat[1]}" # Matches a SLURM date-time format even though it came from Python
            jd = UtilsBase.updated_namespace(jd, heartbeat=heartbeat)
            return job_info_with_formatted_date_time(jd, key="heartbeat", tz=None)
        else:
            return UtilsBase.updated_namespace(jd, heartbeat="no hearbeat")
    else:
        return UtilsBase.updated_namespace(jd, heartbeat="")

def job_info_with_latest_str(*, args, jd):
    """Returns the latest checkoint for job data [jd] per a heuristic."""
    def checkpoint_to_sort_value(c):
        """Returns the prefix for checkpoint [c]."""
        if c.startswith("probe_pretep"):
            pretrain_epoch = UtilsBase.digits_after(c, "probe_pretep")
            probe_epoch = UtilsBase.digits_after(c, "prbep")
            return float(f"{pretrain_epoch}.{probe_epoch}")
        elif c.startswith("fn"):
            fn_epoch = UtilsBase.digits_after(c, "fn")
            return float(f"0.{fn_epoch}")
        elif c[0].isnumeric():
            pretrain_epoch = UtilsBase.digits_after(c, "")
            return float(f"{pretrain_epoch}.0")
        else:
            return 0

    if "comment" in jd:
        if "exp_name" in jd.comment:
            exp_name = jd.comment["exp_name"]
        elif "exp_name_trunc" in jd.comment and "uid" in jd.comment:
            exp_name = f"{jd.comment['exp_name_trunc']}*{jd.comment['uid']}*"
        else:
            return UtilsBase.updated_namespace(jd, chkpt="no name found")

        found_exp_folders = UtilsBase.flatten([glob.glob(osp.join(s, exp_name)) for s in args.exp_search_dirs])
        if len(found_exp_folders) == 0:
            return UtilsBase.updated_namespace(jd, chkpt="not found")
        elif len(found_exp_folders) > 1:
            twrite(f"Multiple experiment folders found for exp_name={exp_name} with search_dirs={args.exp_search_dirs}: {found_exp_folders}")
            return UtilsBase.updated_namespace(jd, chkpt="multiple exp names")
        else:
            exp_folder = found_exp_folders[0]
            checkpoints = [f for f in os.listdir(exp_folder) if f.endswith(".pt") and not f.startswith("wandb_data")]
            checkpoints = sorted(checkpoints, key=checkpoint_to_sort_value, reverse=True)
            checkpoint = checkpoints[0] if len(checkpoints) > 0 else "no checkpoints"
            return UtilsBase.updated_namespace(jd, chkpt=checkpoint)
    else:
        return UtilsBase.updated_namespace(jd, chkpt="")

# def job_info_with_diagnosis(ji):
#     """Returns [jd] with a diagnosis field. Basically, look at the past stderr/stdout
#     and use a heuristic to figure out what bad things have happened.

#     Essentialy, we want to check for 
#     """
#     def line_iter(s):
#         """Returns an iterator over the lines in [s]."""
#         s = s.strip()
#         while len(s):
#             if "\n" in s:
#                 next_newline = s.index("\n") + 1
#             else:
#                 next_newline = len(s)
#             yield s[:next_newline]
#             s = s[next_newline:]

#     def parse_nodes(s):
#         """Returns a list of nodes in [s]."""
#         s = s.strip()
#         if "[" in s:
#             prefix = s[:s.index("[")]
#             if "," in s:
#                 nodes = s[s.index("[")+1:s.index("]")].split(",")
#             elif "-" in s:
#                 range_part = s[s.index("[")+1:s.index("]")]
#                 range_parts = range_part.split("-")
#                 range_start, range_end = int(range_parts[0]), int(range_parts[1])
#                 nodes = [n for n in range(range_start, range_end+1)]
            
#             nodes = [f"{prefix}{n}" for n in nodes]
#         else:
#             nodes = [s]

#     def diagnose(job_output):
#         """Basically, we want to see if a known set of errors occur."""
#         diagnosis_indicators = ["ERROR", "Traceback", "Killed", "OOM", "OutOfMemory",
#             "CUDA out of memory", "RuntimeError", "Segmentation fault", "Aborted",
#             "Aborted (core dumped)", "Bus error", "MemoryError", "std::bad_alloc",
#             "AssertionError", "ValueError", "KeyboardInterrupt"]

#         interesting_line_starts = ["[rank", "srun", "slurm"]
#         interesting_strs = interesting_line_starts + diagnosis_indicators
        
#         problem_dict = dict()
#         sio = io.StringIO(job_output)
        
#         # The first line won't contain an error (probably), but does contain the nodes running the job and the job ID
#         first_line = sio.readline()
#         first_line_parts = first_line.split("")
#         job, nodes = None, []
#         for p in first_line_parts:
#             job = p.replace("job=", "") if p.startswith("job=") else job
#             nodes = parse_nodes(p.replace("nodes=", "")) if p.startswith("nodes=") else nodes

#         for line in sio:
#             if any([line.startswith(s) for s in interesting_strs]):
#                 if " DUE TO TIME LIMIT ***" in line:
#                     break
#                 else:
#                     for indicator in diagnosis_indicators:
#                         indicator_idx = line.find(indicator)
#                         if indicator_idx >= 0:
#                             problem_dict[indicator] = line[indicator_idx:].strip()
#                             _ = diagnosis_indicators.remove(indicator) # Not needed to count again, speeds list iteration
#                             interesting_strs = interesting_line_starts + diagnosis_indicators
#             else:
#                 continue

#         return dict(job=int(job_key), nodes=nodes, problems=problem_dict)

#     if (ji.jobid.startswith("__")
#         or not osp.exists(ji.stderr)
#         or not ji.user == os.environ["USER"]):
#         return UtilsBase.updated_namespace(ji, diagnosis="")

#     job_output = UtilsBase.load_file_lite(ji.stderr)
#     job_output = job_output.split("Starting job=")

#     chunk_jobids2diagnosis = [diagnose(jo) for jo in job_output]


    
        

def job_info_with_parition(ji):
    """Returns job info [jd] with the partition names shortened. However, we don't
    want to shorten partitions in a way that yields duplicates.
    """
    extant_partitions = ji.partition.split(",")
    short_partitions = list()
    for p in extant_partitions:
        short_p = UtilsBase.strip_left(p, "gpubase_")
        short_p = UtilsBase.strip_left(short_p, "gpu")
        if not short_p in extant_partitions:
            short_partitions.append(short_p)
        else:
            short_partitions.append(p)
    short_partitions = sorted(short_partitions)
    short_partitions = " ".join(short_partitions)
    return UtilsBase.updated_namespace(ji, partition=short_partitions)



def jobs_data(*, account=None, cur_user=False, next_chunks=False, nodes=False,
    submit_time=False, eligible_time=False, queue=False, checkpoint=False,
    excluded=False, heartbeat=False, heartbeat_analysis=False, output_files=None,
    partition=False,
    verbose=False):
    """Returns a (job2info, col_names) tuple where job2info is a dictionary mapping
    job IDs to info about their SLURM whatnot, and col_names is a list of column names
    indexing each value of [job2info].

    Args:
    account     -- SLURM account to get jobs for, or none for all applicable accounts
    cur_user    -- if True, only show jobs for the current user
    next_chunks -- if True, include next chunk jobs too
    nodes       -- if True, include the node list for all jobs
    submit_time -- if True, include the submit time for all jobs
    eligible_time -- if True, include the eligible time for all jobs
    queue       -- if True, include the queue time for all jobs
    checkpoint  -- if True, try to find the latest checkpoint for each job
    excluded    -- if True, include the excluded nodes for all jobs
    heartbeat   -- if True, include the heartbeat time for all jobs
    heartbeat_analysis -- if True, include heartbeat analysis info for all jobs
    partition   -- if True, include the partition name for all jobs
    """
    job2info = Utils.get_slurm_status(cur_user=cur_user, account=account, verbose=(verbose > 1))

    job2info = {j: info for j,info in job2info.items() if not info.name in args.hidden}

    if submit_time or heartbeat_analysis:
        job2info = {j: job_info_with_formatted_date_time(v, key="submit_time", tz=args.tz) for j,v in job2info.items()}
    if eligible_time:
        job2info = {j: job_info_with_formatted_date_time(v, key="eligible_time", tz=args.tz) for j,v in job2info.items()}
    if queue or heartbeat_analysis:
        job2info = {j: job_info_with_queue(v) for j,v in job2info.items()}
    if checkpoint:
        job2info = {j: job_info_with_latest_str(args=args, jd=v) for j,v in job2info.items()}
    if heartbeat or heartbeat_analysis:
        job2info = {j: job_info_with_heartbeat(v) for j,v in job2info.items()}
    if partition:
        job2info = {j: job_info_with_parition(v) for j,v in job2info.items()}

    job2info = {j: job_info_with_formatted_resources(info) for j,info in job2info.items()}
    job2info = {j: job_info_with_formatted_date_time(info, key="start_time", tz=args.tz) for j,info in job2info.items()}
    job2info = {j: job_info_with_formatted_time_delta(info, key="time_left") for j,info in job2info.items()}
    job2info = {j: job_info_with_formatted_reason(info) for j,info in job2info.items()}
    job2info = {j: job_info_without_preempt_me_name(info) for j,info in job2info.items()}


    # Combine an abbreviated partition name with the user name on Solar
    if Utils.is_solar() and not cur_user:
        for jobid,info in job2info.items():
            partition = info.partition.replace("-short", "").replace("-long", "").replace("-lab", "").replace("cs-gpu-research", "cs-gpu-")
            job2info[jobid].user = f"{info.user}/{partition}"

    col_names = [
        "host" if Utils.is_solar() or nodes else None,
        "exclude" if excluded else None,
        "partition" if partition else None,
        "jobid", "uid",
        "user" if not cur_user else None,
        "state",
        "submit_time" if submit_time else None,
        "eligible_time" if eligible_time else None,
        "queue" if queue else None,
        "chkpt" if checkpoint else None,
        "start_time",
        "heartbeat" if heartbeat else None,
        "gpus", "name", "time_left", "reason",]
    col_names = [c for c in col_names if not c is None]        
    
    # On Solar, sort all the running jobs by the node name. The node name is printed
    # on the far left.
    if Utils.is_solar():
        running_jobs = [k for k,v in job2info.items() if v.state == "RUNNING"]
        other_jobs = [k for k,v in job2info.items() if not v.state == "RUNNING"]
        running_jobs.sort(key=lambda k: job2info[k].host)
        job2info = {j: job2info[j] for j in running_jobs + other_jobs}

    ##################################################################################
    # I used to try and have jobs presubmit their next chunks, to overlap queue and
    # training time, but it turned out to be more trouble that it was worth. So,
    # deprecating the functionality to handle this.
    ##################################################################################
    
    # On ComputeCanada, there may be duplicate UIDs as jobs pre-submit their next job
    # chunk. So, we will sort all of the duplicates below the rest. In this case, we
    # will sort the jobs so matching UIDs are grouped together and ordered by the
    # start time of the least-job ID of the next chunks.
    # elif Utils.is_cc():
    #     uid2jobids = defaultdict(list)
    #     for idx,(jobid,info) in enumerate(job2info.items()):
    #         # Use indices so jobs without UIDs aren't impacted
    #         uid = info.uid if not info.uid is None else str(idx)
    #         uid2jobids[uid].append(jobid)
        
    #     least_job_ids = set([min(jobids) for _,jobids in uid2jobids.items()])
    #     if next_chunks:
    #         duplicate_job_ids = [j for j in job2info if not j in least_job_ids]
    #         duplicate_job_ids = sorted(duplicate_job_ids, key=lambda j: start_time_to_comparable(job2info[j].start_time))
    #         duplicate_job_ids = sorted(duplicate_job_ids, key=lambda j: job2info[j].uid)

    #         job2info_main = {j: info for j,info in job2info.items() if j in least_job_ids}
    #         job2info_with_duplicates = {j: job2info[j] for j in duplicate_job_ids}

    #         # Insert an indicator into [job2info] giving where the next chunks start
    #         if len(job2info_with_duplicates) > 0:
    #             account_str = "" if account is None else f" ({account})"
    #             next_chunk = {f"__next chunks{account_str}": {c: f"next chunks{account_str}" if c == "name" else (f"__next chunks{account_str}" if c == "jobid" else c) for c in col_names}}
    #             job2info = job2info_main | next_chunk | job2info_with_duplicates
    #     else:
    #         job2info = {j: info for j,info in job2info.items() if j in least_job_ids}

    ##################################################################################
    ##################################################################################
    ##################################################################################

    return list(job2info.values()), col_names

def account_to_levelfs_record(account):
    """Returns a dictionary giving the group and user LevelFS for [account]."""
    s = subprocess.getoutput(f"sshare -l -A {account} --noheader")
    if len(s) > 0:
        group, user = s.split("\n")
        group = float(group.split()[6])
        user = float(user.split()[8])
        return dict(group=group, user=user)
    else:
        return dict(group=None, user=None)

def build_record(*, job_datas, account2lfs):
    """Returns a JSON record giving what was going on with the cluster when run."""
    import os, time # Imported here to be faster when not 
    return dict(date=datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        time=time.time(), # Maybe useful for easy sorting? Idk.
        account2lfs=account2lfs,
        job_data=UtilsBase.try_make_jsonable(job_datas),
        user=os.environ["USER"],  # Maybe useful if multiple people run this and end up with different LevelFS user fields?
        )

def get_cluster_usage_str(job_infos=None, cur_user=False):
    """Returns a string giving the cluster GPU usage.

    Basically, want to show for the current user the running and queueing GPUs,
    their sum, and the same for all uses in the accounts.

    Args:
    job_infos   -- if not None, use this list of job infos instead of querying SLURM
    cur_user    -- whether [job_infos] was collected for the current user only
    
    """
    if job_infos is None or cur_user:
        job2info = Utils.get_slurm_status(cur_user=False, account=None, keys=["jobid", "user", "state", "nodes", "gres", "reason"])
        job2info = {j: job_info_with_formatted_resources(info) for j,info in job2info.items()}
        job2info = {j: job_info_with_formatted_reason(info) for j,info in job2info.items()}
        job_infos = list(job2info.values())
    
    gpu_data = argparse.Namespace(running=0, allocated=0, running_user=0, allocated_user=0)
    for ji in job_infos:
        if (ji.jobid.startswith("__")
            or not UtilsBase.is_numeric(ji.gpus)
            or any([ji.reason.startswith(r) for r in ["Dependency", "JobHeld"]])):
            continue
        
        num_gpus = float(ji.gpus)
        if ji.state in ["RUNNING", "COMPLETING"]:
            gpu_data.running += num_gpus
            gpu_data.running_user += num_gpus if ji.user == os.environ["USER"] else 0
        elif ji.state in ["PENDING"]:
            gpu_data.allocated += num_gpus
            gpu_data.allocated_user += num_gpus if ji.user == os.environ["USER"] else 0
        else:
            twrite(f"[WARNING] write_cluster_allocated_data(): jobid={ji.jobid} has unexpected state={ji.state}. Skipping.")
            continue
    gpu_data.total = gpu_data.running + gpu_data.allocated
    gpu_data.total_user = gpu_data.running_user + gpu_data.allocated_user
    
    s = f"GPUS:\t\t\t{os.environ['USER']}=(run={gpu_data.running_user} alloc={gpu_data.allocated_user} total={gpu_data.total_user})"
    s += f"\tall=(run={gpu_data.running} alloc={gpu_data.allocated} total={gpu_data.total})" 
    return s

def count_jobs_by_state(job_infos):
    """Returns a Namespace giving the number of the current user's jobs in particular
    states. For pending jobs, this is broken down by job health.
    """
    job_infos = [ji for ji in job_infos if "user" in ji and ji.user == os.environ["USER"]]
    state2count = defaultdict(int)
    for ji in job_infos:
        if ji.state == "RUNNING" or ji.state == "COMPLETING":
            state2count["running"] += 1
        elif ji.state == "PENDING":
            # Heuristically determine job health. A job whose experiment folder
            # doesn't exist is healthy (probably hasn't run). If its experiment folder
            # does exist, if it contains checkpoints and a wandb_attempt.txt, it's
            # probably healthy (we can make this heuristic better). Otherwise, there
            # is quite possibly an issue!
            if "comment" in ji and "exp_name" in ji.comment:
                exp = FileFinding.str_to_exp_folder(ji.comment["exp_name"], resolve="pos", if_not_found="none")
                if exp is None:
                    state2count["pending_healthy"] += 1
                else:
                    checkpoints = [f for f in os.listdir(exp) if f.endswith(".pt") and not f.startswith("wandb_data")]
                    if checkpoints and osp.exists(osp.join(exp, "wandb_attempt.txt")):
                        state2count["pending_healthy"] += 1
                    else:
                        state2count["pending_status_uncertain"] += 1
            else:
                state2count["pending_status_no_comment"] += 1
        else:
            state2count["other"] += 1
    state2count["total"] = sum(state2count.values())
    return argparse.Namespace(**state2count)

def format_job_counts_by_state(state2count):
    """Returns a string giving the job counts by state in [state2count]."""
    key2color = defaultdict(lambda: "white",
        running="green",
        pending_healthy="yellow",
        pending_status_uncertain="red",
        pending_status_no_comment="orange",
        other="red",
        total="purple2",
    )
    return "Queue status:\t" + " | ".join([colorize(f"{k}={v}", color=key2color[k]) for k,v in vars(state2count).items()])


if __name__ == "__main__":
    P = argparse.ArgumentParser(add_help=False, prefix_chars="-+")
    P.add_argument("-u", "--users", action="store_true", default=False,
        help="Show only jobs for all users")
    P.add_argument("-a", "--next_chunks", action="store_true", default=False,
        help="Show next chunk jobs too")
    P.add_argument("-n", "--nodes", action="store_true", default=False,
        help="Show show the node list for all jobs")
    P.add_argument("-s", "--submit_time", action="store_true", default=False,
        help="Show show the submit time for all jobs")
    P.add_argument("-e", "--eligible_time", action="store_true", default=False,
        help="Show show the submit time for all jobs")
    P.add_argument("-p", "--partition", action="store_true", default=False,
        help="Show the partition for all jobs")

    P.set_defaults(queue=True)
    P.add_argument("-q", "--queue", action="store_true",
        help="Show the queue times")
    P.add_argument("--no_queue", "--noq", action="store_false", dest="queue",
        help="Do not show the queue times")

    P.add_argument("-v", "--verbose", default=1, choices=[0, 1, 2], type=int,
        help="Verbosity: 0=no output, 1=default output, 2=default+commands being run")
    
    P.set_defaults(hidden=["HeldToProvideLevelFSEstimate"])
    P.add_argument("--hidden", nargs="+", action="append",
        help="Hidden jobs whose names aren't shown")
    P.add_argument("--no_hidden", "--show_hidden", action="store_const", dest="hidden", const=[],
        help="Do not hide jobs whose names aren't shown")

    P.add_argument("-r", "--record", default=False,
        help="Save outputs to this file for recording. 'default' saves to ~/.ClusterData/SqbOutputs/sqb_output_TIMESTR.json")

    P.set_defaults(exclude=True)
    P.add_argument("-x", "--exclude", action="store_true", dest="exclude",
        help="Show excluded nodes")
    P.add_argument("--no_exclude", "--nox", action="store_false", dest="exclude",
        help="Do not show excluded nodes")
    
    P.add_argument("-h", "--heartbeat", action="store_true", default=False,
        help="For jobs that write to a heartbeat.txt file, show the last heartbeat time")
    P.add_argument("--heartbeat_analysis", action="store_true", default=True,
        help="Like --heartbeat but does analysis instead of showing raw values")
    P.add_argument("--help", action="help",
        help="Show this help message and exit")

    P.set_defaults(checkpoint=True)
    P.add_argument("-c", "--checkpoint", action="store_true", dest="checkpoint",
        help="Try and find the latest checkpoint associated to each job with a UID")
    P.add_argument("--no_checkpoint", "--noc", action="store_false", dest="checkpoint",
        help="Do not try and find the latest checkpoint associated to each job with a UID")
    
    
    P.add_argument("--exp_search_dirs", nargs="+",
        default=[osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
            osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
            osp.expanduser("~/scratch/IMLE-SSL/finetunes"),
            osp.expanduser("~/scratch/IMLE-SSL/models_stop")],
        help="Directories to search for latest checkpoints in. If empty, no checkpoints are searched for.")
    P.set_defaults(color=True)
    P.add_argument("--no_color", action="store_false", dest="color",
        help="Do not colorize the output")


    P.add_argument("--tz", default="America/Vancouver",
        help="Timezone to convert times to. Default is America/Vancouver.")
    args = P.parse_args()


    if Utils.is_solar():
        job_datas, colnames = jobs_data(cur_user=not args.users, account=None,
            next_chunks=args.next_chunks,
            nodes=args.nodes,
            excluded=args.exclude,
            submit_time=args.submit_time,
            eligible_time=args.eligible_time,
            queue=args.queue,
            heartbeat=args.heartbeat,
            heartbeat_analysis=args.heartbeat_analysis,
            checkpoint=args.checkpoint,
            partition=args.partition,
            verbose=args.verbose)
        job_datas = [argparse.Namespace(**dict(zip(colnames, colnames)))] + job_datas        
    elif Utils.is_cc():
        
        job_datas = []
        for account in Utils.cluster2info[Utils.get_cluster_type()].accounts:
            job_datas_account, colnames = jobs_data(account=account, cur_user=not args.users,
                next_chunks=args.next_chunks,
                nodes=args.nodes,
                excluded=args.exclude,
                submit_time=args.submit_time,
                eligible_time=args.eligible_time,
                queue=args.queue,
                heartbeat=args.heartbeat,
                heartbeat_analysis=args.heartbeat_analysis,
                checkpoint=args.checkpoint,
                partition=args.partition,
                verbose=args.verbose)
            if len(job_datas_account) > 0:
                colnames_job_data = argparse.Namespace(**{c: f"__account {account}" if c == "jobid" else c for c in colnames})
                job_datas += [colnames_job_data] + job_datas_account
    else:
        # On workstations, the obvious equivalent is finding free GPUs.
        twrite(subprocess.getoutput("python ~/.ScriptsAndAliases/FindFreeGPUs.py --solar 0"))
        time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        twrite(time_str)
        sys.exit(0)
    
    col2max_chars = {c: len(c) for c in colnames}
    for job_data in job_datas:
        jd_dict = vars(job_data)
        for c in colnames:
            value = str(jd_dict[c])
            length = 0 if str(jd_dict["jobid"]).startswith("__") else len(value)
            col2max_chars[c] = max(col2max_chars[c], length)
    col2max_chars = {c: mc for c,mc in col2max_chars.items()}

    # Try building the string representation for each job data.
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)

    # If including the full name would put the output over one line, first try
    # removing GPU and time specifications
    all_on_one_line = len(job_datas) == 0 or max([len(j.to_print) for j in job_datas]) <= shutil.get_terminal_size().columns
    if not all_on_one_line:
        col2max_chars["name"] = 0
        for job_data in job_datas:
            job_data.name = job_name_without_gpu_time_spec(job_data.name)
            col2max_chars["name"] = max(col2max_chars["name"], len(job_data.name))

    # If any job name is still too long, re-order the output so the job name comes
    # last, and then make offending job names print on a line below the rest
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)
    all_on_one_line = len(job_datas) == 0 or max([len(j.to_print) for j in job_datas]) < shutil.get_terminal_size().columns
    if not all_on_one_line:
        col_names = [c for c in colnames if not c == "name"] + ["name"]
        col2max_chars = {c: col2max_chars[c] for c in col_names}
        
        # Have to work with explicitly the name column, since it will have been padded
        # so that short job names have many characters
        other_chars = sum([col2max_chars[c] for c in col_names if not c == "name"])
        max_name_chars = shutil.get_terminal_size().columns - other_chars - 2 * (len(col_names)-1)  # 2 for the spaces between columns
        for j in job_datas:
            if len(j.name.strip()) > max_name_chars:
                j.name = "\n\t\t" + j.name + "\n"
        
        # Exclude jobs whose names are on a new line from the length calculation
        col2max_chars["name"] = 0
        for job_data in job_datas:
            job_name_ = "" if job_data.name.startswith("\n\t\t") else job_data.name.strip()
            col2max_chars["name"] = max(col2max_chars["name"], len(job_name_))
        
    if args.color:
        job_datas = colorize_queues(job_datas) if "queue" in col2max_chars else job_datas
        job_datas = colorize_time_lefts(job_datas) if "time_left" in col2max_chars else job_datas
        job_datas = colorize_start_times(job_datas) if "start_time" in col2max_chars else job_datas
        job_datas = colorize_reasons(job_datas) if "reason" in col2max_chars else job_datas
        job_datas = colorize_states(job_datas) if "state" in col2max_chars else job_datas
        job_datas = colorize_submit_times(job_datas) if "submit_time" in col2max_chars else job_datas
        job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)

    if args.verbose:
        lines = "\n".join([j.to_print for j in job_datas])
        print(lines)

    # Now describe the overall cluster status or roughly how allocated it is
    time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    meta_str = f"--- Overall Cluster Status ({time_str}) ---"
    if Utils.is_slurm():
        state2count = count_jobs_by_state(job_datas)
        job_count_str = format_job_counts_by_state(state2count)
        meta_str += "\n\t| " + job_count_str
    if Utils.is_cc():
        usage_str = get_cluster_usage_str(job_infos=job_datas, cur_user=not args.users)
        meta_str += "\n\t| " + usage_str
    
    if args.verbose:
        print(meta_str)
    
    if Utils.is_cc():
        accounts = Utils.cluster2info[Utils.get_cluster_type()].accounts
        account2lfs = {a: account_to_levelfs_record(a) for a in accounts} # This is a record with saving
        account2lfs_str = {a: {k: f"{l:.2f}" if isinstance(l, float) else str(l) for k,l in lfs.items()} for a,lfs in account2lfs.items() if not lfs["group"] is None}
        level_fs_strs = [f"{a}={lfs['group']:} (user={lfs['user']})" for a,lfs in account2lfs_str.items()]
        level_fs_str = "\t| LevelFS:\t\t" + "\t".join(level_fs_strs)
        level_fs_str = level_fs_str.replace("_gpu", "")
        meta_str2 = level_fs_str + "\n"
    elif Utils.is_solar():
        account2lfs = None
        meta_str2 = ""    
    
    meta_str2 += "\t| " + ClusterInfo2.get_str()
    # meta_str2 +=  "\t| " + ClusterInfo.get_resource_info_summary()
    if args.verbose:
        print(meta_str2)

    # If on ComputeCanada and --record is set, save the data that was computed so we
    # can reference it later.
    if Utils.is_cc() and args.record == "default":
        import os.path as osp # Imported here to be faster when not needed
        time_str = time_str.replace(":", "-")
        args.record = osp.join(osp.expanduser(f"~/.ClusterData"), "SqbOutputs", f"sqb_output_{time_str}.json")
    if Utils.is_cc() and args.record:
        import UtilsBase # Imported here to be faster when not needed
        job_datas = [jd for jd in job_datas if not jd.jobid.startswith("__")]
        record = build_record(job_datas=job_datas, account2lfs=account2lfs)
        _ = UtilsBase.atomic_save_lite(data=record, fname=args.record)

   




