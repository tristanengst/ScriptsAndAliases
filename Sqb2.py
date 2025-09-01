"""Like sqb, but better. Displays the JobID, UID, state, estimated start time,
job name, and time remaining. Jobs are separated by SLURM account.
"""
import argparse
from collections import defaultdict
from datetime import datetime
from functools import partial
import glob
import os
import os.path as osp
import re
import shutil
import subprocess
import sys
import time

from ShowCluster import Node
import Utils
import UtilsBase
from UtilsBase import twrite

##### Colorization utilities #########################################################
# See: https://jakob-bagterp.github.io/colorist-for-python/ansi-escape-codes/extended-256-colors/#extended-palette
color2value_base = dict(
    blue=21,
    green=46,
    yellow=226,
    red=196,
    purple=201,
    lightblue=51,
    white=231,
)

def get_color_scale(*, start, end, mid=None, num_colors=11, light_bias=0):
    """Returns a list of [num_colors] going between [start] and [end].

    Args:
    start       -- start color name
    end         -- end color name 
    mid         -- mid color name. Not required for [num_colors] <= 6
    num_colors  -- number of colors to return
    light_bias  -- shifts the colors to be more grayscale (looks better on black background)
    """
    if num_colors < 2 or num_colors > 11:
        raise ValueError(f"num_colors={num_colors} must be between 2 and 11")

    start = start if isinstance(start, int) else color2value_base[start]
    end = end if isinstance(end, int) else color2value_base[end]
    end_color = UtilsBase.reverse_dict(color2value_base)[end]
    start_color = UtilsBase.reverse_dict(color2value_base)[start]

    # twrite(start=start, end=end, mid=mid, num_colors=num_colors, end_color=end_color, start_color=start_color)
    
    if num_colors >= 6 and mid is None:
        mid = (end - start) // 2 + start
    elif not mid is None:
        mid = mid if isinstance(mid, int) else color2value_base[mid]
    else:
        pass
        
    if num_colors >= 6:
        scale_delta1 = (mid - start) / 5
        light_bias1_mul = (5+ abs(scale_delta1)) // 6 
        scale_delta2 = (end - mid) / 5
        light_bias2_mul = (5 + abs(scale_delta2)) // 6 
        scale1 = [start + i * scale_delta1 + light_bias * light_bias1_mul for i in range(6)] # Total of six values, puts more resolution near start
        scale2 = [mid + i * scale_delta2 + light_bias * light_bias2_mul for i in range(1,6)] # Total of five values, puts less resolution near end
        scale = scale1 + scale2


        # twrite(light_bias=light_bias, light_bias1_mul=light_bias1_mul, light_bias2_mul=light_bias2_mul, start_color=start_color, end_color=end_color, scale_delta1=scale_delta1, scale_delta2=scale_delta2, )
    else:
        scale_delta = (end - start) / 5
        light_bias_mul = (5 + abs(scale_delta)) // 6
        scale = [start + i * scale_delta + light_bias_mul * light_bias for i in range(7)]

    
    scale = [int(s) for s in scale]
    # twrite(scale=scale, len_scale=len(scale))

    scale_inner = scale[1:-1]
    num_to_select = len(scale_inner) // (num_colors - 2)
    scale_inner = scale_inner[::num_to_select]
    scale_inner = scale_inner[:min(num_colors - 2, len(scale_inner))]

    result = [scale[0]] + scale_inner + [scale[-1]]
    return result

color2value = {c: f"\033[38;5;{v}m" for c,v in (color2value_base | dict(
    reset=0,
    green1=46,
    green2=40,
    green3=34,
    green4=118,
    green5=154,
    yellow1=190,
    yellow2=226,
    yellow3=220,
    blue1=39,
    blue2=27,
    purple1=129,
    purple2=165,    
    red1=208,
    red2=202,
    red3=196,
)).items()}

def colorize(s, color="no_change"):
    """Returns [s] colorized with ANSI escape codes."""
    color = color2value[color] if color in color2value else color
    color = f"\033[38;5;{color}m" if isinstance(color, int | float) else color
    return s if color == "no_change" else f"{color}{s}\033[0m".strip()

def decolorize(s):
    """Returns [s] with ANSI escape codes removed, eg. so its length is correct."""
    s_orig = s
    decolorized_s = ""
    while len(s):
        if s.startswith("\x1b["):
            next_valid_idx = s.index("m") + 1
        else:
            next_valid_idx = 1
            decolorized_s += s[0]
        s = s[next_valid_idx:]


    if "\x1b" in decolorized_s:
        raise ValueError(f"decolorized_s={decolorized_s} still has escape codes on s='{s_orig}'")
    return decolorized_s

def colorize_list(l, *, cutoffs="state", interpretaton_fn="delta", color_start=None, color_mid=None, color_end=None, light_bias=0):
    """Returns the values in list [l] colorized.

    Args:
    l                   -- list of values to colorize
    cutoffs             -- list of cutoff values for coloring. Must be one less than the number of colors
    interpretaton_fn    -- sends values in the list to how they should be colorized, or None for no colorization
    color_start         -- color name for the lowest value
    color_mid           -- color name for the middle value
    color_end           -- color name for the highest value
    """
    if len(l) == 0:
        return l
        
    def parse_timestamp_to_hours_from_now(v):
        """Returns the duration in seconds from now to the timestamp [v]."""
        time_stamp = UtilsBase.time_stamp_to_datetime(v)
        seconds = UtilsBase.hours_since_time(time_stamp)
        return abs(seconds)

    if callable(interpretaton_fn):
        fn = interpretaton_fn
    elif interpretaton_fn == "delta":
        fn = lambda v: UtilsBase.time_to_minutes(v) if any([vc.isdigit() for vc in v]) else None
    elif interpretaton_fn == "time_away":
        def fn(v):
            v = v.strip()
            if v == "N/A":
                return float("inf")
            elif any([vc.isdigit() for vc in v]):
                return parse_timestamp_to_hours_from_now(v)
            else:
                return None
    elif interpretaton_fn == "try_make_number":
        def fn(v):
            v = UtilsBase.try_make_number(v)
            return v if isinstance(v, (int, float)) else None
    else:
        raise ValueError(f"interpretaton_fn={interpretaton_fn} not recognized")


    # Shorter duration, more is worse
    if cutoffs == "state":
        color_start = color_start if color_start else "green"
        color_mid = color_mid if color_mid else "yellow"
        color_end = color_end if color_end else "blue"
        cutoff_values = [1, 3, 5, 10, 20, 30, 40, 50, 51, 52]
    
    # Longer duration, more is worse
    elif cutoffs == "queue_time" or cutoffs == "start_time":
        color_start = color_start if color_start else "blue"
        color_mid = color_mid if color_mid else "purple"
        color_end = color_end if color_end else "red"
        light_bias = 2
        cutoff_values = [1, 3, 12, 24, 36, 72]
    
    # Longer duration, more is better
    elif cutoffs == "time_left":
        color_start = color_start if color_start else "red"
        color_mid = color_mid if color_mid else "purple"
        color_end = color_end if color_end else "blue"
        light_bias = 2
        cutoff_values = [0.5, 1, 2, 3, 4, 5, 6, 7, 12, 24]
        cutoff_values = [cutoff_values * 60 for cutoff_values in cutoff_values] # Convert to minutes

    elif (isinstance(cutoffs, list | tuple)
        and all([isinstance(v, (int, float)) for v in cutoffs])
        and len(cutoffs) >= 1 and len(cutoffs) <= 10
        and not color_start is None and not color_end is None
        and (len(cutoffs) <= 5 or not color_mid is None)):
        cutoff_values = cutoffs
    else:
        raise ValueError(f"cutoffs={cutoffs} not usable. Ensure colors are set correctly too.")

    list_element_and_values = [(l1, fn(l1)) for l1 in l]
    color_scale = get_color_scale(
        start=color_start,
        mid=color_mid,
        end=color_end,
        num_colors=len(cutoff_values)+1,
        light_bias=light_bias
    )
    
    colorized = []
    for l1,v in list_element_and_values:
        if v is None:
            colorized.append(l1)
        else:
            idxs = [idx for idx,c in enumerate(cutoff_values) if v <= c]
            min_valid_idx = min(idxs) if len(idxs) else len(cutoff_values)
            colorized.append(colorize(l1, color=color_scale[min_valid_idx]))
    
    return colorized

def apply_along_column(list2d, *, fn, colname_or_idx, apply_to_top=True, apply_elementwise=False):
    """Returns [list2d] with function [fn] applied along column [colname_or_idx]."""
    if isinstance(colname_or_idx, str):
        for idx,c in enumerate(list2d[0]):
            if c.strip() == colname_or_idx:
                colname_or_idx = idx
                break
    if not isinstance(colname_or_idx, int):
        raise ValueError(f"colname_or_idx={colname_or_idx} must be an int or str that is a column name in list2d={list2d}")
    if colname_or_idx >= min([len(l) for l in list2d]):
        raise ValueError(f"colname_or_idx={colname_or_idx} is out of bounds for list2d={list2d}\n\nwith lengths={[len(l) for l in list2d]}")

    col = [l[colname_or_idx] for l in (list2d if apply_to_top else list2d[1:])]
    col = [fn(c) for c in col] if apply_elementwise else fn(col)
    col = col if apply_to_top else [list2d[0][colname_or_idx]] + col

    assert len(col) == len(list2d), f"len(col)={len(col)} != len(list2d)={len(list2d)}"

    return [l[:colname_or_idx] + [c] + l[colname_or_idx+1:] for l,c in zip(list2d, col)]

    
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
                    to_append = f"{to_append:^{col2max_chars[c]}}"
                    s.append(to_append)

                elif c == "name" and j.name == "name" and Utils.is_cc():
                    account = j.jobid.replace("__account ", "")
                    to_append = f"------ {account} ------"
                    to_append = f"{to_append:^{col2max_chars[c] - len(c) * 2}}"
                    to_append = "NAME" + to_append
                    s.append(to_append)
                
                else:
                    to_append = f"{str(data_dict[c]):<{col2max_chars[c]}}"
                    s.append(to_append.upper() if append_upper else to_append)
        
            j.to_print_list = s
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

def job_dict_with_queue_time(jd):
    """Returns job dict [jd] with the queue time added."""
    if "eligible_time" in jd and jd.eligible_time in ["Unknown", "N/A"]:
        return UtilsBase.updated_namespace(jd, queue_time="N/A")
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
    
    queue_time = start_time - eligible_time
    queue_time = f"{queue_time.total_seconds() / 3600:.2f}H"
    return UtilsBase.updated_namespace(jd, queue_time=queue_time)

def job_dict_with_formatted_date_time(jd, *, key):
    """Returns job dict [jd] with the date/time key [key] formatted. This function should be run last!"""
    date_time = vars(jd)[key]

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

    return argparse.Namespace(**vars(jd) | {key: date_time})

def job_dict_with_formatted_time_delta(jd, key="time_left"):
    """Returns job dict [jd] with the time delta key [key] formatted."""
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
            print(f"[INFO] jobid={jd['jobid']} got unexpected time_left={delta}")
            return argparse.Namespace(**vars(jd) | {key: delta})
    
    result = f"{mm:02}:{ss:02}" if hh == 0 else f"{hh}:{mm:02}:{ss:02}"
    result = " " * (9 - len(result)) + result

    return argparse.Namespace(**vars(jd) | {key: result})

def job_dict_with_formatted_resources(jd, num_nodes=1):
    """Returns job dict [jd] with the resources formatted."""
    known_gpu = ["h100", "a100", "l40s", "a40", "a5000", "v100"]
    
    gres_gpu = "N/A" if not jd.gres else jd.gres
    num_nodes = num_nodes if not jd.nodes else jd.nodes
    
    if gres_gpu == "N/A":
        gpus = "N/A"
    else:
        gpus = gres_gpu.replace("gres/gpu:", "").replace("gres:gpu:", "").replace("gpu:", "").split(":")
        
        # If the last element is a GPU, no GPU was requested
        num_gpus = 0 if any([gpus[-1].startswith(g) for g in known_gpu]) else int(gpus[-1])
        gpu_type = None if len(gpus) < 2 else gpus[-2]
        gpus =  f"{num_gpus * int(num_nodes)}"

    return UtilsBase.updated_namespace(jd, gpus=gpus,)

def job_dict_with_formatted_reason(jd):
    """Returns job dict [jd] with the reason formatted."""
    reason = " ".join(jd.reason) if isinstance(jd.reason, list) else jd.reason
    reason = reason.split(":")[0].split(" ")[0].strip()  # Remove the first word and any colons
    return UtilsBase.updated_namespace(jd, reason=reason)

def job_dict_without_preempt_me_name(jd):
    """Returns job dict [jd] with the name without the 'preempt_me_' prefix."""
    name = jd.name.replace("preempt_me_", "") if jd.name.startswith("preempt_me_") else jd.name
    return UtilsBase.updated_namespace(jd, name=name)

def job_dict_with_heartbeat(jd):
    """Returns job dict [jd] with the heartbeat time added if possible."""
    if "comment" in jd:
        if "exp_name" in jd.comment:
            exp_name = jd.comment["exp_name"]
        elif "exp_name_trunc" in jd.comment and "uid" in jd.comment:
            exp_name = f"{jd.comment['exp_name_trunc']}*{jd.comment['uid']}*"
        else:
            return UtilsBase.updated_namespace(jd, heartbeat="no exp_name found")

        found_exp_folders = UtilsBase.flatten([glob.glob(osp.join(s, exp_name)) for s in args.exp_search_dirs])
        if len(found_exp_folders) == 0:
            return UtilsBase.updated_namespace(jd, heartbeat="no exp folders found")
        elif len(found_exp_folders) > 1:
            print(f"Multiple experiment folders found for exp_name={exp_name} with search_dirs={args.exp_search_dirs}: {found_exp_folders}")
            return UtilsBase.updated_namespace(jd, heartbeat="multiple exp names")
        elif osp.exists(osp.join(found_exp_folders[0], "heartbeat.txt")):
            with open(osp.join(found_exp_folders[0], "heartbeat.txt"), "r") as f:
                heartbeat = f.read().strip().split()
                heartbeat = f"{heartbeat[0]}T{heartbeat[1]}" # Matches a SLURM date-time format even though it came from Python
            jd = UtilsBase.updated_namespace(jd, heartbeat=heartbeat)
            return job_dict_with_formatted_date_time(jd, key="heartbeat")
        else:
            return UtilsBase.updated_namespace(jd, heartbeat="no hearbeat file")
    else:
        return UtilsBase.updated_namespace(jd, heartbeat="-")

def job_dict_with_heartbeat_analysis(jd):
    """Returns job dict [jd] with the STATE key colorized to indicate job health.
    
    The state key is colorized in two ways:
    1. The first half is colored based on the job's recorded heartbeat if it exists
    2. The second half is colored differently, depending on if the state.
        - For running jobs, based on how recently the job's output file was modified (green -> red)
        - For pending jobs, based on how long the job has been pending (blue -> red)
    """
    def seconds_to_color_green_to_red(seconds):
        if seconds is None:
            return "no_change"
    
        minutes = seconds / 60
        # Zero to 20 minutes, everything is probably fine
        # 20-30 minutes, probably okay
        # 30-60 minutes, problem
        if minutes < 1:
            return "green1"
        elif minutes < 2:
            return "green2"
        elif minutes < 5:
            return "green3"
        elif minutes < 10:
            return "green4"
        elif minutes < 20:
            return "green5"
        elif minutes < 30:
            return "yellow1"
        elif minutes < 40:
            return "yellow2"
        elif minutes < 50:
            return "yellow3"
        elif minutes < 60:
            return "red1"
        elif minutes < 90:
            return "red2"
        else:
            return "red3"
        
    def seconds_to_color_blue_to_red(seconds):
        if seconds is None:
            return "no_change"
    
        hours = seconds / 3600
        # Less than 1H, good
        # 1-3H, okay
        # 3-12H, meh
        # 12-24H, not great
        # 24-36H, worse
        # 36H-72H, bad
        # 72H+, terrible
        if hours < 1:
            return "blue1"
        elif hours < 3:
            return "blue2"
        elif hours < 12:
            return "purple1"
        elif hours < 24:
            return "purple2"
        elif hours < 36:
            return "red1"
        elif hours < 72:
            return "red2"
        else:
            return "red3"

    job_running = jd.state in ["RUNNING", "COMPLETING"]
    if job_running:
        now = time.time()
        output_files = [jd.stderr, jd.stdout]
        output_files = [f for f in output_files if osp.exists(f)]
        output_file2seconds_elapsed = {f: now - osp.getmtime(f) for f in output_files}
        _ = twrite(f"[INFO] No output files found for jobid={jd.jobid} with stderr={jd.stderr} and stdout={jd.stdout}", verbose=not output_files)
        elapsed2 = min(output_file2seconds_elapsed.values()) if output_file2seconds_elapsed else None
    elif jd.queue_time in ["N/A", None]:
        submit_time = datetime.strptime(jd.submit_time, "%m-%d-%H:%M")
        elapsed2 = (datetime.now() - submit_time).total_seconds()
    else:
        elapsed2 = float(jd.queue_time.replace("H", "")) * 3600
        
    if not "heartbeat" in jd or not jd.heartbeat[0].isnumeric():
        elapsed1 = None
    else:
        heartbeat_time = datetime.strptime(jd.heartbeat, "%m-%d-%H:%M")
        
        # Last possible time a heartbeat could've been written is the submit time for
        # pending jobs (ie. the completion time of a prior chunk) or the current time.
        if job_running:
            last_possible_heartbeat = datetime.now()
        else:
            last_possible_heartbeat = datetime.strptime(jd.submit_time, "%m-%d-%H:%M")

        # Since the year isn't given in the heartbeat time, it is assumed to be 1900.
        # Obviously this is wrong.
        heartbeat_time = heartbeat_time.replace(year=last_possible_heartbeat.year)
        elapsed1 = (last_possible_heartbeat - heartbeat_time).total_seconds()

    color1 = seconds_to_color_green_to_red(elapsed1)
    color2_fn = seconds_to_color_green_to_red if job_running else seconds_to_color_blue_to_red
    color2 = color2_fn(elapsed2)
    
    jd_state_len = len(decolorize(jd.state))
    state_part1 = jd.state[:jd_state_len//2]
    state_part1 = colorize(state_part1, color=color1)
    state_part2 = jd.state[jd_state_len //2:]
    state_part2 = colorize(state_part2, color=color2)
    jd.state = state_part1 + state_part2
    return jd

def jobs_data(*, account=None, cur_user=False, next_chunks=False, nodes=False,
    submit_time=False, eligible_time=False, queue_time=False, latest_checkpoint=False,
    excluded=False, heartbeat=False, heartbeat_analysis=False, output_files=None,
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
    queue_time  -- if True, include the queue time for all jobs
    latest_checkpoint -- if True, try to find the latest checkpoint for each job
    """
    job2info = Utils.get_slurm_status(cur_user=cur_user, account=account, verbose=(verbose > 1))

    if submit_time or heartbeat_analysis:
        job2info = {j: job_dict_with_formatted_date_time(v, key="submit_time") for j,v in job2info.items()}
    if eligible_time:
        job2info = {j: job_dict_with_formatted_date_time(v, key="eligible_time") for j,v in job2info.items()}
    if queue_time or heartbeat_analysis:
        job2info = {j: job_dict_with_queue_time(v) for j,v in job2info.items()}
    if latest_checkpoint:
        job2info = {j: job_dict_with_latest_str(args=args, jd=v) for j,v in job2info.items()}
    if heartbeat or heartbeat_analysis:
        job2info = {j: job_dict_with_heartbeat(v) for j,v in job2info.items()}

    job2info = {j: job_dict_with_formatted_resources(info) for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_date_time(info, key="start_time") for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_time_delta(info, key="time_left") for j,info in job2info.items()}
    job2info = {j: job_dict_with_formatted_reason(info) for j,info in job2info.items()}
    job2info = {j: job_dict_without_preempt_me_name(info) for j,info in job2info.items()}

    if heartbeat_analysis:
        job2info = {j: job_dict_with_heartbeat_analysis(info) for j,info in job2info.items()}

    # Combine an abbreviated partition name with the user name on Solar
    if Utils.is_solar() and not cur_user:
        for jobid,info in job2info.items():
            partition = info.partition.replace("-short", "").replace("-long", "").replace("-lab", "").replace("cs-gpu-research", "cs-gpu-")
            job2info[jobid].user = f"{info.user}/{partition}"

    col_names = [
        "host" if Utils.is_solar() or nodes else None,
        "exc_nodes" if excluded else None,
        "jobid", "uid",
        "user" if not cur_user else None,
        "state",
        "submit_time" if submit_time else None,
        "eligible_time" if eligible_time else None,
        "queue_time" if queue_time else None,
        "chkpt" if latest_checkpoint else None,
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
        job_data=job_datas,
        user=os.environ.user,  # Maybe useful if multiple people run this and end up with different LevelFS user fields?
        )


def job_dict_with_latest_str(*, args, jd):
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
            return UtilsBase.updated_namespace(jd, chkpt="no exp_name found")

        found_exp_folders = UtilsBase.flatten([glob.glob(osp.join(s, exp_name)) for s in args.exp_search_dirs])
        if len(found_exp_folders) == 0:
            return UtilsBase.updated_namespace(jd, chkpt="no exp folders found")
        elif len(found_exp_folders) > 1:
            print(f"Multiple experiment folders found for exp_name={exp_name} with search_dirs={args.exp_search_dirs}: {found_exp_folders}")
            return UtilsBase.updated_namespace(jd, chkpt="multiple exp names")
        else:
            exp_folder = found_exp_folders[0]
            checkpoints = [f for f in os.listdir(exp_folder) if f.endswith(".pt") and not f.startswith("wandb_data")]
            checkpoints = sorted(checkpoints, key=checkpoint_to_sort_value, reverse=True)
            checkpoint = checkpoints[0] if len(checkpoints) > 0 else "no checkpoints found"
            return UtilsBase.updated_namespace(jd, chkpt=checkpoint)
    else:
        return UtilsBase.updated_namespace(jd, chkpt="-")



if __name__ == "__main__":
    P = argparse.ArgumentParser(add_help=False)
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

    P.add_argument("-x", "--exclude", action="store_true", default=False,
        help="Show excluded nodes")

    P.add_argument("-h", "--heartbeat", action="store_true", default=False,
        help="For jobs that write to a heartbeat.txt file, show the last heartbeat time")
    P.add_argument("--heartbeat_analysis", action="store_true", default=True,
        help="Like --heartbeat but does analysis instead of showing raw values")
    P.add_argument("--help", action="help",
        help="Show this help message and exit")

    P.add_argument("-l", "--latest_checkpoint", action="store_true", default=False,
        help="Try and find the latest checkpoint associated to each job with a UID")
    P.add_argument("--exp_search_dirs", nargs="+",
        default=[osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
            osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
            osp.expanduser("~/scratch/IMLE-SSL/finetunes"),
            osp.expanduser("~/scratch/IMLE-SSL/models_stop")],
        help="Directories to search for latest checkpoints in. If empty, no checkpoints are searched for.")
    args = P.parse_args()


    if Utils.is_solar():
        job_datas, colnames = jobs_data(cur_user=not args.users, account=None,
            next_chunks=args.next_chunks,
            nodes=args.nodes,
            excluded=args.exclude,
            submit_time=args.submit_time,
            eligible_time=args.eligible_time,
            queue_time=args.queue_time,
            heartbeat=args.heartbeat,
            heartbeat_analysis=args.heartbeat_analysis,
            latest_checkpoint=args.latest_checkpoint,
            verbose=args.verbose)
        job_datas = [argparse.Namespace(**dict(zip(colnames, colnames)))] + job_datas        
    elif Utils.is_cc():
        accounts = ["rrg-keli_cpu", "def-keli_cpu", "rrg-keli_gpu", "def-keli_gpu"]
        job_datas = []
        for account in accounts:
            job_datas_account, colnames = jobs_data(account=account, cur_user=not args.users,
                next_chunks=args.next_chunks,
                nodes=args.nodes,
                excluded=args.exclude,
                submit_time=args.submit_time,
                eligible_time=args.eligible_time,
                queue_time=args.queue_time,
                heartbeat=args.heartbeat,
                heartbeat_analysis=args.heartbeat_analysis,
                latest_checkpoint=args.latest_checkpoint,
                verbose=args.verbose)
            if len(job_datas_account) > 0:
                colnames_job_data = argparse.Namespace(**{c: f"__account {account}" if c == "jobid" else c for c in colnames})
                job_datas += [colnames_job_data] + job_datas_account
    else:
        # On workstations, the obvious equivalent is finding free GPUs.
        print(subprocess.getoutput("python ~/.ScriptsAndAliases/FindFreeGPUs.py --solar 0"))
        time_str = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        print(time_str)
        sys.exit(0)
    
    col2max_chars = {c: len(c) for c in colnames}
    for job_data in job_datas:
        jd_dict = vars(job_data)
        for c in colnames:
            value = decolorize(str(jd_dict[c])) if c == "state" else str(jd_dict[c])
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
        
    job_datas = job_datas_with_to_prints(job_datas=job_datas, col2max_chars=col2max_chars)
    to_prints_lists = [j.to_print_list for j in job_datas]
    
    # Now, colorize some of the columns if they appear
    if "time_left" in col2max_chars:
        to_prints_lists = apply_along_column(to_prints_lists,
            colname_or_idx="TIME_LEFT",
            fn=partial(colorize_list, interpretaton_fn="delta", cutoffs="time_left"))
    if "queue_time" in col2max_chars:
        to_prints_lists = apply_along_column(to_prints_lists,
            colname_or_idx="QUEUE_TIME",
            fn=partial(colorize_list, interpretaton_fn="delta", cutoffs="queue_time"))
    if "start_time" in col2max_chars:
        to_prints_lists = apply_along_column(to_prints_lists,
            colname_or_idx="START_TIME",
            fn=partial(colorize_list, interpretaton_fn="time_away", cutoffs="start_time"))
    
    lines = ["  ".join(tpl) for tpl in to_prints_lists]
    lines = "\n".join(lines)
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
        job_datas = [jd for jd in job_datas if not jd.jobid.startswith("__")]
        record = build_record(job_datas=job_datas, account2lfs=account2lfs)
        _ = UtilsBase.atomic_save_lite(data=record, fname=args.record)

   




