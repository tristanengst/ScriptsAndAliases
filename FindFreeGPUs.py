"""Pretty prints information on available GPUs. Heuristically, GPUs that aren't
running a process with 'python' in the name are available.

TODO: Why are the last-used GPUs in a range sometimes reported wrong?
"""
import argparse
from collections import defaultdict
import math
from multiprocessing import Pool
import subprocess

import MachineInfo
import SSHCommunication
import Utils
from UtilsBase import twrite

# Sometimes this takes a bit, so tqdm is nice. But we can't assume it's installed.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

class NvidiaSMIError(Exception):
    """Error related to nvidia-smi command execution."""
    pass

def gpu_uid_to_index(h=None):
    """Returns a dictionary mapping GPU UIDs to their indices."""
    cmd = "nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        if_ssh_map_error="HostInfoError")
    result = result.strip() if result else None
    if result is None:
        return dict()
    elif not result[0].isdigit():
        raise NvidiaSMIError(f"nvidia-smi issue for host={h}: {result}")
    else:
        lines = [l.split(",") for l in result.strip().splitlines()]
        return {gpu_uid.strip(): int(gpu_idx.strip()) for gpu_idx, gpu_uid in lines}

def gpu_uid_to_procids(h=None):
    cmd = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        if_ssh_map_error="HostInfoError")
    result = result.strip() if result else None
    if result is None:
        return dict()
    elif not result.startswith("GPU-"):
        raise NvidiaSMIError(f"nvidia-smi issue for host={h}: {result}")
    else:
        lines = [l.split(",") for l in result.strip().splitlines()]
        result = defaultdict(list)
        for uid, pid in lines:
            result[uid].append(int(pid))
        return dict(result)

def procids_to_users(*procids, h=None):
    if len(procids) == 0:
        return dict()
    cmd = f"ps -o pid,user --no-headers -p {','.join([str(pid) for pid in procids])}"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        if_ssh_map_error="HostInfoError")
    if result is None:
        return dict()
    else:
        return {l.split()[0]: l.split()[1] for l in result.strip().splitlines()}

def gpu_index_to_users(h=None):
    try:
        gpu_uid2index = gpu_uid_to_index(h=h)
        gpu_uid2procids = gpu_uid_to_procids(h=h)
        gpu_index2procids = {gpu_uid2index[gpu_uid]: gpu_uid2procids.get(gpu_uid, []) for gpu_uid in gpu_uid2index.keys()}
        gpu_index2users = {gpu_idx: sorted(procids_to_users(*procids, h=h).values()) for gpu_idx, procids in gpu_index2procids.items()}
        return gpu_index2users
    except SSHCommunication.HostInfoError as e:
        return str(e)
    except NvidiaSMIError as e:
        return str(e)
    except Exception as e:
        twrite(f"Unexpected error for host={h}: {e}")
        return dict()

def machine_gpu_usage_summary_str(*, machine2gpu_index2users=None):
    if machine2gpu_index2users is None:
        machine2gpu_index2users = {SSHCommunication.get_machine_name(): gpu_index_to_users()}

    # Separate out the machines that had some issue, eg. SSH connection or nvidia-smi,
    # and handle them later.
    machine2error_strs = {m: gpu_index2users for m, gpu_index2users in machine2gpu_index2users.items() if isinstance(gpu_index2users, str)}
    machine2gpu_index2users = {m: gpu_index2users for m, gpu_index2users in machine2gpu_index2users.items() if not m in machine2error_strs}

    machine2num_free_gpus = {m: sum([len(users) == 0 for users in gpu_index2users.values()]) for m, gpu_index2users in machine2gpu_index2users.items()}
    machine2gpu_index2users = sorted(machine2gpu_index2users.items(), key=lambda x: machine2num_free_gpus[x[0]], reverse=True)
    
    machine2usage_strs = dict()
    for machine, gpu_index2users in machine2gpu_index2users:
        if len(gpu_index2users) == 0:
            machine2usage_strs[machine] = "No GPU information found -> likely SSH map or SSH connection issue"
            continue
        total_gpus = len(gpu_index2users)
        free_gpus = sorted([gpu_idx for gpu_idx, users in gpu_index2users.items() if len(users) == 0])
        free_gpu_str = f"Free GPUS: {len(free_gpus)}/{total_gpus}\t"

        if free_gpus:
            free_gpu_idxs_str = ",".join([str(gpu_idx) for gpu_idx in free_gpus])
            free_gpu_str += f"IDs= {free_gpu_idxs_str}"

        gpu_range2users = dict()
        range_start = min(gpu_index2users.keys()) if len(gpu_index2users) > 0 else 0
        cur_gpu = range_start
        cur_users = tuple(gpu_index2users[cur_gpu])
        for gpu_idx in sorted(gpu_index2users.keys()):
            gpu_users = tuple(gpu_index2users[gpu_idx])
            if gpu_users == cur_users and not int(gpu_idx) == len(gpu_index2users) - 1:
                cur_gpu = gpu_idx
            else:
                cur_users_str = ",".join(cur_users) if len(cur_users) > 0 else "FREE"
                range_end = gpu_idx if gpu_idx == len(gpu_index2users) - 1 else cur_gpu
                gpu_range2users[(range_start, range_end)] = cur_users_str
                range_start = gpu_idx
                cur_gpu = gpu_idx
                cur_users = gpu_users

        gpu_range_strs = []
        for gpu_range, users in gpu_range2users.items():
            gpu_range_str = f"{gpu_range[0]}" if gpu_range[0] == gpu_range[1] else f"{gpu_range[0]}-{gpu_range[1]}"
            gpu_range_strs.append(f"{gpu_range_str}: {users}")
        gpu_range_str = "\t\t\t(" + "\t|\t".join(gpu_range_strs) + ")"

        machine2usage_strs[machine] = free_gpu_str + " " * 20 + gpu_range_str

    machine2usage_str = "\t" + "\n\t\t\t".join([f"{machine}: {usage_str}" for machine, usage_str in machine2usage_strs.items()])

    ##################################################################################
    # Now add in the machines that hand an error
    ##################################################################################
    for machine, error_str in machine2error_strs.items():
        machine_str = f"{machine}: GPU info unavailable: "
        error_str = error_str.replace("\n", "\n\t\t\t" + " " * len(machine_str))
        machine2usage_str += f"\n\t\t\t{machine_str}{error_str}"

    twrite(machine2usage_str)
    return machine2usage_str


excluded_hosts = MachineInfo.machines_cc + ["solar"]
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--hosts", type=str, nargs="*", default=MachineInfo.machine2info.keys(), choices=list(MachineInfo.machine2info.keys()),)
    P.add_argument("-c", "--current", action="store_true", help="Only check the current machine")
    args = P.parse_args()
    args.hosts = [SSHCommunication.get_machine_name()] if args.current else [h for h in args.hosts if h not in excluded_hosts]

    with Pool(processes=min(16, len(args.hosts))) as p:
        gpuindex2users = p.map(gpu_index_to_users, args.hosts, chunksize=math.ceil(len(args.hosts) / 16))
    machine2gpu_index2users = {h: gpu_index2users for h, gpu_index2users in zip(args.hosts, gpuindex2users)}

    _ = machine_gpu_usage_summary_str(machine2gpu_index2users=machine2gpu_index2users)