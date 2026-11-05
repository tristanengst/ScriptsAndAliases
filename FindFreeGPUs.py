"""Pretty prints information on available GPUs. Heuristically, GPUs that aren't
running a process with 'python' in the name are available.

TODO: Why are the last-used GPUs in a range sometimes reported wrong?
"""
import argparse
from collections import defaultdict
import math
from multiprocessing import Pool
import os
import os.path as osp
import re
import subprocess
import tempfile
import uuid

import MachineInfo
import SSHCommunication
import Utils
from UtilsBase import twrite, colorize

# Sometimes this takes a bit, so tqdm is nice. But we can't assume it's installed.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

class NvidiaSMIError(Exception):
    """Error related to nvidia-smi command execution."""
    pass

def split_broken_gpus(result):
    """Returns [result] with the lines nvidia-smi prints for GPUs it can't reach (eg.
    one that has fallen off the bus) removed, and the indices of those GPUs. Such lines
    look like 'Unable to determine the device handle for GPU1: 0000:0A:00.0: Unknown Error'.
    """
    lines, broken_gpu_idxs = [], []
    for l in result.splitlines():
        m = re.match(r"Unable to determine the device handle for GPU(\d+)", l.strip())
        if m is None:
            lines.append(l)
        else:
            broken_gpu_idxs.append(int(m.group(1)))
    return "\n".join(lines).strip(), sorted(broken_gpu_idxs)

def gpu_uid_to_index(h=None, ssh_args=()):
    """Returns a dictionary mapping GPU UIDs to their indices. GPUs nvidia-smi can't
    reach have no UID, so they get a placeholder one starting with 'broken-'.
    """
    cmd = "nvidia-smi --query-gpu=index,gpu_uuid,name --format=csv,noheader"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        ssh_args=ssh_args)
    result = result.strip() if result else None
    if result is None:
        return dict()
    result, broken_gpu_idxs = split_broken_gpus(result)
    if result and not result[0].isdigit():
        raise NvidiaSMIError(f"nvidia-smi issue for host={h}: {result}")
    elif not result and not broken_gpu_idxs:
        return dict()
    else:
        lines = [[x.strip() for x in l.split(",")] for l in result.splitlines()]
        lines += [(str(gpu_idx), f"broken-GPU{gpu_idx}", "broken") for gpu_idx in broken_gpu_idxs]
        # A99 has a GTX 1080 Ti alongside its RTX 2080 SUPER, but the 1080 Ti isn't
        # worth considering, so hide it
        if (h or SSHCommunication.get_machine_name()) == "A99":
            lines = [(gpu_idx, gpu_uid, name) for gpu_idx, gpu_uid, name in lines if not "1080" in name]
        return {gpu_uid: int(gpu_idx) for gpu_idx, gpu_uid, _ in lines}

def gpu_uid_to_procids(h=None, ssh_args=()):
    cmd = "nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        ssh_args=ssh_args)
    result = split_broken_gpus(result)[0] if result else None
    if not result:
        return dict()
    elif not result.startswith("GPU-"):
        raise NvidiaSMIError(f"nvidia-smi issue for host={h}: {result}")
    else:
        lines = [l.split(",") for l in result.strip().splitlines()]
        result = defaultdict(list)
        for uid, pid in lines:
            if pid.strip().isdigit():
                result[uid].append(int(pid)) # Needs to be integer, since it represents an actual process ID we will use to look up the user
            else:
                pass # Ignore non-integer PIDs; hopefully we can figure out that the GPU is weird another way
        return dict(result)

def procids_to_users(*procids, h=None, ssh_args=()):
    if len(procids) == 0:
        return dict()
    cmd = f"ps -o pid,user --no-headers -p {','.join([str(pid) for pid in procids])}"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        ssh_args=ssh_args)
    if result is None:
        return dict()
    else:
        return {l.split()[0]: l.split()[1] for l in result.strip().splitlines()}

def gpu_index_to_errors(h=None, ssh_args=()):
    # cmd = "nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits"
    cmd = "nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total,utilization.gpu --format=csv,noheader,nounits"
    result = SSHCommunication.run_command_on_machine(machine=h, command=cmd,
        if_connect_error="HostInfoError",
        ssh_args=ssh_args)
    result = result.strip() if result else None
    if result is None:
        return dict()

    # Lines for reachable GPUs are in index order but skip the broken ones' indices
    result, broken_gpu_idxs = split_broken_gpus(result)
    lines = result.splitlines()
    gpu_idxs = [i for i in range(len(lines) + len(broken_gpu_idxs)) if not i in broken_gpu_idxs]
    gpu_idx2error = {gpu_idx: True for gpu_idx in broken_gpu_idxs}
    for gpu_idx, line in zip(gpu_idxs, lines):
        split_line = [ll.strip() for ll in line.strip().split(",")]
        if not len(split_line) == 2:
            gpu_idx2error[gpu_idx] = True
        else:
            ecc_error_str, gpu_utilization_str = split_line
            ecc_error = not ecc_error_str in ["0", "[N/A]"]
            gpu_utilization_error = not gpu_utilization_str.isdigit() # Presently heuristic! Maybe better to just check for ['N/A']?
            gpu_idx2error[gpu_idx] = ecc_error or gpu_utilization_error
    return gpu_idx2error
    
def gpu_index_to_users(h=None):
    # All SSH calls for [h] share one connection over a socket private to this call, so
    # nothing is reused from other processes (eg. a stale ~/.ssh/config ControlMaster).
    # ControlPersist closes it 10s after its last use, even if this process is killed.
    # LogLevel=error keeps login banners and known-hosts warnings out of command output
    # if a call ever opens a new connection instead of reusing this one.
    ctrl_path = osp.join(tempfile.gettempdir(), f"ffg-{os.getpid()}-{uuid.uuid4().hex[:8]}-%C")
    ssh_args = ("-o ControlMaster=auto", f"-o ControlPath={ctrl_path}", "-o ControlPersist=10", "-o LogLevel=error")
    try:
        gpu_uid2index = gpu_uid_to_index(h=h, ssh_args=ssh_args)
        gpu_uid2procids = gpu_uid_to_procids(h=h, ssh_args=ssh_args)
        gpu_index2errors = gpu_index_to_errors(h=h, ssh_args=ssh_args)
        gpu_index2procids = {gpu_uid2index[gpu_uid]: gpu_uid2procids.get(gpu_uid, []) for gpu_uid in gpu_uid2index.keys()}
        gpu_index2users = {gpu_idx: sorted(procids_to_users(*procids, h=h, ssh_args=ssh_args).values()) for gpu_idx, procids in gpu_index2procids.items()}
        gpu_index2users = {gpu_idx: users + (["error"] if gpu_index2errors.get(gpu_idx, False) else []) for gpu_idx, users in gpu_index2users.items()}
        gpu_index2users = {gpu_idx: sorted(set(users)) for gpu_idx, users in gpu_index2users.items()} # Remove duplicates and sort
        return gpu_index2users
    except SSHCommunication.HostInfoError as e:
        return str(e)
    except NvidiaSMIError as e:
        return str(e)
    except Exception as e:
        twrite(f"Unexpected error for host={h}: {e}")
        return dict()

def can_use_slurm():
    """Returns if the current user can use SLURM here: its commands are installed and
    the controller answers them. On the APEX workstations only the user running SLURM
    can reach munged, so the controller rejects anyone else's messages.
    """
    try:
        return subprocess.run(["sinfo", "-h", "-o", "%N"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False

def machines_to_slurm_states(*machines):
    """Returns a dictionary mapping each of [machines] to its node's state in SLURM, with
    the reason if there is one (eg. why it's drained), or 'none' if it isn't a SLURM node.
    Returns None if the current user can't use SLURM here.
    """
    if not can_use_slurm():
        return None
    result = subprocess.run(["sinfo", "-N", "-h", "-o", "%N|%T|%E"], capture_output=True, text=True, timeout=10)
    node2state = dict()
    for l in result.stdout.strip().splitlines(): # A node is listed once per partition
        node, state, reason = l.split("|", 2)
        node2state[node] = state if reason == "none" else f"{state} ({reason})"
    return {m: node2state.get(SSHCommunication.machine2hostname.get(m, m).split(".")[0], "none") for m in machines}

def machine_gpu_usage_summary_str(*, machine2gpu_index2users=None, machine2slurm_state=None, color=True):
    """Returns and prints a summary of GPU usage on each machine in
    [machine2gpu_index2users], with its SLURM state from [machine2slurm_state] if given.
    """
    if machine2gpu_index2users is None:
        machine2gpu_index2users = {SSHCommunication.get_machine_name(): gpu_index_to_users()}

    machine2slurm_state = dict() if machine2slurm_state is None else machine2slurm_state
    machine_usage_strs = [MachineUsageStr(machine=m, gpu_index2users=gpu_index2users, slurm_state=machine2slurm_state.get(m))
        for m, gpu_index2users in machine2gpu_index2users.items()]
    machine_usage_strs = sorted(machine_usage_strs, key=lambda mus: mus.usability_score, reverse=True)
    max_property_lens = MachineUsageStr.get_max_property_lens(machine_usage_strs)
    usage_strs = [mus.usage_str(color=color, **max_property_lens) for mus in machine_usage_strs]
    machine2usage_str = "\t" + "\n\t\t\t".join(usage_strs)
    twrite(machine2usage_str)

    return machine2usage_str


class MachineUsageStr:
    """Data about a machine's current GPU usage and its string representation."""
    def __init__(self, *, machine, gpu_index2users, slurm_state=None):
        self.machine = machine
        self.gpu_index2users = gpu_index2users
        self.slurm_state = slurm_state
        self.slurm_state_str = "" if slurm_state is None else f"slurm: {slurm_state}"

        ##############################################################################
        # If [gpu_index2users] is a string, then it would indicate some sort of error.
        # Otherwise if a dictionary, then it indicates a GPU index -> user map
        ##############################################################################
        if isinstance(gpu_index2users, dict):
            self.total_gpus = len(gpu_index2users)
            self.free_gpu_idxs = sorted([gpu_idx for gpu_idx, users in gpu_index2users.items() if len(users) == 0])
            self.free_gpus = len(self.free_gpu_idxs)
            self.free_gpu_str = f"free: {self.free_gpus}/{self.total_gpus}"

            if self.free_gpu_idxs:
                free_gpu_idxs_str = ",".join([str(gpu_idx) for gpu_idx in self.free_gpu_idxs])
                self.free_gpu_idxs_str = f"IDs= {free_gpu_idxs_str}"
            else:
                self.free_gpu_idxs_str = ""

            # Group consecutive GPU indices with the same users into ranges. Indices
            # needn't start at zero or be contiguous, eg. if a machine restricts its GPUs
            self.gpu_range2users = dict()
            for gpu_idx in sorted(gpu_index2users.keys()):
                users = tuple(gpu_index2users[gpu_idx])
                if self.gpu_range2users:
                    (s, e), last_users = list(self.gpu_range2users.items())[-1]
                    if e == gpu_idx - 1 and last_users == users:
                        del self.gpu_range2users[(s, e)]
                        self.gpu_range2users[(s, gpu_idx)] = users
                        continue
                self.gpu_range2users[(gpu_idx, gpu_idx)] = users

            self.gpu_range2users = {(s,e): ",".join(users) if len(users) > 0 else "free" for (s,e), users in self.gpu_range2users.items()}

            self.gpu_range_strs = []
            for gpu_range, users in self.gpu_range2users.items():
                gpu_range_str = f"{gpu_range[0]}" if gpu_range[0] == gpu_range[1] else f"{gpu_range[0]}-{gpu_range[1]}"
                self.gpu_range_strs.append(f"{gpu_range_str}: {users}")

            self.gpu_information_avail = True
       
        elif isinstance(gpu_index2users, str):
            self.gpu_information_avail = False
            self.gpu_range2users = dict()
            self.total_gpus = 0
            self.free_gpus = 0
            self.free_gpu_idxs = []
            self.free_gpu_str = "free: N/A"
            self.free_gpu_idxs_str = ""
            self.gpu_range_strs = []
            self.gpu_information_avail = False
        else:
            raise NotImplementedError()

        # Proxy for how much we would like to use a GPU in question. Higher is better
        self.usability_score = self.free_gpus if self.gpu_information_avail else -1
            

    def __repr__(self):
        return self.__class__.__name__ + f"(machine={self.machine}, gpu_index2users={self.gpu_index2users}, slurm_state={self.slurm_state})"
    def __str__(self): return self.__repr__()

    @property
    def free_gpu_str_len(self): return len(self.free_gpu_str) if self.free_gpu_str else 0
    @property
    def free_gpu_idxs_str_len(self): return len(self.free_gpu_idxs_str) if self.free_gpu_idxs_str else 0
    @property
    def gpu_range_str_num(self): return len(self.gpu_range_strs) if self.gpu_range_strs else 0
    @property
    def gpu_range_str_max_len(self): return max([len(s) for s in self.gpu_range_strs]) if self.gpu_range_strs else 0

    def usage_str(self, *, color=False, max_machine_name_str_len=0, max_free_gpu_str_len=0, max_free_gpu_idxs_str_len=0, max_gpu_range_str_max_len=0,
        max_slurm_state_str_len=0):
        max_machine_name_str_len = max(max_machine_name_str_len, len(self.machine))
        max_free_gpu_str_len = max(max_free_gpu_str_len, len(self.free_gpu_str))
        max_free_gpu_idxs_str_len = max(max_free_gpu_idxs_str_len, len(self.free_gpu_idxs_str))
        max_gpu_range_str_max_len = max(max_gpu_range_str_max_len, self.gpu_range_str_max_len)

        machine_str = self.machine.ljust(max_machine_name_str_len)
        free_gpu_str = self.free_gpu_str.ljust(max_free_gpu_str_len)
        free_gpu_idxs_str = self.free_gpu_idxs_str.ljust(max_free_gpu_idxs_str_len)
        gpu_range_strs = [s.ljust(max_gpu_range_str_max_len) for s in self.gpu_range_strs]
        slurm_state_str = self.slurm_state_str.ljust(max(max_slurm_state_str_len, len(self.slurm_state_str)))
        # Red for states in which new jobs won't start on a node even once its resources
        # free up, including sinfo's suffixes for not responding (*), powering down (%, !),
        # in a maintenance reservation ($) and rebooting (@, ^). 'none' isn't a SLURM node
        state = "" if self.slurm_state is None else self.slurm_state.split(" (")[0]
        slurm_state_bad = state.endswith(tuple("*%!$@^")) or state.rstrip("*~#!%$@^-").startswith(("down", "drain",
            "fail", "maint", "reserved", "inval", "unknown", "future", "power_down", "powering_down", "reboot", "none"))
        slurm_state_str = colorize(slurm_state_str, color="red" if slurm_state_bad else "green") if color and state else slurm_state_str
        slurm_state_str = f"{slurm_state_str}\t" if self.slurm_state_str else ""

        if self.gpu_information_avail:
            if color:

                if self.free_gpus == self.total_gpus:
                    free_gpus_amount_color = "green"
                elif self.free_gpus > 0:
                    free_gpus_amount_color = "orange"
                else:
                    free_gpus_amount_color = "red"

                machine_str = colorize(machine_str, color=free_gpus_amount_color)
                free_gpu_str = colorize(free_gpu_str, color=free_gpus_amount_color)
                free_gpu_idxs_str = colorize(free_gpu_idxs_str, color=free_gpus_amount_color)

                gpu_range_colors = []
                for user_str in self.gpu_range_strs:
                    if user_str.split(" ")[-1] == "free":
                        gpu_range_colors.append("green")
                    elif "error" in user_str:
                        gpu_range_colors.append("red")
                    else:
                        gpu_range_colors.append("orange")
                gpu_range_strs = [colorize(s, color=c) for s, c in zip(gpu_range_strs, gpu_range_colors)]

            gpu_range_str = " | ".join(gpu_range_strs)
            return f"{machine_str}\t{slurm_state_str}{free_gpu_str}\t{free_gpu_idxs_str}\t| {gpu_range_str}"

        else:
            s = "No GPU information found -> likely SSH map or SSH connection issue"
            machine_str, s = (colorize(machine_str, color=88), colorize(s, color=88)) if color else (machine_str, s)
            return f"{machine_str}\t{slurm_state_str}{s}"

    @staticmethod
    def get_max_property_lens(machine_usage_strs):
        max_machine_name_str_len = max([len(mus.machine) for mus in machine_usage_strs])
        max_free_gpu_str_len = max([mus.free_gpu_str_len for mus in machine_usage_strs])
        max_free_gpu_idxs_str_len = max([mus.free_gpu_idxs_str_len for mus in machine_usage_strs])
        max_gpu_range_str_max_len = max([mus.gpu_range_str_max_len for mus in machine_usage_strs])
        max_slurm_state_str_len = max([len(mus.slurm_state_str) for mus in machine_usage_strs])
        return dict(max_machine_name_str_len=max_machine_name_str_len,
            max_free_gpu_str_len=max_free_gpu_str_len,
            max_free_gpu_idxs_str_len=max_free_gpu_idxs_str_len,
            max_gpu_range_str_max_len=max_gpu_range_str_max_len,
            max_slurm_state_str_len=max_slurm_state_str_len)

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

    _ = machine_gpu_usage_summary_str(machine2gpu_index2users=machine2gpu_index2users,
        machine2slurm_state=machines_to_slurm_states(*args.hosts))
