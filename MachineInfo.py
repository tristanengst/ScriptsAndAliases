"""Contains dictionaries of host information and methods for accessing them, all in one place.

Feel free to submit a pull request to add additional SSH names for hosts if you don't
use what's here. The only requirement is that no two hosts can share an SSH name.
"""
import argparse
from collections import defaultdict
import json
import math
import os
import os.path as osp
import sys
import subprocess

import Utils


# Dictionary machine names to their information. Note that hostnames are not included,
# and some functionality requires this. By assumption, for at least one element of
# [ssh_names] in each entry, you will have the corresponding Hostname or IP address in
#your ~/.ssh/config file.
#
# Each entry is thus:
#   num_cpus: Number of CPU cores on the host
#   num_gpus: Number of GPUs on the host
#   hyperthread: Whether the host has hyperthreading enabled
#   ssh_names: List of SSH all SSH names that anyone in our group uses to SSH into the
#   host (DO NOT add the actual hostname or IP address here)
machine2info = {
    "S1": dict(num_cpus=128, num_gpus=10, hyperthread=True, ssh_names=["S1"]),
    "S2": dict(num_cpus=128, num_gpus=10, hyperthread=True, ssh_names=["S2"]),
    "S3": dict(num_cpus=128, num_gpus=10, hyperthread=True, ssh_names=["S3"]),
    # "A1": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A1"]),
    # "A2": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A2"]),
    "A3": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A3"]),
    "A4": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A4"]),
    # "A5": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A5"]),
    # "A6": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A6"]),
    # "A7": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A7"]),
    "A8": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A8"]),
    "A9": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A9"]),
    "A99": dict(num_cpus=8, num_gpus=1, hyperthread=False, ssh_names=["A99", "emily"]),
}

def get_ssh_config():
    """Returns the SSH config file as a dictionary. It should be the case that each
    key is an element of 'ssh_names' for some machine in [machine2info], and that it
    will contain a 'HostName' entry for it.
    """
    ssh_config_file = osp.expanduser("~/.ssh/config")
    if not osp.exists(ssh_config_file):
        raise FileNotFoundError(f"Aborting: {ssh_config_file} not found. Please create it with the correct hosts.")

    with open(ssh_config_file, "r") as f:
        lines = f.readlines()

    cur_host = None
    machine2ssh_config = defaultdict(lambda: dict())
    for line in lines:
        line = line.strip().split()
        if len(line) == 2 and not line[0].startswith("#"):
            k, v = line
            if k == "Host" and not v == "*":
                cur_host = v
            elif cur_host is None:
                pass
            else:
                machine2ssh_config[cur_host][k] = v
    return machine2ssh_config

def machine_to_ssh_name(m):
    """Returns the SSH-able name corresponding to the machine [m], or  None if it
    can't be found. The mapping should exist in your ~/.ssh/config file.
    """
    machine2ssh_config = get_ssh_config()
    if m in machine2info:
        possible_ssh_names = machine2info[m]["ssh_names"]
    elif hostname_to_machine(m) in machine2info:
        possible_ssh_names = machine2info[hostname_to_machine(m)]["ssh_names"]
    else:
        raise ValueError()

    for m_user in machine2ssh_config:
        if m_user in possible_ssh_names:
            return m_user
    print(f"Couldn't find SSH name for machine {m}. Ensure that for some element of ssh_names={possible_ssh_names} of {m} in machine2info in HostInfo.py, there is a corresponding entry in your ~/.ssh/config file")
    return None
    
def machine_to_hostname(m):
    """Returns the SSH-able name of [m] or None if it can't be found."""
    ssh_config = get_ssh_config()
    ssh_name = machine_to_ssh_name(m)
    return ssh_config[ssh_name]["HostName"]

def hostname_to_machine(hostname):
    """Returns the SSH name in [ssh_name2info] of [hostname]."""
    hostname_numeric = [c for c in hostname if c.isdigit()]
    prefix = "S" if "r" in hostname else "A" # Heuristic, possibly brittle
    return f"{prefix}{int(''.join(hostname_numeric))}"

def hostname_is_current_machine(hostname):
    """Returns True if [hostname] is the current machine."""
    return os.uname().nodename == hostname

def run_command_on_machine(m, command):
    """Runs [command] on machine [m] and returns the output."""
    os.chdir("/") # Not sure why this fixes an issue
    hostname = machine_to_hostname(m)
    if os.uname().nodename == hostname:
        return subprocess.getoutput(command)
    ssh_name = machine_to_ssh_name(m)
    if ssh_name is None:
        raise ValueError(f"Could not find SSH name for machine {m} with hostname={hostname}. Please check your ~/.ssh/config file.")
    else:
        return subprocess.getoutput(f"ssh {ssh_name} '{command}'")

def get_updated_machine_info(m, verbose=0):
    """Returns a Namespace giving the nvidia-smi output, number of GPUs, and number
    CPU cores on machine [m].

    Args:
    m           -- machine name, which must be a key in [machine2info], or a hostname
    """
    m = hostname_to_machine(m) if not m in machine2info else m
    result = run_command_on_machine(m, "nvidia-smi ; nvidia-smi --query-gpu=name --format=csv,noheader | wc -l ; nproc")

    result = result.split("\n")
    nvidia_smi_lines = result[:-2]
    nvidia_smi_output = "\n".join(nvidia_smi_lines)
    
    # Find the total number of GPUs and if nvidia-smi is working
    if not len([idx for idx,l in enumerate(nvidia_smi_lines) if l.startswith("|=")]) == 2:
        if verbose:
            print(f"Error: {m} doesn't have any output. Probably nvidia-smi isn't working.\n\n{nvidia_smi_output}")
        nvidia_smi_ok = False
        total_gpus = machine2info[m]["num_gpus"]
        total_cpus = machine2info[m]["num_cpus"]
    else:
        nvidia_smi_ok = True
        total_gpus = int(result[-2])
        total_cpus = int(result[-1]) // (1 if machine2info[m]["hyperthread"] else 2)

    if nvidia_smi_ok:
        user2gpu_ids = defaultdict(lambda: set())
        result = run_command_on_machine(m, "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader")

        if len(result) == 0:
            user2gpu_ids = dict()
        else:
            for line in result.split("\n"):
                pid = line.split()[1].replace(",", "")
                user = run_command_on_machine(m, f"ps -o user= -p {pid}").strip()
                if not user == "":
                    user2gpu_ids[user].add(line.split()[0])

    else:
        user2gpu_ids = dict()

    user2num_gpus = {k: len(v) for k,v in user2gpu_ids.items()}
    return argparse.Namespace(nvidia_smi=nvidia_smi_output,
        nvidia_smi_ok=nvidia_smi_ok,
        total_gpus=total_gpus,
        total_cpus=total_cpus,
        user2num_gpus=user2num_gpus)

######################################################################################
# Useful for Solar
######################################################################################
slurm_gpu_name2gpu = dict(quadro_rtx_6000="q6000", rtx_a5000="a5000", a40="a40",
    a100="a100", rtx_a6000="a6000", l40s="l40s") | {"2080_ti": "rtx2080"}
gpu2vram = dict(q6000=24, a5000=24, a40=48, a100=80, a6000=48, l40s=48, rtx2080=8)

class SlurmNodeInfo:
    def __init__(self, s):
        def extract_key(*, s, key, default=None):
            """Returns the value of [key] from [s]. If it can be interpeted as an
            integer, it is returned as one. Further, if it can be interpreted as an
            integer number of space (ie. number ending with 'G' or 'M'), returns the
            number in GB.
            """
            kv = [t for t in s.split(",") if key in t]
            if len(kv) == 0 and default is None:
                raise ValueError(f"Couldn't kind key={key} in s={s}")
            elif len(kv) == 0:
                k, v = key, default
            elif len(kv) == 2:
                raise ValueError(f"Found multiple matches for key={key} in s={s}: {kv}")
            elif len(kv) == 1 and not kv[0].count("=") == 1:
                raise ValueError(f"Computed kv={kv[0]} from s={s}, but it did not contain exactly one equals sign")
            else:
                k, v = key, kv[0].split("=")[-1]

            if isinstance(v, str) and v.rstrip("G").isdigit():
                result = int(v.rstrip("G"))
            elif isinstance(v, str) and v.rstrip("M").isdigit():
                result = int(v.rstrip("M").isdigit() * 1024)
            elif isinstance(v, str) and  v.isdigit():
                result = int(v)
            else:
                result = v

            return result
            
        lines = s.split()
        info = {l.split("=")[0]: "=".join(l.split("=")[1:]) for l in lines}
        self.info = argparse.Namespace(**{k.lower(): v for k,v in info.items()})

        self.gpu_type = slurm_gpu_name2gpu[self.info.gres.split(":")[1]]
        self.gpu_vram = gpu2vram[self.gpu_type]
        self.alloc_gpus = extract_key(s=self.info.alloctres, key="gres/gpu", default=0)
        self.total_gpus = extract_key(s=self.info.cfgtres, key="gres/gpu")
        self.alloc_cpus = extract_key(s=self.info.alloctres, key="cpu", default=0) // 2
        self.total_cpus = extract_key(s=self.info.cfgtres, key="cpu") // 2
        self.alloc_mem = extract_key(s=self.info.alloctres, key="mem", default=0)
        self.total_mem = extract_key(s=self.info.cfgtres, key="mem")
        
        self.alloc_frac = max([self.alloc_gpus / self.total_gpus, self.alloc_cpus / self.total_cpus, self.alloc_mem / self.total_mem])
        self.alloc_gpus_eff = math.ceil(self.alloc_frac * self.total_gpus)
        self.free_gpus_eff = self.total_gpus - self.alloc_gpus_eff

    def __repr__(self): return f"{self.__class__.__name__}(nodename={self.info.nodename}, total_gpus={self.total_gpus}, free={self.free_gpus_eff} vram={self.gpu_vram})"

    @staticmethod
    def get_all_slurm_node_infos():
        s = "scontrol show nodes" if Utils.is_solar() else f"ssh {host_to_ssh_name('solar')} 'scontrol show nodes'"
        scontrol_show_nodes = subprocess.getoutput(s).split("\n\n")
        node_infos = [SlurmNodeInfo(s) for s in scontrol_show_nodes]
        return node_infos