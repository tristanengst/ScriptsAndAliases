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

# Specifies configuration for possible nodes/types of nodes, grouped by cluster.
# Commented out lines are for nodes not known to the scheduler.
#
# cpus_per_gpu  -- number of LOGICAL CPUs per GPU. Apparently SLURM interprets --cpus-per-task in terms of logical CPUs. This means that the Solar spreadsheet is wrong.
# mem_per_gpu   -- amount of memory per GPU
# gpu_alias      -- type of GPU (what we call it)
# gpu_name      -- type of GPU (what the scheduler calls it)
# gpus_per_node -- number of GPUs per node
# can_allocate  -- whether the node can be allocated
# max_time      -- maximum time in hours that can be requested
# constraint    -- constraint to use for the scheduler if possible
cluster2node2config = dict(
     solar={
        "cs-venus-01": dict(cpus_per_gpu=10, mem_per_gpu=84, gpu_alias="q6000", gpus_per_node=6, can_allocate=True, gpu_name="quadro_rtx_6000"),
        "cs-venus-03": dict(cpus_per_gpu=12, mem_per_gpu=64, gpu_alias="2080", gpus_per_node=4, can_allocate=True, gpu_name="2080_ti"),
        "cs-venus-05": dict(cpus_per_gpu=16, mem_per_gpu=60, gpu_alias="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000"),
        "cs-venus-06": dict(cpus_per_gpu=16, mem_per_gpu=60, gpu_alias="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000"),
        "cs-venus-07": dict(cpus_per_gpu=8, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        "cs-venus-08": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a100", gpus_per_node=4, can_allocate=True, gpu_name="a100"),
        "cs-venus-09": dict(cpus_per_gpu=7, mem_per_gpu=60, gpu_alias="a40", gpus_per_node=8, can_allocate=True, gpu_name="a40"),
        "cs-venus-13": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        "cs-venus-17": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
        "cs-venus-18": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s")},
    solar1={
        "cs-venus-02": dict(cpus_per_gpu=8, mem_per_gpu=64, gpu_alias="2080", gpus_per_node=8, can_allocate=True, gpu_name="2080_ti"),
        "cs-venus-12": dict(cpus_per_gpu=20, mem_per_gpu=128, gpu_alias="a6000", gpus_per_node=2, can_allocate=True, gpu_name="rtx_a6000"),
        "cs-venus-14": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40"),
        "cs-venus-15": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
        "cs-venus-16": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s"),
    },

    cedar=dict(default=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        v100l=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        p100l=dict(cpus_per_gpu=6, mem_per_gpu=56, gpu_alias="p100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        p100=dict(cpus_per_gpu=6, mem_per_gpu=30, gpu_alias="p100", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0)),

    narval=dict(default=dict(cpus_per_gpu=12, mem_per_gpu=124, gpu_alias="a100", gpus_per_node=4, can_allocate=True, gpu_name="a100", gpu_frac=1.0),
        a100=dict(cpus_per_gpu=12, mem_per_gpu=124, gpu_alias="a100", gpus_per_node=4, can_allocate=True, gpu_name="a100", gpu_frac=1.0),
        a101=dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_alias="a101", gpus_per_node=7, can_allocate=True, gpu_name="a100_1g.5gb", gpu_frac=0.125),
        a112=dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_alias="a112", gpus_per_node=4, can_allocate=True, gpu_name="a100_2g.10gb", gpu_frac=0.25),
        a123=dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_alias="a123", gpus_per_node=2, can_allocate=True, gpu_name="a100_3g.20gb", gpu_frac=0.5),
        a124=dict(cpus_per_gpu=6, mem_per_gpu=60, gpu_alias="a124", gpus_per_node=1, can_allocate=True, gpu_name="a100_4g.20gb", gpu_frac=0.5)),
    trillium=dict(default=dict(cpus_per_gpu=24, mem_per_gpu=186, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h100=dict(cpus_per_gpu=24, mem_per_gpu=188, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0)),
    fir=dict(default=dict(cpus_per_gpu=12, mem_per_gpu=280, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100"),
        h100=dict(cpus_per_gpu=12, mem_per_gpu=280, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h111=dict(cpus_per_gpu=6, mem_per_gpu=64, gpu_alias="h111", gpus_per_node=8, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpu_frac=0.125),
        h122=dict(cpus_per_gpu=6, mem_per_gpu=128, gpu_alias="h122", gpus_per_node=4, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpu_frac=0.25),
        h143=dict(cpus_per_gpu=6, mem_per_gpu=128, gpu_alias="h143", gpus_per_node=4, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpu_frac=0.5)),
    rorqual=dict(default=dict(cpus_per_gpu=16, mem_per_gpu=124, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h100=dict(cpus_per_gpu=16, mem_per_gpu=124, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h111=dict(cpus_per_gpu=8, mem_per_gpu=64, gpu_alias="h111", gpus_per_node=8, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpu_frac=0.125),
        h122=dict(cpus_per_gpu=8, mem_per_gpu=64, gpu_alias="h122", gpus_per_node=4, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpu_frac=0.25),
        h143=dict(cpus_per_gpu=8, mem_per_gpu=64, gpu_alias="h143", gpus_per_node=4, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpu_frac=0.5)),
    nibi=dict(default=dict(cpus_per_gpu=14, mem_per_gpu=248, gpu_alias="h100", gpus_per_node=8, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h100=dict(cpus_per_gpu=7, mem_per_gpu=250, gpu_alias="h100", gpus_per_node=8, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h111=dict(cpus_per_gpu=8, mem_per_gpu=62, gpu_alias="h111", gpus_per_node=16, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpu_frac=0.125),
        h122=dict(cpus_per_gpu=8, mem_per_gpu=124, gpu_alias="h122", gpus_per_node=8, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpu_frac=0.25),
        h143=dict(cpus_per_gpu=8, mem_per_gpu=124, gpu_alias="h143", gpus_per_node=8, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpu_frac=0.5)),
    cs_apex=dict(default=dict(cpus_per_gpu=8, mem_per_gpu=48, gpu_alias="3090", gpus_per_node=2, can_allocate=True, gpu_frac=1.0))
)

gpu2info = {
    "2080": dict(vram=8, good=False, gpu_name="2080_ti", ddp=True),
    "3090": dict(vram=24, good=True, gpu_name="3090", ddp=True)} | dict(
    p100=dict(vram=16, good=False, gpu_name="p100", ddp=True),
    p100l=dict(vram=32, good=False, gpu_name="p100l", ddp=True),
    q4000=dict(vram=8, good=False, gpu_name="quadro_rtx_4000", ddp=True),
    q6000=dict(vram=24, good=False, gpu_name="quadro_rtx_6000", ddp=True),
    v100=dict(vram=16, good=True, gpu_name="v100", ddp=True),
    v100l=dict(vram=32, good=True, gpu_name="v100l", ddp=True),
    a5000=dict(vram=24, good=True, gpu_name="rtx_a5000", ddp=True),
    a6000=dict(vram=48, good=True, gpu_name="rtx_a6000", ddp=True),
    a40=dict(vram=48, good=True, gpu_name="a40", ddp=True),
    a100=dict(vram=80 if Utils.is_solar() else 40, good=True, gpu_name="a100", ddp=True),
    a101=dict(vram=5, good=False, gpu_name="a100_1g.5gb", ddp=False),
    a112=dict(vram=10, good=False, gpu_name="a100_2g.10gb", ddp=False),
    a123=dict(vram=20, good=True, gpu_name="a100_3g.20gb", ddp=False),
    a124=dict(vram=20, good=True, gpu_name="a100_4g.20gb", ddp=False),
    l40s=dict(vram=48, good=True, gpu_name="l40s", ddp=True),
    h100=dict(vram=80, good=True, gpu_name="h100", ddp=True),
    h111=dict(vram=10, good=False, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", ddp=False),
    h122=dict(vram=20, good=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", ddp=False),
    h143=dict(vram=40, good=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", ddp=False))

gpu2vram = {k: v["vram"] for k,v in gpu2info.items()}
good_gpus = [k for k,v in gpu2info.items() if v["good"]]
bad_gpus = [k for k,v in gpu2info.items() if not v["good"]]
gpu_alias2name = {k: v["gpu_name"] for k,v in gpu2info.items()}
gpu_name2alias = {v["gpu_name"]: k for k,v in gpu2info.items()}





# # Maps GPU type to the amount of VRAM it has.
# gpu2vram = {"2080": 8, "3090": 24} | dict(
#     p100=16, p100l=16,
#     q4000=8, q6000=24,
#     v100=16, v100l=32,
#     a5000=24, a6000=48, a40=48,
#     a100=80 if Utils.is_solar() else 40, a112=10, a123=20, a124=20,
#     l40s=48,
#     h100=80, h111=10, h122=20, h143=40
# )

# good_gpus = ["3090", "v100", "v100l", "a5000", "a6000", "a40", "a100", "a123", "a124", "l40s", "h100", "h122", "h143"]
# bad_gpus = ["2080", "p100", "p100l", "q4000", "q6000", "a112", "h111"]
# gpu_alias2name = {"2080": "2080", "3090": "3090"} | dict(
#     h100="h100", h111="nvidia_h100_80gb_hbm3_1g.10gb", h122="nvidia_h100_80gb_hbm3_2g.20gb", h143="nvidia_h100_80gb_hbm3_3g.40gb",
#     l40s="l40s",
#     a100="a100", a112="a100_2g.10gb", a123="a100_3g.20gb", a124="a100_4g.20gb",
#     a6000="rtx_a6000", a5000="rtx_a5000", a40="a40",
#     v100="v100l", v100l="v100l",
#     p100="p100", p100l="p100l",
#     q6000="quadro_rtx_6000", q4000="quadro_rtx_4000")
# gpu_name2alias= {v: k for k,v in gpu_alias2name.items()}

# Maps cluster names to unique prefixes for their compute nodes
cluster2node_prefix = dict(narval="ng", cedar="cdr", solar="cs-venus", cs_apex="cs-apex", nibi="g", rorqual="rg", trillium="trig", fir="fc", solar1="cs-venus")

cluster2misc_reqs = dict(
    nibi=dict(wandb_default_mode="online",
        default_account="rrg-keli"),
    rorqual=dict(wandb_default_mode="online",
        default_account="def-keli"),
    fir=dict(wandb_default_mode="online",
        default_account="def-keli"),
    trillium=dict(wandb_default_mode="online",
        default_account="def-keli"),
    narval=dict(wandb_default_mode="online",
        default_account="def-keli"),
    cedar=dict(wandb_default_mode="online",
        default_account="def-keli"),
    solar=dict(wandb_default_mode="online",
        default_account="cs-gpu-research"),
    solar1=dict(wandb_default_mode="online",
        default_account="cs-gpu-research"),
    cs_apex=dict(wandb_default_mode="online",
        default_account=""))

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


def get_current_machine():
    """Returns the machine name of the current machine, or None if it can't be found."""
    return Utils.get_cluster_type() if Utils.is_slurm() else machine_to_ssh_name(os.uname().nodename)

def machine_to_ssh_name(m):
    """Returns the SSH-able name corresponding to the machine [m], or  None if it
    can't be found. The mapping should exist in your ~/.ssh/config file.
    """
    m = os.uname().nodename if m is None else m
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
    cwd = os.getcwd()
    os.chdir("/") # Not sure why this fixes an issue. Need to change back to the normal directory after running the command
    hostname = machine_to_hostname(m)
    if os.uname().nodename == hostname:
        result = subprocess.getoutput(command)
        os.chdir(cwd)
        return result
    ssh_name = machine_to_ssh_name(m)
    if ssh_name is None:
        raise ValueError(f"Could not find SSH name for machine {m} with hostname={hostname}. Please check your ~/.ssh/config file.")
    else:
        result = subprocess.getoutput(f"ssh {ssh_name} '{command}'")
        os.chdir(cwd)
        return result

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

        self.gpu_alias = gpu_name2alias[self.info.gres.split(":")[1]]
        self.gpu_vram = gpu2vram[self.gpu_alias]
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