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
import UtilsBase
from UtilsBase import twrite
from UserConfig import cluster2accounts


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
    "A5": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A5"]),
    # "A6": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A6"]),
    # "A7": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A7"]),
    "A8": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A8"]),
    "A9": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A9"]),
    "A99": dict(num_cpus=8, num_gpus=1, hyperthread=False, ssh_names=["A99", "emily"]),
} | dict( # Made up info for the ComputeCanada clusters so we can run commands on them. This should probably be refactored!
    narval=dict(num_cpus=40, num_gpus=4, hyperthread=False, ssh_names=["narval"]),
    cedar=dict(num_cpus=48, num_gpus=4, hyperthread=False, ssh_names=["cedar"]),
    killarney=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["killarney", "killa"]),
    vulcan=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["vulcan"]),
    trillium=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["trillium"]),
    fir=dict(num_cpus=48, num_gpus=4, hyperthread=False, ssh_names=["fir"]),
    rorqual=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["rorqual"]),
    nibi=dict(num_cpus=114, num_gpus=8, hyperthread=False, ssh_names=["nibi"]),)

# All ComputeCanada hosts. Filtering these out of [machine2info] is useful gives just
# workstation-y machines and Solar.
machines_cc = ["narval", "cedar", "killarney", "vulcan", "trillium", "fir", "rorqual", "nibi", "tamia"]

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
        "cs-gpu1": dict(cpus_per_gpu=10, mem_per_gpu=1, gpu_alias="titan", gpus_per_node=3, can_allocate=False, gpu_name="titan_xp", gpu_frac=1.0),
        "cs-gpu2": dict(cpus_per_gpu=8, mem_per_gpu=1, gpu_alias="1080ti", gpus_per_node=4, can_allocate=False, gpu_name=None, gpu_frac=1.0),
        "cs-gpu3": dict(cpus_per_gpu=4, mem_per_gpu=63, gpu_alias="2080", gpus_per_node=4, can_allocate=False, gpu_name="2080_ti", gpu_frac=1.0),
        "cs-venus-01": dict(cpus_per_gpu=10, mem_per_gpu=84, gpu_alias="q6000", gpus_per_node=6, can_allocate=True, gpu_name="quadro_rtx_6000", gpu_frac=1.0),
        "cs-venus-02": dict(cpus_per_gpu=8, mem_per_gpu=64, gpu_alias="2080", gpus_per_node=8, can_allocate=True, gpu_name="2080_ti", gpu_frac=1.0),
        "cs-venus-03": dict(cpus_per_gpu=12, mem_per_gpu=64, gpu_alias="2080", gpus_per_node=4, can_allocate=True, gpu_name="2080_ti", gpu_frac=1.0),
        "cs-venus-05": dict(cpus_per_gpu=16, mem_per_gpu=60, gpu_alias="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000", gpu_frac=1.0),
        "cs-venus-06": dict(cpus_per_gpu=16, mem_per_gpu=60, gpu_alias="a5000", gpus_per_node=8, can_allocate=True, gpu_name="rtx_a5000", gpu_frac=1.0),
        "cs-venus-07": dict(cpus_per_gpu=8, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40", gpu_frac=1.0),
        "cs-venus-08": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a100", gpus_per_node=4, can_allocate=True, gpu_name="a100", gpu_frac=1.0),
        "cs-venus-09": dict(cpus_per_gpu=7, mem_per_gpu=60, gpu_alias="a40", gpus_per_node=8, can_allocate=True, gpu_name="a40", gpu_frac=1.0),
        "cs-venus-12": dict(cpus_per_gpu=20, mem_per_gpu=128, gpu_alias="a6000", gpus_per_node=2, can_allocate=True, gpu_name="rtx_a6000", gpu_frac=1.0),
        "cs-venus-13": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40", gpu_frac=1.0),
        "cs-venus-14": dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="a40", gpus_per_node=4, can_allocate=True, gpu_name="a40", gpu_frac=1.0),
        "cs-venus-15": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        "cs-venus-16": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        "cs-venus-17": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        "cs-venus-18": dict(cpus_per_gpu=32, mem_per_gpu=240, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
    },

    cedar=dict(default=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        v100l=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        p100l=dict(cpus_per_gpu=6, mem_per_gpu=56, gpu_alias="p100l", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0),
        p100=dict(cpus_per_gpu=6, mem_per_gpu=30, gpu_alias="p100", gpus_per_node=4, can_allocate=True, extra_env_vars=dict(WANDB_DISABLE_SERVICE="'True'"), gpu_frac=1.0)),
    killarney=dict(default=dict(cpus_per_gpu=16, mem_per_gpu=255, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        l40s=dict(cpus_per_gpu=16, mem_per_gpu=255, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        h100=dict(cpus_per_gpu=6, mem_per_gpu=255, gpu_alias="h100", gpus_per_node=8, can_allocate=True, gpu_name="h100", gpu_frac=1.0)),
    vulcan=dict(default=dict(cpus_per_gpu=16, mem_per_gpu=125, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        l40s=dict(cpus_per_gpu=16, mem_per_gpu=125, gpu_alias="l40s", gpus_per_node=4, can_allocate=True, gpu_name="l40s", gpu_frac=1.0),
        l40s_shard=dict(cpus_per_gpu=1, mem_per_gpu=7, gpu_alias="l40s_shard", gpus_per_node=16, can_allocate=False, gpu_name="l40s_shard", gpu_frac=0.0625)),
    tamia=dict(
        default=dict(cpus_per_gpu=12, mem_per_gpu=125, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h100=dict(cpus_per_gpu=12, mem_per_gpu=125, gpu_alias="h100", gpus_per_node=4, can_allocate=True, gpu_name="h100", gpu_frac=1.0),
        h200=dict(cpus_per_gpu=8, mem_per_gpu=125, gpu_alias="h200", gpus_per_node=8, can_allocate=True, gpu_name="h200", gpu_frac=1.0)),
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
        h143=dict(cpus_per_gpu=8, mem_per_gpu=124, gpu_alias="h143", gpus_per_node=8, can_allocate=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpu_frac=0.5),
        mi300a=dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="mi300a", gpus_per_node=4, can_allocate=True, gpu_name="mi300a", gpu_frac=1.0),
        a5000=dict(cpus_per_gpu=16, mem_per_gpu=30.5, gpu_alias="a5000", gpus_per_node=4, can_allocate=True, gpu_name="a5000", gpu_frac=1.0),
        t4=dict(cpus_per_gpu=11, mem_per_gpu=45, gpu_alias="t4", gpus_per_node=4, can_allocate=True, gpu_name="t4", gpu_frac=1.0)),
    cs_apex=dict(default=dict(cpus_per_gpu=8, mem_per_gpu=48, gpu_alias="3090", gpus_per_node=2, can_allocate=True, gpu_frac=1.0))
)

gpu2info = {
    "default_gpu": dict(vram=0, good=False, gpu_name="default_gpu", ddp=True, gpu_frac=1),
    "titan": dict(vram=8, good=False, gpu_name="titan_xp", ddp=True, gpu_frac=1.0),
    "2080": dict(vram=8, good=False, gpu_name="2080_ti", ddp=True, gpu_frac=1.0),
    "3090": dict(vram=24, good=True, gpu_name="3090", ddp=True, gpu_frac=1.0)} | dict(
    t4=dict(vram=16, good=True, gpu_name="t4", ddp=True, gpu_frac=1.0),
    p100=dict(vram=16, good=False, gpu_name="p100", ddp=True, gpu_frac=1.0),
    p100l=dict(vram=32, good=False, gpu_name="p100l", ddp=True, gpu_frac=1.0),
    q4000=dict(vram=8, good=False, gpu_name="quadro_rtx_4000", ddp=True, gpu_frac=1.0),
    q6000=dict(vram=24, good=False, gpu_name="quadro_rtx_6000", ddp=True, gpu_frac=1.0),
    v100=dict(vram=16, good=True, gpu_name="v100", ddp=True, gpu_frac=1.0),
    v100l=dict(vram=32, good=True, gpu_name="v100l", ddp=True, gpu_frac=1.0),
    a5000=dict(vram=24, good=True, gpu_name="rtx_a5000" if Utils.is_solar() else "a5000", ddp=True, gpu_frac=1.0),
    a6000=dict(vram=48, good=True, gpu_name="rtx_a6000", ddp=True, gpu_frac=1.0),
    a40=dict(vram=48, good=True, gpu_name="a40", ddp=True, gpu_frac=1.0),
    a100=dict(vram=80 if Utils.is_solar() else 40, good=True, gpu_name="a100", ddp=True, gpu_frac=1.0),
    a101=dict(vram=5, good=False, gpu_name="a100_1g.5gb", ddp=False, gpu_frac=0.125),
    a112=dict(vram=10, good=False, gpu_name="a100_2g.10gb", ddp=False, gpu_frac=0.25),
    a123=dict(vram=20, good=True, gpu_name="a100_3g.20gb", ddp=False, gpu_frac=0.5),
    a124=dict(vram=20, good=True, gpu_name="a100_4g.20gb", ddp=False, gpu_frac=0.5),
    l40s=dict(vram=48, good=True, gpu_name="l40s", ddp=True, gpu_frac=1.0),
    l40s_shard=dict(vram=3, good=False, gpu_name="l40s_shard", ddp=False, gpu_frac=0.0625),
    h100=dict(vram=80, good=True, gpu_name="h100", ddp=True, gpu_frac=1.0),
    h111=dict(vram=10, good=False, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", ddp=False, gpu_frac=0.125),
    h122=dict(vram=20, good=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", ddp=False, gpu_frac=0.25),
    h143=dict(vram=40, good=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", ddp=False, gpu_frac=0.5),
    h200=dict(vram=141, good=True, gpu_name="h200", ddp=True, gpu_frac=1.0),
    mi300a=dict(vram=128, good=False, gpu_name="mi300a", ddp=True, gpu_frac=1.0))

gpu2vram = {k: v["vram"] for k,v in gpu2info.items()}
good_gpus = [k for k,v in gpu2info.items() if v["good"]]
bad_gpus = [k for k,v in gpu2info.items() if not v["good"]]
gpu_alias2name = {k: v["gpu_name"] for k,v in gpu2info.items()}
gpu_name2alias = {v["gpu_name"]: k for k,v in gpu2info.items()}

# Maps cluster names to unique prefixes for their compute nodes
cluster2node_prefix = dict(cs_apex="cs-apex", solar="cs-venus", # SFU-only
    beluga="bg",  cedar="cdr", # Deprecated
    nibi="g", fir="fc", rorqual="rg", narval="ng",  trillium="trig", # def/rrg
    vulcan="rack", killarney="kn", tamia="tg") # aip

cluster2misc_reqs = dict(
    nibi=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["nibi"][0], "_gpu")),
    rorqual=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["rorqual"][0], "_gpu")),
    fir=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["fir"][0], "_gpu")),
    narval=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["narval"][0], "_gpu")),
    trillium=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["trillium"][0], "_gpu")),
    vulcan=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["vulcan"][0], "_gpu")),
    killarney=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["killarney"][0], "_gpu")),
    tamia=dict(wandb_default_mode="offline",
        default_account=UtilsBase.strip_right(cluster2accounts["tamia"][0], "_gpu")),
    cedar=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["cedar"][0], "_gpu")),
    beluga=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["beluga"][0], "_gpu")),
    solar=dict(wandb_default_mode="online",
        default_account=UtilsBase.strip_right(cluster2accounts["solar"][0], "_gpu")),
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
        if len(line) >= 2 and not line[0].startswith("#"):
            k, v = line[:2]
            if k == "Host" and not v == "*":
                cur_host = v
            elif cur_host is None:
                pass
            else:
                machine2ssh_config[cur_host][k] = v
    return machine2ssh_config

class HostInfoError(Exception):
    """Custom exception for HostInfo-related errors."""
    pass

def get_current_machine():
    """Returns the machine name of the current machine, or None if it can't be found."""
    return Utils.get_cluster_type() if Utils.is_slurm() else machine_to_ssh_name(os.uname().nodename)

def to_ssh_name(x=None):
    """Returns an SSH-able name corresponding to machine/hostname/SSH name [x]. If no name can be determined, return None.
    """
    ssh_config = get_ssh_config()
    x = os.uname().nodename if x is None else x

    # CASE 1: [x] is an SSH-able name already
    if x in ssh_config:
        return x
    # CASE 2: [x] is the machine name of a machine in [machine2info], and it has a recorded SSH name in the user's config file
    if x in machine2info:
        possible_ssh_names = machine2info[x]["ssh_names"]
        matches = [p for p in possible_ssh_names if p in ssh_config]
        if matches:
            return matches[0]
    # CASE 3: [x] is a hostname already; we can hopefully just SSH to it directly. In this case, we can query the connection quickly.
    connection_test_command = f"ssh -o ConnectTimeout=3 {x} 'echo connected'"
    connection_test_result = subprocess.getoutput(connection_test_command)
    if connection_test_result == "connected":
        return x
    else:
        return None


def to_hostname(x=None):
    """Returns the hostname corresponding to machine/hostname/SSH name [x]. If no
    hostname can be determined, return None.
    """
    x = os.uname().nodename if x is None else x

    # CASE 1: [x] is an SSH-able name, so we should be able to read the hostname
    # directly from the ~/.ssh/config file.
    # CASE 2: [x] is a machine name in [machine2info]. In this case, one of its
    # [ssh_names] might be an entry in the user's ~/.ssh/config file. However, we
    # would've already tried this route in CASE 1, so nothing to do here.
    # CASE 3: [x] is a hostname already. In this case, it would be SSH-able, so we
    # would've already returned it in CASE 1.
    ssh_name = to_ssh_name(x)
    ssh_config = get_ssh_config()
    if ssh_name in ssh_config and not ssh_name is None and "HostName" in ssh_config[ssh_name]:
        return ssh_config[ssh_name]["HostName"]
    elif not ssh_name is None:
        return ssh_name
    else:
        return None

def to_machine_name(x=None):
    """Returns the machine name corresponding to machine/hostname/SSH name [x]. If no
    machine name can be determined, return None.
    """
    # CASE 1: is an SSH-able name, so we should be able to read the hostname
    # directly from the ~/.ssh/config file. From this we can determine the machine name.
    # CASE 2: [x] is a machine name in [machine2info]. In this case, we can just return it.
    # CASE 3: [x] is a hostname already. In this case, we can determine the machine name from it.
    x = os.uname().nodename if x is None else x
    if x in machine2info:
        return x
    
    ssh_name = to_ssh_name(x)
    ssh_config = get_ssh_config()
    if ssh_name in ssh_config and not ssh_name is None and "HostName" in ssh_config[ssh_name]:
        x = ssh_config[ssh_name]["HostName"]
        # Try again, this might work sometimes
        if x in machine2info:
            return x
    # Now [x] is either a hostname or an IP address or something else. Either way,
    # hostname_to_machine() will give us the machine name if possible.
    return hostname_to_machine(x)

        
def hostname_to_machine(hostname):
    """Returns the SSH name in [ssh_name2info] of [hostname]."""
    def extract_machine_from_true_hostname_heuristic(true_hostname):
        """Given a true hostname of form REDACTED, returns the machine name using a
        convoluted heuristic.
        """
        try:
            digits = [c for c in hostname if c.isdigit()]            
            prefix = true_hostname[8].upper()
            result = f"{prefix}{int(''.join(digits))}"
            return result if result in machine2info else None
        except:
            return None

    if hostname in cluster2node2config:
        return hostname
    elif all([c.isdigit() or c == "." for c in hostname]):
        # If it's an IP address, we can SSH to it and ask for the hostname directly
        command = f"ssh -o ConnectTimeout=3 {hostname} 'hostname'"
        result = subprocess.getoutput(command)
        return hostname_to_machine(result)
    else:
        return extract_machine_from_true_hostname_heuristic(hostname)
        

def hostname_is_current_machine(hostname):
    """Returns True if [hostname] is the current machine."""
    return os.uname().nodename == hostname

def run_command_on_machine(*, machine, command, ssh_args=[], **ssh_kwargs):
    """Runs [command] on machine [m] and returns the output."""
    cwd = os.getcwd()
    os.chdir("/") # Not sure why this fixes an issue. Need to change back to the normal directory after running the command
    if os.uname().nodename == to_hostname(machine):
        result = subprocess.getoutput(command)
        os.chdir(cwd)
        return result
    ssh_name = to_ssh_name(machine)
    if ssh_name is None:
        raise HostInfoError(f"Could not find SSH name for machine {machine}. Please check your ~/.ssh/config file.")
    else:
        ssh_args_str = " ".join(ssh_args)
        command_to_run = f"ssh {ssh_args_str} {ssh_name} '{command}'"
        result = subprocess.getoutput(command_to_run)
        os.chdir(cwd)
        return result

def get_updated_machine_info(m, verbose=0):
    """Returns a Namespace giving the nvidia-smi output, number of GPUs, and number
    CPU cores on machine [m].

    Args:
    m           -- machine name, which must be a key in [machine2info], or a hostname
    """
    try:
        result = run_command_on_machine(machine=m, command="nvidia-smi ; nvidia-smi --query-gpu=name --format=csv,noheader | wc -l ; nproc")
    except HostInfoError as e:
        twrite(f"Error: Could not connect to machine {m} to get updated machine info: {e}. Skipping.")
        return argparse.Namespace(nvidia_smi="",
            nvidia_smi_ok=False,
            total_gpus=0,
            total_cpus=0,
            user2num_gpus=dict())

    result = result.split("\n")
    nvidia_smi_lines = result[:-2]
    nvidia_smi_output = "\n".join(nvidia_smi_lines)
    
    # Find the total number of GPUs and if nvidia-smi is working
    m = to_machine_name(m)
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
        result = run_command_on_machine(machine=m, command="nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader")

        if len(result) == 0:
            user2gpu_ids = dict()
        else:
            for line in result.split("\n"):
                pid = line.split()[1].replace(",", "")
                user = run_command_on_machine(machine=m, command=f"ps -o user= -p {pid}").strip()
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