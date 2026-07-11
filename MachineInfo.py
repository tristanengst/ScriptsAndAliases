"""Contains dictionaries of host information and methods for accessing them, all in one place.

Feel free to submit a pull request to add additional SSH names for hosts if you don't
use what's here. The only requirement is that no two hosts can share an SSH name.
"""
import argparse
from collections import defaultdict
from functools import lru_cache
import json
import math
import os
import os.path as osp
import sys
import subprocess

import SSHCommunication
import Utils
import UtilsBase
from UtilsBase import twrite
from UserConfig import cluster2accounts

# Information about GPUs. Keys are GPU aliases (shorthand/nice names for GPUs). Values are:
# vram              -- amount of VRAM in GB, useful for heuristic computations
# gen               -- generation of the GPU, useful for heuristic computations.
#                     Basically the leading digit of the generation's gaming GPUs,
#                     or made-up as needed.
# good              -- overall judgement on if the GPU is good (worth displaying in sqb) or not
# gpu_name          -- name of the GPU as per SLURM, often not what we'd call it
# ddp               -- whether the GPU can be used with DDP
# gpu_frac        -- how many fractions of a regular GPU's VRAM it takes up
# rgus_per_gpu    -- multiplier for RGUs when using this GPU type. See
#                       docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling.
#                        Where estimated, should be read as essentially made-up. A
#                        value over of 3.0+ generally indicates a decent GPU.
# NOTES ON GPU NAMING:
# For multi-instance GPUs, the pattern is given by (for example)
# H100-3g.40gb -> H143, ie. replacing the last two digits with first the amount of
# VRAM in 10GB units, and then the number of SMs out of however many MiG GPUs have
# them divided into.
#
# NOTES ON RGUS_PER_GPU:
# While the rgus_per_gpu values here are direct from ComputeCanada documentation, for
# MIG GPUs (at least?), they actually DON'T accurately predict job's billing. Instead,
# values visible on CCDB actually do! The difference is most significant for H143s:
# the documentation says 6.1 RGUs per GPU, but CCDB shows 5.23, and this is consistent
# with the billing reported by SLURM. Fun stuff.
gpu2info = dict(
    default_gpu=dict(vram=0, gen=0, good=False, gpu_name="default_gpu", ddp=True, gpu_frac=1, rgus_per_gpu=1.0),
    titan=dict(vram=8, gen=1, good=False, gpu_name="titan_x", ddp=True, gpu_frac=1.0, rgus_per_gpu=0.1),
    t4=dict(vram=16, gen=2, good=True, gpu_name="t4", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.3),
    # p100=dict(vram=16, gen=1, good=False, gpu_name="p100", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.0),
    # p100l=dict(vram=32, gen=1, good=False, gpu_name="p100l", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.1),
    q4000=dict(vram=8, gen=2, good=False, gpu_name="quadro_rtx_4000", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.2),
    q6000=dict(vram=24, gen=2, good=False, gpu_name="quadro_rtx_6000", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.3),
    v100=dict(vram=16, gen=2, good=True, gpu_name="v100", ddp=True, gpu_frac=1.0, rgus_per_gpu=2.2),
    v100l=dict(vram=32, gen=2, good=True, gpu_name="v100l", ddp=True, gpu_frac=1.0, rgus_per_gpu=2.6),
    a5000=dict(vram=24, gen=3, good=True, gpu_name="rtx_a5000" if Utils.is_solar() else "a5000", ddp=True, gpu_frac=1.0, rgus_per_gpu=3.0),
    a6000=dict(vram=48, gen=3, good=True, gpu_name="a6000", ddp=True, gpu_frac=1.0, rgus_per_gpu=3.1),                   # Estimated rgus_per_gpu
    a6000a=dict(vram=48, gen=4, good=True, gpu_name="rtx_6000_ada", ddp=True, gpu_frac=1.0, rgus_per_gpu=3.4),                   # Estimated rgus_per_gpu
    rtx6000=dict(vram=96, gen=5, good=True, gpu_name="rtx_pro_6000_blackwell_se", ddp=True, gpu_frac=1., rgus_per_gpu=4),                   # Estimated rgus_per_gpu
    a40=dict(vram=48, gen=1, good=True, gpu_name="a40", ddp=True, gpu_frac=1, rgus_per_gpu=3.2),                           # Estimated rgus_per_gpu
    a100=dict(vram=80 if Utils.is_solar() else 40, good=True, gpu_name="a100", ddp=True, gpu_frac=1.0, rgus_per_gpu=4.0),
    a101=dict(vram=5, good=False, gpu_name="a100_1g.5gb", ddp=False, gpu_frac=0.125, rgus_per_gpu=0.57),
    a112=dict(vram=10, good=False, gpu_name="a100_2g.10gb", ddp=False, gpu_frac=0.25, rgus_per_gpu=1.14),
    a123=dict(vram=20, good=True, gpu_name="a100_3g.20gb", ddp=False, gpu_frac=0.5, rgus_per_gpu=2.0),
    a124=dict(vram=20, good=True, gpu_name="a100_4g.20gb", ddp=False, gpu_frac=0.5, rgus_per_gpu=2.3),
    l40s=dict(vram=48, good=True, gpu_name="l40s", ddp=True, gpu_frac=1.0, rgus_per_gpu=3.4),                          # Estimated rgus_per_gpu
    l40s_shard=dict(vram=3, good=False, gpu_name="l40s_shard", ddp=False, gpu_frac=0.0625, rgus_per_gpu=0.3),          # Estimated rgus_per_gpu
    h100=dict(vram=80, good=True, gpu_name="h100", ddp=True, gpu_frac=1.0, rgus_per_gpu=12.2),    
    h111=dict(vram=10, good=False, gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", ddp=False, gpu_frac=0.125, rgus_per_gpu=1.74),
    h122=dict(vram=20, good=True, gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", ddp=False, gpu_frac=0.25, rgus_per_gpu=3.48),
    h143=dict(vram=40, good=True, gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", ddp=False, gpu_frac=0.5, rgus_per_gpu=6.1),
    h200=dict(vram=141, good=True, gpu_name="h200", ddp=True, gpu_frac=1.0, rgus_per_gpu=20.0),                        # Estimated rgus_per_gpu
    mi300a=dict(vram=128, good=False, gpu_name="mi300a", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.0),                    # Estimated rgus_per_gpu
) | {
    "default_gpu": dict(vram=0, gen=0, good=False, gpu_name="default_gpu", ddp=True, gpu_frac=1, rgus_per_gpu=1.0),          # Placeholder values
    "titan": dict(vram=8, gen=1, good=False, gpu_name="titan_x", ddp=True, gpu_frac=1.0, rgus_per_gpu=0.1),                 # Estimated rgus_per_gpu
    "2080": dict(vram=8, gen=2, good=False, gpu_name="2080_ti", ddp=True, gpu_frac=1.0, rgus_per_gpu=1.2),                   # Estimated rgus_per_gpu
    "3090": dict(vram=24, gen=3, good=True, gpu_name="3090", ddp=True, gpu_frac=1.0, rgus_per_gpu=3.0),
    }
gpu2info = {k: argparse.Namespace(**v) for k,v in gpu2info.items()}

gpu2vram = {k: v.vram for k,v in gpu2info.items()}
good_gpus = [k for k,v in gpu2info.items() if v.good]
bad_gpus = [k for k,v in gpu2info.items() if not v.good]
gpu_alias2name = {k: v.gpu_name for k,v in gpu2info.items()}
gpu_name2alias = {v.gpu_name: k for k,v in gpu2info.items()} | dict(rtx_a6000="a6000")

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
    "A1": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A1"]),
    "A2": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A2"]),
    "A3": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A3"]),
    "A4": dict(num_cpus=16, num_gpus=2, hyperthread=False, ssh_names=["A4"]),
    "A5": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A5"]),
    "A6": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A6"]),
    "A7": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A7"]),
    "A8": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A8"]),
    "A9": dict(num_cpus=12, num_gpus=2, hyperthread=False, ssh_names=["A9"]),
    "A99": dict(num_cpus=8, num_gpus=1, hyperthread=False, ssh_names=["A99", "emily"]),
} | dict( # Made up info for the ComputeCanada and Solar clusters so we can run commands on them. This should probably be refactored!
    solar=dict(num_cpus=1, num_gpus=0, hyperthread=False, ssh_names=["solar"]),
    narval=dict(num_cpus=40, num_gpus=4, hyperthread=False, ssh_names=["narval"]),
    cedar=dict(num_cpus=48, num_gpus=4, hyperthread=False, ssh_names=["cedar"]),
    killarney=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["killarney", "killa"]),
    vulcan=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["vulcan"]),
    trillium=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["trillium"]),
    fir=dict(num_cpus=48, num_gpus=4, hyperthread=False, ssh_names=["fir"]),
    rorqual=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["rorqual"]),
    nibi=dict(num_cpus=114, num_gpus=8, hyperthread=False, ssh_names=["nibi"]),
    tamia=dict(num_cpus=64, num_gpus=4, hyperthread=False, ssh_names=["tamia"]),)

# All ComputeCanada hosts. Filtering these out of [machine2info] is useful gives just
# workstation-y machines and Solar.
machines_cc = ["narval", "cedar", "killarney", "vulcan", "trillium", "fir", "rorqual", "nibi", "tamia"]


def get_solar_node2config():
    """Returns a dictionary describing Solar nodes as in [cluster2node2config]. This
    is useful since they tend to change a bit, so we don't want to hardcode them as
    with ComputeCanada stuff.
    """
    node_datas = json.loads(subprocess.getoutput("scontrol show nodes --json"))["nodes"]
    def parse_node_data(nd):
        # print(nd)       
        # We will find something like {"gres": "gpu:l40s:4(S:0-1)"}. So, remove the
        # initial 'gpu:' string and the first parenthetical and everything after
        if "gres" in nd:
            gpu_str = UtilsBase.strip_left(nd["gres"], "gpu:")
            gpu_str = gpu_str.split("(")[0]
            gpu_name, gpus_per_node = gpu_str.split(":")
            gpus_per_node = int(gpus_per_node)
        else:
            gpu_name, gpu = "unknown", 0
        gpu_alias = gpu_name2alias.get(gpu_name, "unknown")
        
        cpus_per_gpu, mem_per_gpu = 1, 1
        if "tres"in nd:
            tres_resources = nd["tres"].split(",")
            for tr_key,tr_val in [tr.split("=") for tr in tres_resources]:
                if tr_key == "cpu":
                    cpus_per_gpu = int(tr_val) // max(1, gpus_per_node)
                elif tr_key == "mem":
                    total_mem = UtilsBase.unit_conversion(tr_val, target="GiB")
                    mem_per_gpu = total_mem // max(1, gpus_per_node)
                else:
                    pass

        can_allocate = any([p in cluster2accounts["solar"] for p in nd["partitions"]])
        can_allocate = can_allocate and (gpu2info[gpu_alias].good if gpu_alias in gpu2info else False)
        nodename = nd["name"]
        return nodename, dict(
            can_allocate=can_allocate,
            gpu_name=gpu_name, gpu_alias=gpu_alias, gpus_per_node=gpus_per_node,
            cpus_per_gpu=cpus_per_gpu, mem_per_gpu=mem_per_gpu,
            gpu_frac=1.,
        )
    
    nodes_configs = [parse_node_data(nd) for nd in node_datas]
    node2config = dict(nodes_configs)

    return node2config

# Data from https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling/en
# for Alliance clusters. For these the amounts below record one bundle of compute.
# IMPORTANT: these values reflect the 'bundle-per-gpu' values, which aren't
# necessarily what job's should request per node. Probably, rounding down to the
# nearest integer for CPU cores and integer or nice-round value for memory is smart.
# The 'default' entry is basically the fallback for where a GPU type can't be found

# FOR RGU COMPUTATIONs
# cpus_per_gpu - default number of CPUs for GPU
# mem_per_gpu - default amount of memory for GPU, in GiB (base 2)

# FOR COMPUTING FREE RESOURCE AMOUNTS.
# cpus_per_gpu_target - target number of CPUs to allocate per GPU (ie. sometimes we want more than one compute bundle's worth)
# mem_per_gpu_target - target amount of memory to allocate per GPU (ie. sometimes we want more than one compute bundle's worth)

# gpu_alias - type of GPU (what we call it)
# gpu_name - type of GPU (what the scheduler calls it)
# gpus_per_node - number of GPUs per node
# can_allocate - whether the node can be allocated

# IMPORANT: whether a cluster's documentation uses base-10 or base-2 units for memory
# isn't consistent. In my understanding, nodes list their total memory in base-2 'M'
# units CfgTRES in 'scontrol show node' output. So, what's below reflects me checking.
cluster2node2config = dict(
    nibi=dict(
        default=dict(cpus_per_gpu=14, mem_per_gpu=250, gpu_alias="h100", gpu_name="h100", gpus_per_node=8, can_allocate=True),
        h100=dict(cpus_per_gpu=14, mem_per_gpu=250, gpu_alias="h100", gpu_name="h100", gpus_per_node=8, can_allocate=True),
        h111=dict(cpus_per_gpu=2, mem_per_gpu=35.7, gpu_alias="h111", gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpus_per_node=16, can_allocate=True),
        h122=dict(cpus_per_gpu=4, mem_per_gpu=71.4, cpus_per_gpu_target=8, mem_per_gpu_target=124, gpu_alias="h122", gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpus_per_node=8, can_allocate=True),
        h143=dict(cpus_per_gpu=7, mem_per_gpu=125, cpus_per_gpu_target=8, mem_per_gpu_target=124, gpu_alias="h143", gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpus_per_node=8, can_allocate=True),
        mi300a=dict(cpus_per_gpu=16, mem_per_gpu=128, gpu_alias="mi300a", gpu_name="mi300a", gpus_per_node=4, can_allocate=False),
        a100=dict(cpus_per_gpu=32, mem_per_gpu=250, gpu_alias="a100", gpu_name="a100", gpus_per_node=8, can_allocate=False),
        a5000=dict(cpus_per_gpu=16, mem_per_gpu=30.5, gpu_alias="a5000", gpu_name="a5000", gpus_per_node=4, can_allocate=False),
        t4=dict(cpus_per_gpu=11, mem_per_gpu=45, gpu_alias="t4", gpu_name="t4", gpus_per_node=4, can_allocate=False)
    ),
    fir=dict(
        default=dict(cpus_per_gpu=12, mem_per_gpu=288, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True),
        h100=dict(cpus_per_gpu=12, mem_per_gpu=288, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True),
        h111=dict(cpus_per_gpu=1.7, mem_per_gpu=41, gpu_alias="h111", gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpus_per_node=8, can_allocate=True),
        h122=dict(cpus_per_gpu=3.4, mem_per_gpu=82, cpus_per_gpu_target=6, mem_per_gpu_target=124, gpu_alias="h122", gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpus_per_node=4, can_allocate=True),
        h143=dict(cpus_per_gpu=6, mem_per_gpu=144, cpus_per_gpu_target=8, mem_per_gpu_target=124, gpu_alias="h143", gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpus_per_node=4, can_allocate=True),
    ),
    rorqual=dict(
        default=dict(cpus_per_gpu=16, mem_per_gpu=124.5, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True,),
        h100=dict(cpus_per_gpu=16, mem_per_gpu=124.5, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True,),
        h111=dict(cpus_per_gpu=2.3, mem_per_gpu=17.7, gpu_alias="h111", gpu_name="nvidia_h100_80gb_hbm3_1g.10gb", gpus_per_node=8, can_allocate=True,),
        h122=dict(cpus_per_gpu=4.5, mem_per_gpu=35.4, cpus_per_gpu_target=6, mem_per_gpu_target=64, gpu_alias="h122", gpu_name="nvidia_h100_80gb_hbm3_2g.20gb", gpus_per_node=4, can_allocate=True,),
        h143=dict(cpus_per_gpu=8, mem_per_gpu=62.2, cpus_per_gpu_target=8, mem_per_gpu_target=64, gpu_alias="h143", gpu_name="nvidia_h100_80gb_hbm3_3g.40gb", gpus_per_node=4, can_allocate=True,),
    ),
    narval=dict(
        default=dict(cpus_per_gpu=12, mem_per_gpu=124.5, gpu_alias="a100", gpu_name="a100", gpus_per_node=4, can_allocate=True,),
        a100=dict(cpus_per_gpu=12, mem_per_gpu=124.5, gpu_alias="a100", gpu_name="a100", gpus_per_node=4, can_allocate=True,),
        a101=dict(cpus_per_gpu=1.7, mem_per_gpu=17.7, gpu_alias="a101", gpu_name="a100_1g.5gb", gpus_per_node=1, can_allocate=True,),
        a112=dict(cpus_per_gpu=3.4, mem_per_gpu=35.4, gpu_alias="a112", gpu_name="a100_2g.10gb", gpus_per_node=1, can_allocate=True,),
        a123=dict(cpus_per_gpu=6.0, mem_per_gpu=62.2, gpu_alias="a123", gpu_name="a100_3g.20gb", gpus_per_node=1, can_allocate=True,),
        a124=dict(cpus_per_gpu=6.9, mem_per_gpu=71.5, cpus_per_gpu_target=8, mem_per_gpu_target=64, gpu_alias="a124", gpu_name="a100_4g.20gb", gpus_per_node=1, can_allocate=True,),
    ),
    trillium=dict(
        default=dict(cpus_per_gpu=24, mem_per_gpu=188, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True,),
        h100=dict(cpus_per_gpu=24, mem_per_gpu=188, gpu_alias="h100", gpu_name="h100", gpus_per_node=4, can_allocate=True,),
    ),
    tamia=dict( # TODO: confirm mem_per_gpu values
        default=dict(cpus_per_gpu=12, mem_per_gpu=125, gpu_alias="h100", gpus_per_node=4, gpu_name="h100", can_allocate=True),
        h100=dict(cpus_per_gpu=12, mem_per_gpu=125, gpu_alias="h100", gpus_per_node=4, gpu_name="h100", can_allocate=True),
        h200=dict(cpus_per_gpu=8, mem_per_gpu=125, gpu_alias="h200", gpus_per_node=8, gpu_name="h200", can_allocate=True),
    ),
    vulcan=dict( # NOTE: these mem_per_gpu are computed from looking at nodes.
        default=dict(cpus_per_gpu=16, mem_per_gpu=125.8, gpu_alias="l40s", gpus_per_node=4, gpu_name="l40s", can_allocate=True),
        l40s=dict(cpus_per_gpu=16, mem_per_gpu=125.8, gpu_alias="l40s", gpus_per_node=4, gpu_name="l40s", can_allocate=True),
        # Not really sure what this is, but it seems like a not-very-good GPU we're unlikely to allocate, so it can be ignored.
        l40s_shard=dict(cpus_per_gpu=1, mem_per_gpu=1, gpu_alias="l40s_shard", gpus_per_node=1, gpu_name="l40s_shard", can_allocate=False),
    ),
    killarney=dict( # NOTE: these mem_per_gpu are computed from looking at nodes.
        default=dict(cpus_per_gpu=16, mem_per_gpu=125.7, gpu_alias="l40s", gpus_per_node=4, gpu_name="l40s", can_allocate=True),
        l40s=dict(cpus_per_gpu=16, mem_per_gpu=125.7, gpu_alias="l40s", gpus_per_node=4, gpu_name="l40s", can_allocate=True),
        h100=dict(cpus_per_gpu=6, mem_per_gpu=251.46, gpu_alias="h100", gpus_per_node=8, gpu_name="h100", can_allocate=True),
    ),
    cedar=dict( # NOTE: these mem_per_gpu are old
        default=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, gpu_name="v100l", can_allocate=True),
        v100l=dict(cpus_per_gpu=8, mem_per_gpu=46, gpu_alias="v100l", gpus_per_node=4, gpu_name="v100l", can_allocate=True),
        p100l=dict(cpus_per_gpu=6, mem_per_gpu=56, gpu_alias="p100l", gpus_per_node=4,  gpu_name="p100l", can_allocate=True),
    ),
    solar=get_solar_node2config() if Utils.is_solar() else dict(),
    # These values don't have to be right
    cs_apex=dict(default=dict(cpus_per_gpu=8, mem_per_gpu=48, gpu_alias="3090", gpus_per_node=2, can_allocate=True),)
)
cluster2node2config = {k: {nk: dict(cpus_per_gpu_target=nv["cpus_per_gpu"], mem_per_gpu_target=nv["mem_per_gpu"]) | nv for nk,nv in v.items()} for k,v in cluster2node2config.items()}
cluster2node2config = {k: {nk: argparse.Namespace(**nv) for nk,nv in v.items()} for k,v in cluster2node2config.items()}


######################################################################################
# Information for figuring out how many RGUs jobs will consume
######################################################################################
# If a job allocates C cores and M GiB of memory and G GPUs of some GPU type, then the
# RGU consumption is max(C/cores_per_rgu, M/mem_per_rgu, G*rgus_per_gpu). For clusters
# not listed here, where GPUs aren't partitioned into multi-instance GPUs, set
# cores_per_rgu=(cores_per_gpu / rgus_per_gpu) and
# mem_per_rgu=(mem_per_gpu / rgus_per_gpu) for the default GPU type.
@lru_cache(maxsize=1)
def cluster_to_resources_per_rgu(c=Utils.get_cluster_type()):
    rgus_per_default_gpu = gpu2info[cluster2node2config[c]["default"].gpu_alias].rgus_per_gpu
    return argparse.Namespace(
        cpus_per_rgu=cluster2node2config[c]["default"].cpus_per_gpu / rgus_per_default_gpu,
        mem_per_rgu=cluster2node2config[c]["default"].mem_per_gpu / rgus_per_default_gpu
    )
cluster2resources_per_rgu = defaultdict(cluster_to_resources_per_rgu)
    # dict(
    #     nibi=dict(cpus_per_rgu=1.15, mem_per_rgu=20.5),
    #     fir=dict(cpus_per_rgu=0.98, mem_per_rgu=23.6),
    #     rorqual=dict(cpus_per_rgu=1.31, mem_per_rgu=10.2),
    #     narval=dict(cpus_per_rgu=3.00, mem_per_rgu=31.5),
    #     trillium=dict(cpus_per_rgu=1.97, mem_per_rgu=15.4),
    # )

######################################################################################
######################################################################################
######################################################################################

######################################################################################
# Other important information on different clusters
######################################################################################
# node_prefixes - prefixes of GOOD nodes. To find these, you can do
#                   sinfo -h -o "%N" | paste -sd, - | xargs scontrol show hostlist
#                   and look at the result.
# wandb_default_mode - default mode for WandB on compute nodes. Note that there's a
#                       difference between nominally offline nodes for which there are
#                       ways of making them be online, and nodes that I haven't
#                       figured out how to make able to online-connect to WandB. If
#                       curious and you're at SFU, reach out.
# default_account - default account (or, for Solar, partition) for jobs on the cluster
cluster2misc_info = dict(
    nibi=dict(node_prefixes=["g"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["nibi"][0], "_gpu")),
    fir=dict(node_prefixes=["fc"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["fir"][0], "_gpu")),
    rorqual=dict(node_prefixes=["rg"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["rorqual"][0], "_gpu")),
    narval=dict(node_prefixes=["ng"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["narval"][0], "_gpu")),
    trillium=dict(node_prefixes=["trig"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["trillium"][0], "_gpu")),
    vulcan=dict(node_prefixes=["rack"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["vulcan"][0], "_gpu")),
    killarney=dict(node_prefixes=["kn"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["killarney"][0], "_gpu")),
    tamia=dict(node_prefixes=["tg"], wandb_default_mode="offline", default_account=UtilsBase.strip_right(cluster2accounts["tamia"][0], "_gpu")),
    solar=dict(node_prefixes=["cs-venus", "cs-bd"], wandb_default_mode="online", default_account=UtilsBase.strip_right(cluster2accounts["solar"][0], "_gpu")),
)
######################################################################################
######################################################################################
######################################################################################

def get_updated_machine_info(m, verbose=0):
    """Returns a Namespace giving the nvidia-smi output, number of GPUs, and number
    CPU cores on machine [m].

    Args:
    m           -- machine name, which must be a key in [machine2info], or a hostname
    """
    try:
        result = SSHCommunication.run_command_on_machine(machine=m, command="nvidia-smi ; nvidia-smi --query-gpu=name --format=csv,noheader | wc -l ; nproc")
    except SSHCommunication.HostInfoError as e:
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
    m = SSHCommunication.hostname_to_machine(m)
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
        result = SSHCommunication.run_command_on_machine(machine=m, command="nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader")

        if len(result) == 0:
            user2gpu_ids = dict()
        else:
            for line in result.split("\n"):
                pid = line.split()[1].replace(",", "")
                user = SSHCommunication.run_command_on_machine(machine=m, command=f"ps -o user= -p {pid}").strip()
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

def str_to_gpu_type(s):
    """Given a string [s], try and figure out what kind of GPUs it means."""
    node_config = cluster2node2config[Utils.get_cluster_type()]
    if len(node_config) == 1 and list(node_config.keys())[0] in gpu2info:
        result = list(node_config.keys())
    elif len(node_config) == 2 and "default" in node_config:
        result = [n for n in node_config.keys() if not n == "default"][0]
    else:
        matched_gpu_name2alias = {gpu_name: gpu_alias for gpu_alias,gpu_name in gpu_alias2name.items() if gpu_name in s}
        if len(matched_gpu_name2alias) == 0:
            result = None
        else:
            matched_gpu_name_alias = sorted(matched_gpu_name2alias.items(), key=lambda x: len(x[0]))
            result = matched_gpu_name_alias[-1][1]
    return None if (result is None or not result in gpu2info) else result

if __name__ == "__main__":
    node2config_gt = cluster2node2config["solar"]
    node2config = get_solar_node2config()

    missing = {k: v for k,v in node2config_gt.items() if not k in node2config}
    print(missing.keys())

    node2diffs = dict()
    for n,config in node2config.items():
        if not n in node2config_gt:
            continue

        config_gt = node2config_gt[n]
        key2diff = {k: (v1, config_gt[k], (v1 == config_gt[k])) for k,v1 in config.items()}
        node2diffs[n] = {k: v for k,v in key2diff.items() if not v[2]}
    
    for node,diff in node2diffs.items():
        print(node)
        print(diff)
        



