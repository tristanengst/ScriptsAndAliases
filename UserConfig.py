"""Per user configuration. Edit this file as needed. The default values are for my use
case.
"""
import argparse
import os.path as osp

# EDIT THESE VALUES TO MATCH YOUR USE CASE
# If you're in APEX lab, you don't need to change these
# If multiple accounts are available, list the default one first
cluster2accounts = dict(
    nibi=["rrg-keli_gpu", "def-keli_gpu"],
    trillium=["def-keli"],
    fir=["def-keli_gpu"],
    rorqual=["def-keli_gpu"],
    narval=["def-keli_gpu"],
    cedar=["def-keli_gpu"],
    beluga=["def-keli_gpu"],
    vulcan=["aip-keli"],
    killarney=["aip-keli"],
    tamia=["aip-keli"],
    solar=["cs-gpu-research"],
)

# Paths that will work from any login node on the cluster
# They can be absolute or start with ~ for your home directory
# They don't necessarily need to exist
checkpoints_search_dirs = ["~/scratch/IMLE-SSL/models_imle",
    "~/scratch/IMLE-SSL/models_mae",
    "~/scratch/IMLE-SSL/finetunes",
    "~/Development/IMLE-SSL-2/probes",]
job_result_search_dirs = [
    "~/Development/IMLE-SSL-2/pretrain_results",
    "~/Development/IMLE-SSL-2/finetune_results",
    "~/Development/IMLE-SSL-2/probe_results",
    "~/Development/IMLE-SSL-Dev/pretrain_results",
    "~/Development/IMLE-SSL-Dev/finetune_results",
    "~/Development/IMLE-SSL-Dev/probe_results",
]
slurm_script_search_dirs = [
    "~/Development/IMLE-SSL-2/slurm",
    "~/Development/IMLE-SSL-Dev/slurm",
]

checkpoints_search_dirs = [osp.expanduser(p) for p in checkpoints_search_dirs]
job_result_search_dirs = [osp.expanduser(p) for p in job_result_search_dirs]
slurm_script_search_dirs = [osp.expanduser(p) for p in slurm_script_search_dirs]