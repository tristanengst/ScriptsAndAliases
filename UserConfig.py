"""Per user configuration. Edit this file as needed. The default values are tuned for
my use case.
"""
import argparse
import os.path as osp

######################################################################################
# VALID ACCOUNTS TO CHECK FOR JOBS/LEVELFS ON DIFFERENT CLUSTERS
# If you're in APEX lab, you don't need to change these
# If multiple accounts are available, list the default one first
######################################################################################
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
    solar=["cs-gpu-research"], # This is actually a partition! Scripts referencing this should be aware.
)
######################################################################################
######################################################################################
######################################################################################

######################################################################################
# SEARCH DIRECTORIES FOR CHECKPOINTS, JOB RESULTS, AND SLURM SCRIPTS
# Paths that will work from any login node on the cluster
# They can be absolute or start with ~ for your home directory
# They don't necessarily need to exist
######################################################################################
checkpoints_search_dirs = ["~/scratch/IMLE-SSL/models_imle",
    "~/scratch/IMLE-SSL/models_mae",
    "~/scratch/IMLE-SSL/finetunes",
    "~/Development/IMLE-SSL-2/probes",
    "~/scratch/openPiMLE/exps",
    "~/scratch/LeRobot/exps"]
job_result_search_dirs = [
    "~/Development/IMLE-SSL-2/pretrain_results",
    "~/Development/IMLE-SSL-2/finetune_results",
    "~/Development/IMLE-SSL-2/probe_results",
    "~/Development/IMLE-SSL-Dev/pretrain_results",
    "~/Development/IMLE-SSL-Dev/finetune_results",
    "~/Development/IMLE-SSL-Dev/probe_results",
    "~/Development/openPiMLE-lerobot/results",
    "~/Development/LeRobot/results"
]
job_error_search_dirs = [
    "~/Development/openPiMLE-lerobot/errors"
]
slurm_script_search_dirs = [
    "~/Development/IMLE-SSL-2/slurm",
    "~/Development/IMLE-SSL-Dev/slurm",
    "~/Development/openPiMLE-lerobot/slurm",
    "~/Development/LeRobot/slurm"
]

checkpoints_search_dirs = [osp.expanduser(p) for p in checkpoints_search_dirs]
job_result_search_dirs = [osp.expanduser(p) for p in job_result_search_dirs]
job_error_search_dirs = [osp.expanduser(p) for p in job_error_search_dirs]
slurm_script_search_dirs = [osp.expanduser(p) for p in slurm_script_search_dirs]
######################################################################################
######################################################################################
######################################################################################


######################################################################################
# LATEST CHECKPOINT FINDING FOR SQB OUTPUT
# The sqb command can display the latest checkpoint if this isn't empty. To figure out
# which file is the latest checkpoint, it will only files that satisfy:
# 1. The file ends with an extension in [checkpoint_extensions]
# 2. The file's basename starts with one of the prefixes in [checkpoint_prefixes]. The
#    empty string prefix is special in that (a) it matches only files whose basenames'
#    actual first character is numeric, and (b) it has higher priority than all other
#    prefixes if multiple are found.
# 3. Among all these files, the one with the latest modification time is chosen.
######################################################################################
checkpoint_extensions = [".pt", ".pth"]
checkpoint_prefixes = ["fn", "probe_pretep", ""]
######################################################################################
######################################################################################
######################################################################################

######################################################################################
# COLORIZATION SETTINGS FOR SQB OUTPUT
# Different parts of the sqb output are colorized depending on where a particular
# value sits within a list of thresholds.
# Each list below contains the thresholds for a particular part of the output. The
# color scale is reasonable but hardcoded.
# You may change the values in the lists to be discriminative for your use case, but
# the number of values in each probably shouldn't be changed.
######################################################################################

# 10 values. Colorizes TIME_LEFT column red-to-green depending on how many HOURS there
# is left for a job to run
colorize_time_lefts_cutoffs = [0.25, 0.5, 1, 2, 3, 5, 7, 12, 24, 48] # In HOURS

# 10 values. Colorizes the SUBMIT_TIME column (if present) green-to-red depending on
# how many HOURS ago the job was submitted
colorize_submit_times_cutoffs = [0.25, 0.5, 1, 2, 3, 5, 7, 12, 24, 48]

# 10 values. Colorizes START_TIME column blue-to-red depending on how many HOURS
# (or N/A) in the future the job is expected to start in
colorize_start_times_cutoffs = [0.25, 1, 3, 6, 12, 18, 24, 36, 48, 72]

# 10 values. Colorizes the QUEUE column green-to-red depending on how many HOURS the
# job has been in the queue
colorize_queue_times_cutoffs = [0.25, 0.5, 1, 3, 6, 12, 18, 24, 36, 72]

# 10 values. Colorizes the STATE column green-to-red depending on how many MINUTES ago
# a job has either (a) written a heartbeat file (first half of the output), or (b)
# changed its output file was updated
colorize_states_cutoff_values = [1, 2, 5, 10, 20, 30, 40, 50, 60, 90]
######################################################################################
######################################################################################
######################################################################################

slurm_job_data_dir = osp.join(osp.dirname(osp.abspath(__file__)), "slurm_job_data")
