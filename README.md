# 📜 Scripts and Aliases
Useful scripts and aliases for manipulating SLURM and other ML jobs. Primarily, they support a number of core functionalities:
1. Reduced friction in keeping many SLURM experiments running, 
2. Real-time knowledge of which GPUs of which partitions or which ComputeCanada clusters are better and worse to submit to.
3. Many miscellaneous utilities. Literally software candy.

### Installation
This code is explicitly designed to work with Python>=3.11, and without additional dependencies. The aliases expect Python scripts to live in the `~/.ScriptsAndAliases` directory:
```
git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases
python ~/.ScriptsAndAliases/WriteAliases.py
source ~/.bashrc
```
To update:
```
cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py ; source ~/.bashrc
```

### Basic Usage
These commands won't require you to change how you do anything. 

### Better Usage
These commands require you to slightly change how you do things in that you need to assign experiments UIDs and adopt a one-unique-SLURM-script-per-model-run model. This allows every unique instance of training a neural net to be associated to **(1)**  all the files that configure the training (eg. SLURM `sbatch` scripts, config files), **(2)** all the files/data generated (SLURM job output, checkpoints, logged results), **(3)** the SLURM job(s) that perform the training. UIDs giving this property will dramatically reduce the friction in research.

Concretely, you need to:
1. Generate UIDs---I use `wandb.util.generate_id()`.
2. Have your code either read a UID from the command line, or generate one automatically if it's not provided.
3. When you generate a SLURM job script, generate a UID. Include the UID towards the end of the experiment's name. You will then **(a)** give this name to the job, **(2)** name the SLURM script as `/path/to/slurm_scripts/experiment_name_with_uid.sh`, **(3)** have the job write outputs to `/path/to/job_outputs/experiment_name_with_uid.txt`, **(4)** ensure the job will create and write checkpoints under the directory `/path/to/checkpoints/experiment_name_with_uid/`, **(5)** pass the UID to the code actually running the experiment in the SLURM script, **(6)** comment the SLURM script as follows:
   ```
   #SBATCH --comment="{'uid': 'UID', 'exp_name': 'experiment_name_with_uid'}" 
   ```
4. Modify `UserConfig.py` by adding `/path/to/slurm_scripts`, `/path/to/job_outputs`, and `/path/to/checkpoints` to their respective lists.
5. Ensure that `/path/to/checkpoints` from your home directory is canonical on all the systems you'd ever consider using. Use symlinks.

This enables the following super-useful commands:

Extract the UIDs of jobs from `sqb` output:
```
exu "copy-and-paste lines from sqb"
```

Print experiment output:
```
jcat UID or substring of experiment name containing enough of the UID to uniquely identify the experiment
# short for 'job cat'
```

Print the SLURM script for an experiment:
```
jcats UID or substring of experiment name containing enough of the UID to uniquely identify the experiment
# short for 'job cat script'
```

Send experiment checkpoints from cluster `source_cluster` to the current machine:
```
rsyncb source_cluster list of UID or substring of experiment name containing enough of the UID to uniquely identify the experiments
# short for 'rsync better'
```

Send experiment checkpoints from the current machine to cluster `destination_cluster`:
```
rsyncb list of UID or substring of experiment name containing enough of the UID to uniquely identify the experiments destination_cluster
# Essentially iterates over all the uniquely-identified checkpoint folders: rsync -rh --info=progress2 /path/to/checkpoint destination_cluster:/path/to/
```

Print info on a particular SLURM job:
```
scb JOBID or UID
# alias for scontrol show job JOBID
```

Update SLURM job(s):
```
scu Key=Value list of JOBID or UID
# On each individual JOBID, does scontrol update job JOBID Key=Value
# eg. make jobs for two experiments have an 8H time limit: `scu TimeLimit=8:00:00 abcdef uvwxyz
```

### Useful on our SLURM Clusters
Make jobs `123` and `456` run on the `def-keli` or `rrg-keli` accounts:
```
makedef 123 456
makerrg 123 456
```

View information about a job (syntactic sugar for `scontrol show job 123`):
```
scb 123
```

Update jobs `123` and `456` to have a different configuration. See [https://slurm.schedmd.com/sbatch.html](slurm.schedmd.com/sbatch.html) for what you can change. _**Note:** only some parts of a job's configuration can be updated while it's running or pending; otherwise, you will need to resubmit it with the modification, wasting the time it's spent queuing on ComputeCanada._ 
```
scu KEY=VALUE 123 456
scu TimeLimit=12:00:00 123 456 # Updates jobs 123 and 456 to have a time limit of 12H
```

Exclude nodes `node123` and `node456` from being used by any job currently running or queueing—after it is submitted again. 🤔 _This has to modify job submission scripts, so it will only work correctly only if each job is submitted from a unique one... see below._
```
exclude_nodes node123 node456
```

View all your jobs (nicer version of `squeue`):
```
sqbf
```

View all `def-keli` and `rrg-keli` jobs on ComputeCanada, or all users' jobs on Solar:
```
sqbau
```

View `LevelFS` (nicer version of `sshare`):
```
sshareb
```

Extracts all job IDs from string `s` of newline-separated job names (eg. `sqb` output):
```
extract_job_ids 's'
```

### Useful on SLURM clusters with smart job naming
I always include a matching UID in **(1)** my experiments' names and **(2)** the directories they save checkpoints to, ensuring an unambiguous provenance to any result or file. On SLURM this extends to **(3)** SLURM jobs, **(4)** their submission scripts, and **(5)** their output files. These UIDs become a central handle with which to interact with the cluster; for example, finding the script that ran an experiment with UID `asdfgh` is simple: `cat some/path/*asdfgh*`!

_To effect this, the Python script that runs an experiment has `--uid ` argument, with the UID generated automatically if `--uid` isn't included. Each of these scripts is run inside a SLURM script, generated from a template file by another Python script. This script can then generate a UID to find the name of the experiment it's submitting, and then include it as a keyword argument to the Python script run inside the job. This allows setting the job's name, output file, and the generated SLURM script to include the UID. I also include the UID in a dictionary of metadata stored in the job's `COMMENT` attribute (up to 256 characters). This is probably the most unambiguous way to specify it, as how the UID appears in how things are named doesn't matter._

View all your jobs (nicer version of `sqb`) and show UIDs too:
```
sqb [-a show jobs with duplicate UIDs] [-s show start times] [-u all users as in sqbau]
```
The bash commands don't require using flags for ease of typing, eg `sqba` and `sqbus` and `sqbsu` are valid.

Extract UIDs of all jobs from string `s` of newline-separated job names  (eg. `sqb` output):
```
extract_uids 's'
```

Checks to make sure that no two jobs are running the same experiment:
```
check_duplicate_jobs
```

### Useful on Workstations and Servers


Combines single-node `torchrun` and GPU-CPU assignment. _Imagine you wanted to run on GPUs `6` and `7` and their assigned CPUs:_
```
# CUDA_VISIBLE_DEVICES=6,7 taskset -c 38-49 torchrun --standalone --nnodes=1 --nproc-per-node 2 PythonScript.py ... ...
tpython_ddp2 PythonScript.py ... --gpus 6 7 ...
```
- If `PythonScript.py` does not take `--gpus` argument, also include `--strip_gpus 1`
- Using `python_ddpX` instead is just syntactic sugar for single-node torchrun without the CPU assignment

View available GPUs across all workstations and servers:
```
find_free_gpus
```
**Note:** This command requires that your `~/.ssh/config` file has an entry `ssh_name` for some entry in `ssh_names` for each machine in the `machine2info` dictionary in `MachineInfo.py` such that `ssh ssh_name` will SSH onto the given machine without password authentication. _This is for network security—our machines' hostnames won't be publicly available in plaintext. If you use an SSH name not in the dictionary that doesn't give away the hostname, submit a pull request!_

Updates this repo on every machine that `find_free_gpus` would query:
```
python UpdateScriptsAndAliases.py
```

### Miscellaneous
Generate a UID (requires WandB to be installed):
```
get_wandb_id
```

PKill WandB when it's slow. _Probably a good idea not to use this on servers or where someone else might be using WandB_:
```
killwandb
```

Displays history without line numbers:
```
historyb
```

Tar files modified within the last `--last_k_days` for saving. _Unlike many not-obviously-wrong ways of saving experiments on ComputeCanada, this takes hours and not days:_
```
python TarFiles.py --dir directory_to_tar --out name_of_tar_file --last_k_days 60 --ignore_no_pt 0
```
If `directory_to_tar/some_file_or_folder` exists, you can extract it with `tar -xf name_of_tar_file -C directory_to_extract_under some_file_or_folder`

### Notes
- The development model for this is very ad-hoc; I fix bugs when they are sufficiently annoying to justify the time. This could change if this impacts other people.





