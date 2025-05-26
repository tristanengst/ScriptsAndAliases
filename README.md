# 📜 Scripts and Aliases
Useful scripts and their aliases, and more aliases useful for manipulating SLURM and other ML jobs.

### Installation
The aliases expect Python scripts to live in the `~/.ScriptsAndAliases` directory:
```
git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases
python ~/.ScriptsAndAliases/WriteAliases.py
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
sqb
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

Tar files modified within the last `--last_k_days` for saving. _Unlike many not-obviously-wrong ways of saving experiments on ComputeCanada, this runs in some number of ours even when there are hundreds of GB to tar:_
```
python TarFiles.py --dir directory_to_tar --out name_of_tar_file --last_k_days 60 --ignore_no_pt 0
```
If `directory_to_tar/some_file_or_folder` exists, you can extract it with `tar -xf name_of_tar_file -C directory_to_extract_under some_file_or_folder







