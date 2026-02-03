# 📜 Scripts and Aliases
Useful scripts and aliases for manipulating SLURM and other ML jobs. Primarily, they support a number of core functionalities:
1. Reduced friction in keeping many SLURM experiments running
2. Real-time knowledge of which GPUs of which partitions or which ComputeCanada clusters are better and worse to submit to
3. Many miscellaneous utilities. Literally software candy!

### Installation
This code is explicitly designed to work with Python after 3.11, and without additional dependencies. The aliases expect Python scripts to live in the `~/.ScriptsAndAliases` directory:
```
git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases
python ~/.ScriptsAndAliases/WriteAliases.py
source ~/.bashrc
```

To update:
```
cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py ; source ~/.bashrc
```

### Who this is for, updates, and configuration
While these utilities are primarily for myself and other members of APEX lab, anyone using ComputeCanada could also get a fair amount of use from this, and I expect the algorithms and ideas are more broadly useful. This primarily manifests through hardcoding things that could in principle be variable, like SLURM account names, cluster names, and types of compute nodes.

This repo is updated frequently, and is provided as-is. Expect things and especially the core useful functionality to work well, but there are likely corner cases I don't know about or aren't yet worth handling.

Please submit pull requests if you want something handled.

**Configuration**. APEX lab users can get the basic functionality without any configuration. Other users will need to modify `cluster2account` in `UserConfig.py` by adding the elevant `def-your-PI-name`, `rrg-your-PI-name`, and `aip-your-PI-name` accounts for each cluster, and maybe should modify dictionaries in `MachineInfo.py`. For all users, the advanced functionality is unlocked by modifying `...search_dirs` lists in `UserConfig.py`.

### Basic Usage
These commands won't require you to change how you do anything.

Display cluster state:
```
sqb [-p show partitions] [-n show nodes] [-s show start times] [-u show all users in account(s)] .......
```

Display cluster state without needing working Python, but less info:
```
sqbf
```

Show just LevelFS:
```
sshareb
```

Extract many job IDs:
```
exj "copy-and-paste-sqb-output"
```

Show job info:
```
scb JOBID # Or UID, if advanced usage works
```

Update job info (works on one or many jobs):
```
scu Key=Value JOBID1 ... JOBIDN # Or UIDs, if advanced usage works
makedef JOBID1 ... JOBIDN # Or UIDs, if advanced usage works
makerrg JOBID1 ... JOBIDN # Or UIDs, if advanced usage works
```

Show node info:
```
scn NODE_HOSTNAME # scn without a node hostname shows all nodes
```

### Advanced Usage
These commands require you to slightly change how you do things in that you need to assign experiments UIDs and adopt a one-unique-SLURM-script-per-model-run model. This allows every unique instance of training a neural net to be associated to **(1)**  all the files that configure the training (eg. SLURM `sbatch` scripts, config files), **(2)** all the files/data generated (SLURM job output, checkpoints, logged results), **(3)** the SLURM job(s) that perform the training. UIDs giving this property will dramatically reduce the friction in research.

Concretely, you need to:
1. Generate UIDs—I use `wandb.util.generate_id()`.
2. Have your code either read a UID from the command line, or generate one automatically if it's not provided.
3. When you generate a SLURM job script, generate a UID. Include the UID towards the end of the experiment's name. You will then **(a)** give this name to the job, **(2)** name the SLURM script as `/path/to/slurm_scripts/experiment_name_with_uid.sh`, **(3)** have the job write outputs to `/path/to/job_outputs/experiment_name_with_uid.txt`, **(4)** ensure the job will create and write checkpoints under the directory `/path/to/checkpoints/experiment_name_with_uid/`, **(5)** pass the UID to the code actually running the experiment in the SLURM script, **(6)** comment the SLURM script as follows:
   ```
   #SBATCH --comment="{'uid': 'UID', 'exp_name': 'experiment_name_with_uid'}"
   # The total length of the comment is limited to 256 characters. Abbreviate the experiment name as needed with an eye towards making it uniquely identify checkpoints to ensure the comment is valid JSON
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
# Essentially iterates over all the uniquely-identified checkpoint folders: rsync -rh --info=progress2 source_cluster:~/path/to/checkpoint ~/path/to
# Note: you'll need to have `source_cluster` in your ~/.ssh/config` file for this to work. See below for details.
```

Send experiment checkpoints from the current machine to cluster `destination_cluster`:
```
rsyncb list of UID or substring of experiment name containing enough of the UID to uniquely identify the experiments destination_cluster
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

Cancel SLURM job(s) by JOBID _or_ UID:
```
scancelb list of JOBID or UID
```

You can also leverage this inside the code you write. Some key use cases are thus:
1. Letting jobs modify the SLURM script that submitted them. For example, imagine a job discovers that the node its on has a bad GPU. It then **(1)** modifies its SLURM script to exclude the bad node by appending it to the list of nodes excluded for the job—`#SBATCH exclude=possibly,empty,list,of,bad,nodes`, **(2)** resubmits the SLURM script, **(3)** ends.
2. Being vastly more stateless with respect to the codebase—so you don't have to spend nearly as much time worrying about what code actually generated a particular result. Your SLURM scripts should **(1)** check to see if a `/path/to/experiment_name_with_uid/code.tar` exists, and create this file from the code you want to run if it doesn't (or, create the file when the SLURM script is submitted!). Then, **(2)** untar this onto a job-specific directory on the compute node—usually `$SLURM_TMPDIR`. Now, **(3)** run the code from this directory. Once the tarfile is created, you can modify your code arbitrarily without impacting jobs that use this tarfile.



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



### SSH Config
Some functionality for this library assumes SSH is set up in specific good ways. For ComputeCanada clusters, you'll need the name for them used in this codebase as a key for them in your `~/.ssh/config` file. It will also be helpful to configure the connections thus:
```
Host *
  User YOUR_USERNAME_ON_COMPUTECANADA
  ServerAliveInterval 60
  StrictHostKeyChecking no
  ControlPath ~/.ssh/cm-%r@%h:%p
  ControlPersist yes
  ControlMaster auto

Host narval
  HostName narval.computecanada.ca
Host trillium
  HostName trillium-gpu.alliancecan.ca
Host rorqual
  HostName rorqual.alliancecan.ca
Host fir
  HostName fir.alliancecan.ca
Host nibi
  HostName nibi.sharcnet.ca
Host vulcan
  HostName vulcan.alliancecan.ca
Host killarney killa
  HostName killarney.alliancecan.ca
Host tamia
  HostName tamia.alliancecan.ca
```

Some functionality for APEX workstations and servers is predicated on a certain naming convention. We don't want to make hostnames public, so they're actually read from your `~/.ssh/config` file, rather than harcoded here. See the lab's Notion.






