# Scripts and Aliases
Useful Python scripts called from bash aliases, all for manipulating SLURM.... and more!
1. 🚀 Reduce friction in keeping many SLURM experiments running
2. ℹ️ Provide real-time knowledge of which GPUs of which partitions of which ComputeCanada clusters are better and worse to submit to
3. 🍭 Make for many miscellaneous useful utilities

A bunch of this functionality is given by the [**basic usage**](#basic-usage), while additional, cooler functionality comes from the [**advanced usage**](#advanced-usage), which requires a little more configuration and you to submit SLURM jobs in smart ways.

**Intended users.** These utilities are primarily for myself, other members of [APEX lab](https://sfuapex.ca/), and secondarily others at [Simon Fraser University](https://www.sfu.ca/fas/computing.html). However, not only should anyone using ComputeCanada be able to get a fair amount of use from them, but also I expect the algorithms and ideas are more broadly useful.

### Installation
This code is explicitly designed to work with `Python>=3.11`, and without additional dependencies. The aliases expect Python scripts to live in the `~/.ScriptsAndAliases` directory:
```
git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases
python ~/.ScriptsAndAliases/WriteAliases.py # Maintains a chunk of ~/.bashrc containing aliases that call Python scripts
source ~/.bashrc
```

**Configuration**. APEX lab users get the [**basic functionality**](#basic-usage) without any configuration. Other users will need to modify `cluster2account` in `UserConfig.py` by adding the elevant `def-your-PI-name`, `rrg-your-PI-name`, and `aip-your-PI-name` accounts for each cluster, and maybe should modify dictionaries in `MachineInfo.py`. _For all users, the [**advanced functionality**](#advanced-usage) is unlocked by modifying `...search_dirs` lists in `UserConfig.py`, and taking actions described below._

### Updates and Future Development
_This repo is provided as-is._ It uses a **move-fast-and-fix-things** development model, since often it needs to adapt to unforeseen needs or unannounced changes to clusters. Moreover, my job is research and not research tooling, so while the core useful functionality should work, there are likely corner case issues I'm unaware of or haven't had time to deal with.

Please submit a issues or pull requests as desired. To update:
```
cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py ; source ~/.bashrc
```

### Basic Usage
These commands won't require you to change how you do anything.

Display the state of your/your group's jobs on a cluster. Especially with the [advanced usage](#advanced-usage) enabled, this becomes a great dashboard:
```
sqb [-p show partitions] [-n show nodes] [-s show start times] [-u show all users in your account(s)] .......
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
# Example: scu TimeLimit=8:00:00 123456 654321
```

Swap (lists of) jobs between `def-` and `rrg-` partitions:
```
makedef JOBID1 ... JOBIDN # Or UIDs, if advanced usage works
makerrg JOBID1 ... JOBIDN # Or UIDs, if advanced usage works
```

Show node info:
```
scn NODE_HOSTNAME
# scn without a node hostname shows all nodes.
# Sometimes this reveals information not in the ComputeCanada wiki!
```

### Advanced Usage
These commands require you to slightly change how you do things in that you need to assign experiments UIDs and adopt a one-unique-SLURM-script-per-run research paradigm. This allows every unique instance of training a neural net to be associated to **(1)**  all the files that configure the training (eg. SLURM `sbatch` scripts, config files), **(2)** all the files/data generated (SLURM job output, checkpoints, logged results), **(3)** the SLURM job(s) that perform the training. _UIDs giving this property will dramatically reduce the friction in research!_

#### Setup
Concretely, you need to do the following. It will be much easier if you generate SLURM scripts (very easy, few mistakes) from templates rather than writing them manually (typos kill jobs).
1. Decide on (possibly several of each) paths for SLURM scripts, folders of checkpoints, and job outputs respectively. `/path/to/slurm_scripts`, `/path/to/checkpoints`, and `/path/to/job_outputs`. These should be canonical across clusters and workstations, and be either from your home directory or absolute—prefixed by `~/` or `/`. _**These can be symlinks!**_ The goal is to end up with a filesystem that might be structured something like this:
   ```
   ├── /path/to/slurm_scripts
   │   ├──experiment_with_uidABCD1234.sh
   │   └──experiment_with_uidEFGH5678.sh
   ├── /path/to/checkpoints
   │   ├──experiment_with_uidABCD1234
   │   │   ├── checkpoint_86.pt
   │   │   └── checkpoint_99_latest.pt
   │   ├── experiment_with_uidEFGH5678
   │   │   └──checkpoint_0.pt
   ├── /path/to/job_outputs
   │   ├── experiment_with_uidABCD1234.txt
   │   └── experiment_with_uidEFGH5678.txt
   ```
2. Have a way of generating  UIDs of about 8 characters, eg. `wandb.util.generate_id()` or steal `generate_uid()` from `YourCode.py`
3. Have your code for any run either read and assign a UID from the command line, or generate and assign one automatically (eg. as above) if it's not provided.
4. Modify `UserConfig.py` by adding `/path/to/slurm_scripts`, `/path/to/job_outputs`, and `/path/to/checkpoints` as needed
5. When you generate a SLURM job script for an experiment, it should have a UID contained within its name that can uniquely identify everything associated to the experiment. You shoud:
   1. Include the UID towards the end of the experiment's name, eg. `experiment_name_with_uid`
   2. Name the SLURM script file as `/path/to/slurm_scripts/experiment_name_with_uid.sh`
   3. Give this name to the job running the SLURM script:
      ```
      #SBATCH --job-name=experiment_name_with_uid
      ```
   4. Make the job **append** outputs to `/path/to/job_outputs/experiment_name_with_uid.txt`:
      ```
      #SBATCH --output=/path/to/job_outputs/experiment_name_with_uid.txt
      #SBATCH --open-mode=append
      ```
   5. Comment the SLURM script as follows (see `get_sbatch_comment()` in `YourCode.py` for implementation):
      ```
      #SBATCH --comment="{'uid': 'UID', 'exp_name': 'experiment_name_with_uid'}"
      # The total length of the comment is limited to 256 characters, and my code expects valid JSON. Abbreviate the experiment name—not the UID—as needed
      ```
   6. Pass the UID to the code actually running the experiment in the SLURM script, allowing it to
   7. Ensure the job will create and write checkpoints under the directory `/path/to/checkpoints/experiment_name_with_uid/`

<details><summary><b>Extra:</b> Make <code>sqb</code> colorize by experiment heartbeat</summary>Have your code occassionally write a <code>heartbeat.txt</code> file under <code>/path/to/checkpoints/experiment_name_with_uid/</code>. Its sole content should be the current time in <code>YYYY--MM-DD HH:MM:SS</code> format. The first half of the entry in the `STATE` column will be colorized from green to red depending on the extent to which this timestamp is old. See <code>write_heartbeat()</code> in <code>YourCode.py</code> for implementation.</details>

<details><summary><b>Extra:</b> Make <code>sqb</code> display latest checkpoints</summary>Modify the <code>checkpoint_extensions</code> and <code>checkpoint_prefixes</code> lists in <code>UserConfig.py</code></details>

#### Advanced Usage Commands and Functionality
All this not only enables the following super-useful commands, but also expands the functionality of many basic usage commands. For instance,
- `sqb` gets much more useful! It will:
   - Allow spotting issues super easily by colorizing the `STATE` column based on **(1)** experiment heartbeat if possible (first half) and **(2)** the last write to its output file (second half). Green is more recent.
   - Display UIDs
   - Display the latest saved checkpoint
- In most places where you can provide a job ID, a UID will also work

Find and print experiment output:
```
jcat UID or substring of experiment name containing enough of the UID to uniquely identify the experiment
# think: 'job cat'
```

Find and print the SLURM script for an experiment:
```
jcats UID or substring of experiment name containing enough of the UID to uniquely identify the experiment
# think: 'job cat script'
```

Run `ls` on the directory an experiment would've saved checkpoints in:
```
lse [list of UIDs or identifying substrings] [-l as in normal ls] [-t as in normal ls] ...
# Accepts many (or all?) flags to the ls command too, AFTER the UIDs
```

Extract all the UIDs of jobs from some lines of `sqb` output (like `exj`):
```
exu "copy-and-paste-sqb-output"
```

Send folders of experiment checkpoints from cluster `source_cluster` to the current machine:
```
rsyncb source_cluster [list of UID or substring of experiment name containing enough of the UID to uniquely identify the experiments]
# Example: rsyncb nibi UIDA UIDB UIDC
# think: 'rsync better'
# Essentially iterates over all the uniquely-identified checkpoint folders: rsync -rh --info=progress2 source_cluster:~/path/to/checkpoint ~/path/to
# Note: you'll need to have `source_cluster` in your ~/.ssh/config` file for this to work. See below for details.
```

Send folders of experiment checkpoints from the current machine to cluster `destination_cluster`:
```
rsyncb [list of UID or substring of experiment name containing enough of the UID to uniquely identify the experiments] destination_cluster
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
sqb # Completely different functionality on workstations/servers, but it remains probably the first command you'll run when you log in!
```
**Note:** This command requires that your `~/.ssh/config` file has an entry `ssh_name` for some entry in `ssh_names` for each machine in the `machine2info` dictionary in `MachineInfo.py` such that `ssh ssh_name` will SSH onto the given machine without password authentication. _This is for network security—APEX lab workstations and servers' hostnames won't be publicly available in plaintext. If you use an SSH name not in the dictionary that doesn't give away the hostname, submit a pull request!_

Updates this repo on every machine that `find_free_gpus` would query:
```
python UpdateScriptsAndAliases.py
```

### Miscellaneous
Generate a UID (requires WandB to be installed):
```
get_wandb_id
```

PKill WandB when it's slow. _Maybe not a good idea not to use this on servers or where someone else might be using WandB?_:
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


### More Commands
The functionality above is just the key things I use on a day-to-day basis. There is, in fact, more—see all the scripts herein for details!

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
For APEX lab servers and workstations, we don't want to make hostnames public, so they're actually read from your `~/.ssh/config` file. The code then tries to associate the hostnames to servers and workstations it knows about. See the lab's Notion for a sample `~/.ssh/config` file that has names for which this association works.






