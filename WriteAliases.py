"""Writes useful ML aliases to .zshrc and .bashrc files."""

import argparse
import os.path as osp
import re

start_str = """
# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
. /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

# USEFUL
# "salloc --time=1:0:0 --ntasks=1 --account=rrg-keli --mem=100G --nodes=1 --cpus-per-task=12 --gres=gpu:h100:1"

# LOAD PYTHON
source ~/py311IMLESSL/bin/activate
"""

aliases = [
    "# START USEFUL ML ALIASES",

    # Useful on SLURM, doesn't require a Python script
    "alias sqbf=\"squeue -u $USER -O 'JobArrayID:.10,Name:.175,State:.8,TimeLeft:.10'\"",
    "alias historyb=\"history | cut -c 8-\"",
    "alias sshareb=\"sshare -l -A rrg-keli_gpu; sshare -l -A def-keli_gpu\"",
    "alias scn=\"scontrol show nodes\"",

    ##################################################################################
    # Useful on SLURM, requires a Python script
    ##################################################################################
    "alias makedef=\"python ~/.ScriptsAndAliases/SwitchAccounts.py --account def --job\"",
    "alias makerrg=\"python ~/.ScriptsAndAliases/SwitchAccounts.py --account rrg --job\"",
    "alias scb=\"python ~/.ScriptsAndAliases/Scb.py --job \"",
    "alias scu=\"python ~/.ScriptsAndAliases/Scu.py \"",
    "alias extract_job_ids=\"python ~/.ScriptsAndAliases/ExtractJobIds.py \"",
    "alias extract_jobids=\"python ~/.ScriptsAndAliases/ExtractJobIds.py \"",
    "alias exj=\"python ~/.ScriptsAndAliases/ExtractJobIds.py \"",

    # LS a model without typing an annoying path
    "alias lse=\"python ~/.ScriptsAndAliases/LSExperiment.py -m \"",

    # Common ways to run Sqb2.py.
    "alias sqb=\"python ~/.ScriptsAndAliases/Sqb2.py \"",       
    "alias sqba=\"python ~/.ScriptsAndAliases/Sqb2.py -a \"",
    "alias sqbls=\"python ~/.ScriptsAndAliases/Sqb2.py -ls \"",
    "alias sqbas=\"python ~/.ScriptsAndAliases/Sqb2.py -as \"",
    "alias sqbsa=\"python ~/.ScriptsAndAliases/Sqb2.py -ns \"",
    "alias sqbau=\"python ~/.ScriptsAndAliases/Sqb2.py -au \"",
    "alias sqbasu=\"python ~/.ScriptsAndAliases/Sqb2.py -asu \"",
    "alias sqbaus=\"python ~/.ScriptsAndAliases/Sqb2.py -aus \"",
    "alias sqbl=\"python ~/.ScriptsAndAliases/Sqb2.py -l \"",

    # Quiet output, saves cluster state to a file
    "alias sqbr=\"python ~/.ScriptsAndAliases/Sqb2.py -ausq --record default --verbose 0\"", 

    "alias scancelb=\"python ~/.ScriptsAndAliases/Scancelb.py \"",
    "alias extract_uids=\"python ~/.ScriptsAndAliases/ExtractUIDs.py --jobs \"",
    "alias exu=\"python ~/.ScriptsAndAliases/ExtractUIDs.py --jobs \"",
    "alias exclude_nodes=\"python ~/.ScriptsAndAliases/ModifyNodeExclusion.py -n\"",
    "alias include_nodes=\"python ~/.ScriptsAndAliases/ModifyNodeExclusion.py +n\"",
    "alias modify_node_exclusion=\"python ~/.ScriptsAndAliases/ModifyNodeExclusion.py \"",
    "alias check_duplicate_jobs=\"python ~/.ScriptsAndAliases/CheckDuplicateJobs.py\"",

    # Prints a job's output and/or SLURM script given a substring from its name
    "alias jcat=\"python ~/.ScriptsAndAliases/JobCat.py -r --substr \"",
    "alias jcats=\"python ~/.ScriptsAndAliases/JobCat.py -s --substr \"",
    "alias jcatsr=\"python ~/.ScriptsAndAliases/JobCat.py -rs --substr \"",
    "alias jcatrs=\"python ~/.ScriptsAndAliases/JobCat.py -rs --substr \"",

    # Allocates a node
    "alias sallocb=\"python ~/.ScriptsAndAliases/AllocateNode.py --nodelist \"",

    # Makes job(s) interactive
    "alias make_interac=\"python ~/.ScriptsAndAliases/MakeInteractive.py --jobs \"",

    # Updates ScriptsAndAliases
    "alias update_scripts_and_aliases=\"cd ~/.ScriptsAndAliases ; python WriteAliases.py ; source ~/.bashrc\"",

    # Updates SLURM scripts' sbatch arguments
    "alias sbu=\"python ~/.ScriptsAndAliases/UpdateSlurmScript.py --substrs \"",
    
    ##################################################################################
    # Useful APEX workstations and servers: DDP and TaskSet
    ##################################################################################
    "alias python_ddp1=\"torchrun --standalone --nnodes=1 --nproc-per-node 1\"",
    "alias python_ddp2=\"torchrun --standalone --nnodes=1 --nproc-per-node 2\"",
    "alias python_ddp3=\"torchrun --standalone --nnodes=1 --nproc-per-node 3\"",
    "alias python_ddp4=\"torchrun --standalone --nnodes=1 --nproc-per-node 4\"",
    "alias python_ddp5=\"torchrun --standalone --nnodes=1 --nproc-per-node 5\"",
    "alias python_ddp6=\"torchrun --standalone --nnodes=1 --nproc-per-node 6\"",
    "alias python_ddp7=\"torchrun --standalone --nnodes=1 --nproc-per-node 7\"",
    "alias python_ddp8=\"torchrun --standalone --nnodes=1 --nproc-per-node 8\"",
    "alias python_ddp9=\"torchrun --standalone --nnodes=1 --nproc-per-node 9\"",
    "alias python_ddp10=\"torchrun --standalone --nnodes=1 --nproc-per-node 10\"",
    "alias tpython=\"python ~/.ScriptsAndAliases/TaskSet.py python\"",
    "alias tpython_ddp1=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp1\"",
    "alias tpython_ddp2=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp2\"",
    "alias tpython_ddp3=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp3\"",
    "alias tpython_ddp4=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp4\"",
    "alias tpython_ddp5=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp5\"",
    "alias tpython_ddp6=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp6\"",
    "alias tpython_ddp7=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp7\"",
    "alias tpython_ddp8=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp8\"",
    "alias tpython_ddp9=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp9\"",
    "alias tpython_ddp10=\"python ~/.ScriptsAndAliases/TaskSet.py python_ddp10\"",
    ##################################################################################
    ##################################################################################
    ##################################################################################

    # Useful on APEX workstations and servers: Miscellanous
    "alias get_wandb_id=\"python -c 'import wandb ; print(wandb.util.generate_id())'\"",
    "alias find_free_gpus=\"python ~/.ScriptsAndAliases/FindFreeGPUs.py --hosts \"",
    "alias killwandb=\"pkill -u $USER -9 wandb\"",
    "alias rsyncb=\"python ~/.ScriptsAndAliases/Rsyncb.py \"",
    "alias send=\"python ~/.ScriptsAndAliases/Rsyncb.py \"",
    "alias getfrom=\"python ~/.ScriptsAndAliases/Rsyncb.py \"",

    "alias check_results=\"python ~/.ScriptsAndAliases/CheckResults.py --files \"",
    "alias update_code=\"python ~/.ScriptsAndAliases/UpdateCode.py --substrs \"",
    
    
    "# END USEFUL ML ALIASES"]


def alias_to_name(alias): return alias.split("=")[0].strip()

def write_aliases_to_file(fname):
    """Writes the aliases to file [fname], removing any that are already there."""
    aliases_names = set([alias_to_name(a) for a in aliases])
    with open(fname, "r") as f:
        existing_lines = f.readlines()

    # Aliases are only edited between the START and END comments if they exist;
    # otherwise they are added to the end
    if any(l.startswith("# START USEFUL ML ALIASES") for l in existing_lines) and any(l.startswith("# END USEFUL ML ALIASES") for l in existing_lines):
        start_idx = existing_lines.index("# START USEFUL ML ALIASES\n")
        end_idx = existing_lines.index("# END USEFUL ML ALIASES\n")
        lines_before = existing_lines[:start_idx]
        lines_after = existing_lines[end_idx+1:]
    else:
        lines_before = existing_lines
        lines_after = []

    new_lines = lines_before + ["\n"] + aliases + ["\n"] + lines_after
    new_lines = [l.strip() if not l == "\n" else l for l in new_lines]
    new_lines_str = "\n".join(new_lines)

    # Allow at most two newlines in a row
    new_lines_str = re.sub(r"\n{3,}", "\n\n", new_lines_str)

    with open(fname, "w") as f:
        f.write(new_lines_str)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--files", nargs="+", default=["~/.bashrc", "~/.zshrc"],
        help="Files to write aliases to")
    args = P.parse_args()

    args.files = [osp.expanduser(f) for f in args.files]
    for fname in args.files:
        if not osp.exists(fname):
            print(f"File {fname} doesn't exist. Create it and run this script again if you really want it.")
            continue
        _ = write_aliases_to_file(fname)
        print(f"Aliases written to {fname}")
