import argparse
import os
import os.path as osp
import subprocess

import ExtractJobIds


def add_to_exclude_list(excluded, slurm_script):
    with open(slurm_script, "r") as f:
        content = f.read()
        lines = content.split("\n")
    exclude_line = [l for l in lines if l.startswith("#SBATCH --exclude=")]
    if len(exclude_line) == 0:
        # Insert an exlude line right below '#!/bin/bash'
        if not content.startswith("#!/bin/bash"):
            print(f"Slurm script {slurm_script} does not start with '#!/bin/bash'. Skipping. Are you sure it's a correct slurm script?")
            return
        content = content.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --exclude=" + ",".join(excluded) + "\n")
        print(f"No exclude line found in {slurm_script}. Adding one...")
    elif len(exclude_line) > 1:
        print(f"Multiple exclude lines found in {slurm_script}. Skipping")
        return
    else:
        exclude_line = exclude_line[0]
        already_excluded = exclude_line.split("=")[1].strip().split(",")
        already_excluded = [n for n in already_excluded if not n == ""]
        excluded = sorted(set(excluded + already_excluded))
        exclude_line_new = f"#SBATCH --exclude={','.join(excluded)}"
        content = content.replace(exclude_line, exclude_line_new)

    with open(slurm_script, "w") as f:
        f.write(content)
    print(f"Updated {slurm_script} to exclude {','.join(excluded)}")


if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("nodes", type=str, nargs="+", help="List of nodes to exclude. If comma-separated, the list is split on commas.")
    args = P.parse_args()


    exclude_nodes = []
    for n in args.nodes:
        if "," in n:
            exclude_nodes += n.split(",")
        else:
            exclude_nodes.append(n)
    exclude_nodes = [n.strip() for n in exclude_nodes]

    command = f"/bin/bash -i -c sq"
    sq = subprocess.getoutput(f"squeue -u $USER -O 'JobArrayID:.10,Name:.175,State:.8,TimeLeft:.10'")
    lines = sq.split("\n")
    lines = [ExtractJobIds.extract_before_underscore(l) for l in lines]
    job_ids = [l.split()[0].strip() for l in lines][1:]

    for j in job_ids:
        scb_output = subprocess.getoutput(f"scontrol show job {j}")
        scb_lines = scb_output.split("\n")
        slurm_script_line = [l for l in scb_lines if l.strip().startswith("Command=")][0]
        slurm_script = slurm_script_line.split("=")[1].strip()

        _ = add_to_exclude_list(exclude_nodes, slurm_script)

