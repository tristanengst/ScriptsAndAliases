import argparse
import os
import os.path as osp
from multiprocessing import Pool
import subprocess

import ExtractJobIds


def modify_exclude_list(*, included, excluded, slurm_script):
    with open(slurm_script, "r") as f:
        content = f.read()
        lines = content.split("\n")
    exclude_lines = [l for l in lines if l.startswith("#SBATCH --exclude=")]
    if len(exclude_lines) == 0:
        if not content.startswith("#!/bin/bash"):
            print(f"Slurm script {slurm_script} does not start with '#!/bin/bash'. Skipping. Are you sure it's a correct slurm script?")
            return
        else:
            content = content.replace("#!/bin/bash", "#!/bin/bash\n#SBATCH --exclude=")
            exclude_line = "#SBATCH --exclude="
            print(f"No exclude line found in {slurm_script}. Adding one...")
    elif len(exclude_lines) > 1:
        print(f"Multiple exclude lines found in {slurm_script}. Skipping")
        return
    else:
        exclude_line = exclude_lines[0]

    already_excluded = exclude_line.split("=")[1].strip().split(",")
    already_excluded = [n for n in already_excluded if not n == ""]
    all_excluded = already_excluded + excluded
    all_nodes = [n for n in all_excluded if n not in included]
    all_nodes = sorted(set(all_nodes))

    exclude_str = ",".join(all_nodes) if all_nodes else ""
    exclude_line_new = f"#SBATCH --exclude={exclude_str}"
    content = content.replace(exclude_line, exclude_line_new)

    with open(slurm_script, "w") as f:
        f.write(content)

    to_print_exclude_str = ",".join(all_nodes) if all_nodes else "no nodes"
    print(f"[INFO] slurm_script={slurm_script} excludes={to_print_exclude_str}")


def job_id_to_slurm_script(job_id):
    """Returns the SLURM script that generated the job with id [job_id]."""
    scb_output = subprocess.getoutput(f"scontrol show job {job_id}")
    scb_lines = scb_output.split("\n")
    slurm_script_line = [l for l in scb_lines if l.strip().startswith("Command=")][0]
    slurm_script = slurm_script_line.split("=")[1].strip()
    return slurm_script


if __name__ == "__main__":
    P = argparse.ArgumentParser(prefix_chars="-+")
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("-s", "--substrs", type=str, nargs="+", default=None,
        help="List of SUBMITTED job name substrings to modify. None by default and modifies all jobs.")
    P.add_argument("-n", "--exclude", nargs="+", default=[],
        help="List of nodes to exclude")
    P.add_argument("+n", "--include", type=str, nargs="+", default=[],
        help="List of nodes to exclude")
    args = P.parse_args()

    exclude_nodes = [n.split(",") for n in args.exclude]
    exclude_nodes = UtilsBase.flatten(exclude_nodes)
    exclude_nodes = [n.strip() for n in exclude_nodes]

    include = [n.split(",") for n in args.include]
    include = UtilsBase.flatten(include)
    include = [n.strip() for n in include]

    included_and_excluded = set(include) & set(exclude_nodes)
    if included_and_excluded:
        print(f"[ERROR] nodes={sorted(included_and_excluded)} are both included and excluded. Fix this.")
        exit(1)

    # I haven't tested this with actual array jobs!
    sq_output = subprocess.getoutput(f"squeue -u $USER --noheader -O 'JobArrayID:.20,Name:.400'")
    job_id2job_name = {s.split()[0]: " ".join(s.split()[1:]) for s in sq_output.split("\n")}

    if args.substrs:
        args.substrs = [s.strip("*") for s in args.substrs]
        job_id2job_name = {j: n for j,n in job_id2job_name.items() if any([s in n for s in args.substrs])}

    with Pool(min(os.cpu_count(), max(1, len(sq)))) as p:
        slurm_scripts = p.map(job_id_to_slurm_script, job_id2job_name.keys())

    for slurm_script in slurm_scripts:
        _ = modify_exclude_list(included=include, excluded=exclude_nodes, slurm_script=slurm_script)

