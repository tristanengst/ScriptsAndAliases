"""Finds jobs whose names are identical up to the GPUs and time required."""
from collections import defaultdict
import subprocess
import re

import Utils

if __name__ == "__main__":
    sq_output = subprocess.getoutput("squeue -u $USER -O 'JobArrayID:.20,Name:.300'")
    lines = sq_output.split("\n")[1:]  # Skip the header line

    job_name2job_ids = defaultdict(list)
    for line in lines:
        job_id = line.split()[0].strip()
        job_name = line.split()[1].strip()
        job_name = re.sub(r'-gpus\d+-\d\w*$', '', job_name)
        job_name2job_ids[job_name].append(job_id)

    if all(len(job_ids) == 1 for job_ids in job_name2job_ids.values()):
        print(f"Checked {len(job_name2job_ids)} jobs, no duplicates found")
    else:
        print("Duplicate jobs found:")
        for job_name, job_ids in job_name2job_ids.items():
            if len(job_ids) > 1:
                print(f"Job name (prefix): {job_name}, Job IDs: {', '.join(job_ids)}")

    

