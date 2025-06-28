"""Better version of scancel.

When run on JobIDs, identical to scancel. However, it can also handle UIDs, and will cancel *all* jobs with a given UID.
"""
import subprocess
import argparse
import Utils

if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Cancel jobs by JobID or UID.")
    P.add_argument("jobs", nargs="+", help="JobID or UID to cancel")
    args = P.parse_args()

    jobid2info = Utils.get_slurm_status(cur_user=True)
    job_ids_to_cancel = []

    uid2job_ids = None
    for j in args.jobs:
        if j in jobid2info:
            job_ids_to_cancel.append(j)
        else:
            if uid2job_ids is None:
                uid2job_ids = dict()
                for job_id,info in jobid2info.items():
                    if not info["UID"] is None and not info["UID"] in uid2job_ids:
                        uid2job_ids[info["UID"]] = [job_id]
                    elif not info["UID"] is None:
                        uid2job_ids[info["UID"]].append(job_id)

            if j in uid2job_ids:
                job_ids_to_cancel += uid2job_ids[j]
            
            else:
                print(f"Job {j} not found. Skipping.")
                continue
    
    subprocess.getoutput(f"scancel {' '.join(job_ids_to_cancel)}").strip()
