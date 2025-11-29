"""Finds all jobs that can run on an interactive partition and updates them to do so."""
import argparse
import random
import subprocess
import sys
import time

import Utils

def job_is_miggpu(jobinfo_or_jobid):
    """Returns if the job is a MIG GPU job."""
    multi_instance_gpus = ["nvidia_h100_80gb_hbm3_1g.10gb",
        "nvidia_h100_80gb_hbm3_2g.20gb",
        "nvidia_h100_80gb_hbm3_3g.40gb"
        # TODO: Add the A100 variants
        ]

    if isinstance(jobinfo_or_jobid, argparse.Namespace):
        gres = UtilsBase.strip_left(jobinfo_or_jobid.gres, "gres/gpu:")
        gres = UtilsBase.strip_right(gres, ":1")
        return gres in multi_instance_gpus

    elif isinstance(jobinfo_or_jobid, str) and jobinfo_or_jobid.isnumeric():
        cmd = f"scontrol show job {jobinfo_or_jobid} | grep TresPerNode="
        output = subprocess.getoutput(cmd)
        output = output.strip()
        output = output.replace("TresPerNode=gres/gpu", "").replace("TresPerNode=gres/mig", ":1")
        return output in multi_instance_gpus
    else:
        raise NotImplementedError()

def jobid_to_partition(jobid):
    """Returns the partition of [jobid] using scontrol."""
    cmd = f"scontrol show job {jobid} | grep Partition"
    output = subprocess.getoutput(cmd)
    output = output.strip()
    if "Partition=" not in output:
        twrite(f"[ERROR] Could not find partition for job {jobid}. Output was: {output}")
        sys.exit(1)
    partition = output.split()[0].split("Partition=")[1]
    return partition

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--dry_run", action="store_true", help="If set, don't actually do anything.")
    P.add_argument("--interval_mu_sigma", type=float, default=[60, 30], nargs="*",
                   help="Interval (mean, stddev) in seconds to wait between updates.")
    P.add_argument("--jobs", default=[], type=str, nargs="*", help="If set, only consider these jobs.")
    args = P.parse_args()

    jobid_to_update2extant_partitions = dict()
    # In this case, we don't need to do anything fancy to figure out the jobs to
    # update. However, the partition can only be interac
    if len(args.jobs) and all([j.isnumeric() for j in args.jobs]):
        jobid_to_update2extant_partitions = {j: jobid_to_partition(j) for j in args.jobs}
    else:
        import Utils
        import UtilsBase
        from UtilsBase import twrite

        job2info = Utils.get_slurm_status(cur_user=True)
    
        # Anything that requests a single, MIG GPU for under 8H can run interactively
        jobid_to_update2extant_partitions = {j: info.partition for j,info in job2info.items()
            if job_is_miggpu(info)
            and UtilsBase.time_to_hours(info.time_limit) <= 8
            and not info.state in ["RUNNING", "COMPLETING"]}

    for jobid,extant_partitions in jobid_to_update2extant_partitions.items():

        # Hack for Vulcan
        if Utils.get_cluster_type() == "vulcan":
            extant_partitions = [p for p in extant_partitions.split(",") if not p.startswith("gpubase_bygpu") or p =="interac"]
            extant_partitions = ",".join(extant_partitions)
            new_partitions = ["gpubase_interac"]
        else:
            new_partitions = ["gpubase_interac", "interac"]

        partitions = list(set(extant_partitions.split(",")) | set(new_partitions))
        partitions_str = ",".join(partitions)

        print(f"[INFO] Updating job {jobid} Partition={extant_partitions} -> {partitions_str}")
        cmd = f"scontrol update JobId={jobid} Partition={partitions_str}"
        print(f"[INFO] Running command: {cmd}")
        
        if not args.dry_run:
            output = subprocess.getoutput(cmd)
            print(f"[INFO] Command output: {output}")

        if len(args.interval_mu_sigma) and args.interval_mu_sigma[0]:
            wait_time = max(0, int(random.gauss(args.interval_mu_sigma[0], args.interval_mu_sigma[1])) )
            print(f"[INFO] Sleeping for seconds={wait_time} before continuing...")
            time.sleep(wait_time)
        