import argparse
import subprocess

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--script", required=True, help="Path to the script to be executed")
    P.add_argument("--num_jobs", type=int, required=True, help="Number of jobs to submit")
    P.add_argument("--prev_job_id", default=None, help="Previous job ID to depend on")
    args = P.parse_args()

    for idx in range(args.num_jobs):
        dependency_str = f"-d afterany:{args.prev_job_id}" if args.prev_job_id else ""
        resubmit_str = f"sbatch {dependency_str} --parsable {args.script}"
        print(f"Submission {idx+1}/{args.num_jobs}: {resubmit_str}")
        new_job_id = subprocess.run(resubmit_str, capture_output=True, shell=True, text=True)
        new_job_id = new_job_id.stdout.strip()
        args.prev_job_id = new_job_id