import argparse
from collections import defaultdict
import os
import subprocess
import sys
import Utils
from UtilsBase import twrite

if __name__ == "__main__":
	P = argparse.ArgumentParser()
	P.add_argument("cmd", help="scontrol command like TimeLimit=8:00:00")
	P.add_argument("jobs", nargs="+", help="sequence of job IDs or identifying job name substrings to update. Substrings must contain a UID")
	P.add_argument("-f", "--force", action="store_true",
		help="force the command to be applied to all jobs")
	P.add_argument("--dry_run", action="store_true",
		help="print the commands that would be run, but do not actually run them")
	args = P.parse_args()

	cmd_prefix = args.cmd if args.cmd in ["hold", "release"] else "update job"
	scontrol_command = "" if args.cmd in ["hold", "release"] else args.cmd

	job2info = Utils.get_slurm_status(cur_user=True)

	# First, check that the UIDs in job2info are unique. If they are not, it is dangerous
	# to match on them.
	uid2count = defaultdict(int)
	for info in job2info.values():
		uid2count[info.uid] += 0 if info.state == "COMPLETING" else 1
	non_unique_uid2count = {uid: count for uid,count in uid2count.items() if count > 1}
	allow_uid_matching = len(non_unique_uid2count) == 0
	_ = twrite(f"[WARNING] Detected non-unique UIDS with counts: {non_unique_uid2count}. UID matching is disabled.", quiet=allow_uid_matching)

	job_substr2jobid_to_update = {j: info.jobid for j in args.jobs for info in job2info.values() if (
		j == info.jobid
		or (not info.uid is None and info.uid in j and allow_uid_matching)
		or (j in info.name and len([jn for jn in job2info.values() if j in jn.name]) == 1)
	)}

	jobs_with_unfound_jobid = [j for j in args.jobs if not j in job_substr2jobid_to_update]
	if len(jobs_with_unfound_jobid) > 0:
		twrite(f"[WARNING] Could not find job IDs for the following job substrings: {jobs_with_unfound_jobid}")
		twrite(f"[INFO] Found jobid2update: {job_substr2jobid_to_update}")
		twrite(f"Correct the input and try again.")
		sys.exit(1)


	for jobid in job_substr2jobid_to_update.values():
		if "TimeLimit" in args.cmd and not job2info[jobid].state == "PENDING" and not args.force:
			twrite(f"[WARNING] Job {jobid} is not pending, cannot change TimeLimit. Use --force to override.")
			continue
		elif args.cmd == "hold" and not job2info[jobid].state in ["RUNNING"] and not args.force:
			# I actually don't recall the exact semantics of holding running jobs, so this feels safer
			twrite(f"[WARNING] Job {jobid} is not running, cannot hold. Use --force to override.")
			continue
		
		command_to_run = f"scontrol {cmd_prefix} {jobid} {scontrol_command}"

		if args.dry_run:
			twrite(f"[DRY RUN] Would run command: {command_to_run}")
			result = ""
		else:
			result = subprocess.getoutput(command_to_run)

		if len(result) == 0:
			twrite(f"[INFO] Successfully ran command '{command_to_run}'")
		else:
			twrite(f"[WARNING] Command '{command_to_run}' failed with result: {result}")
