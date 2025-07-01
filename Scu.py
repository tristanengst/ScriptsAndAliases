import argparse
import os
import subprocess
import Utils

def apply_to_jobid(*, jobid, cmd):
	"""Tries to cancel [jobid] and returns True if it was successful."""
	if cmd == "hold":
		result = subprocess.getoutput(f"scontrol hold {jobid}")
	else:
		result = subprocess.getoutput(f"scontrol update job {jobid} {cmd}")
	return len(result) == 0, result

P = argparse.ArgumentParser()
P.add_argument("cmd", help="scontrol command like TimeLimit=8:00:00")
P.add_argument("jobs", nargs="+", help="sequence of job ids or UIDs to update")
args = P.parse_args()

uid2jobs = None
for j in args.jobs:
	success_by_jobid, result_by_jobid = apply_to_jobid(jobid=j, cmd=args.cmd)
	if success_by_jobid:
		continue

	uid2jobs = Utils.jobid2info_to_uid2jobids() if uid2jobs is None else uid2jobs
	if j in uid2jobs:
		jobid2result = dict()
		for j_ in uid2jobs[j]:
			success_by_uid, result_by_uid = apply_to_jobid(jobid=j_, cmd=args.cmd)
			jobid2result[j_] = dict(result=result_by_uid, success=success_by_uid)
			
		if all(jobid2result[j_]["success"] for j_ in uid2jobs[j]):
			continue