import argparse
from collections import defaultdict
from functools import cache
import os
import subprocess
import sys
import Utils
import UtilsBase
from UtilsBase import twrite

@cache
def time2partitions():
	"""Returns a dict mapping time limits to lists of partitions for which that time
	limit is the maximum. Cached because it's unlikely to change. The time limits are
	in seconds. Note that interactive partitions are included by default!
	"""
	from ClusterInfo2 import Partition
	partitions = Partition.get_partitions_from_sinfo()
	time2partitions = defaultdict(list)
	for p in partitions:
		time2partitions[p.time].append(p.name)
	time2partitions = {UtilsBase.time_to_seconds(t): p for t,p in time2partitions.items()}
	return time2partitions

@cache
def get_partitions(interac=False):
	"""Returns a list of partitions on the current cluster, excluding interactive ones by default."""
	from ClusterInfo2 import Partition
	partitions = Partition.get_partitions_from_sinfo()
	return partitions if interac else [p for p in partitions if not "interac" in p.name]

def args_with_sanitized_partitions(args):
	"""Returns [args] with partition inputs sanitized."""
	def update_partition(*, partition, avail_partitions):
		if not partition in avail_partitions and f"gpubase_{partition}" in avail_partitions:
			twrite(f"[INFO] Mapping partition={partition} -> gpubase_{partition}")
			return f"gpubase_{partition}"
		elif not partition in avail_partitions:
			twrite(f"[WARNING] Partition '{partition}' is not in available partitions={avail_partitions}. This may be fine if it is a CPU-only one, but otherwise might not work....")
			return partition
		else:
			return partition
		
	if args.partition is None:
		return args

	time2partitions = time2partitions if time2partitions else get_time_to_partitions()
	avail_partitions = UtilsBase.flatten(time2partitions.values())
	partitions = [update_partition(partition=p, avail_partitions=avail_partitions) for p in args.partition.split(",")]
	partitions = ",".join(partitions)
	return UtilsBase.update_argparse(args, partition=partitions)

def args_to_scontrol_prefix(args):
	if args.hold:
		return "hold"
	elif args.release:
		return "release"
	else:
		return "update job"

def args_to_scontrol_update_dict(args):
	"""Returns a mapping from scontrol update keys to their new values according to
	[args]. For example, if [args] has time=8:00:00, then the returned dict will have
	"TimeLimit" as a key and "8:00:00" as its value.
	"""
	def update_to_update_str(update):
		if update == "time":
			return "TimeLimit"
		else:
			return update.capitalize()

	kv_scontrol_updates = ["dependency", "account", "partition", "time"]
	update2update_str = dict(time="TimeLimit")

	args_update2value = {k: v for k,v in vars(args).items() if k in kv_scontrol_updates and not v is None}
	update2value = {update_to_update_str(u): v for u,v in args_update2value.items()}
	return update2value 

def update_job(*, job_info, args, verbose=True):
	"""Updates the job of [job_info] according to [args]."""
	cmd_prefix = args_to_scontrol_prefix(args)

	update2value = args_to_scontrol_update_dict(args)
	if "TimeLimit" in update2value and not job_info.state == "PENDING" and not args.force:
		twrite(f"[WARNING] Job {job_info.jobid} is not pending, cannot change TimeLimit. Use --force to override.")
		return
	elif cmd_prefix == "hold" and not job_info.state in ["RUNNING"] and not args.force:
		# I actually don't recall the exact semantics of holding running jobs, so this feels safer
		twrite(f"[WARNING] Job {job_info.jobid} is not running, cannot hold. Use --force to override.")
		return

	# If updating a time limit, then optionally also update the partition to match.
	# NOTE: we actually want to append to existing partitions rather than replace them
	if "TimeLimit" in update2value and args.match_partition_to_time and not "Partition" in update2value:
		from Sqb2 import job_info_with_formatted_resources
		from ClusterInfo2 import Partition

		# Find the number of GPUs requested by the job as a lower bound/proxy for its
		# node fraction. Then filter possible partitions by (1) whether they are
		# interactive if that's disallowed, (2) whether their time limits are
		# sufficient, (3) whether they are full-node partitions and the job requests
		# at least a node of resources. Finally, any partition that is strictly
		# contained within another partition is removed.
		job_info = job_info_with_formatted_resources(job_info) 
		partitions = get_partitions(interac=args.interac)
		job_time_limit = UtilsBase.time_to_seconds(args.time)
		partitions = [p for p in partitions if p.seconds >= job_time_limit]
		partitions = [p for p in partitions if not "bynode" in p.name or (not job_info.gpus is None and p.max_total_gpus <= float(job_info.gpus))]
		partitions = Partition.filter_partitions(partitions)
		update2value |= dict(Partition=",".join([p.name for p in partitions]))

		if verbose:
			twrite(f"[INFO] Matching partitions to new time limit {args.time} for job {job_info.jobid}. Candidate partitions: {partitions}")
	else:
		twrite(match_partition_to_time=args.match_partition_to_time, interac=args.interac, time=args.time, update2value=update2value, verbose=verbose)

		
	
	cmd_suffix = " ".join([f"{u}={v}" for u,v in update2value.items()])
	
	
	command_to_run = f"scontrol {cmd_prefix} {jobid} {cmd_suffix}"

	if args.dry_run:
		twrite(f"[DRY RUN] Would run command: {command_to_run}")
		result = ""
	else:
		result = subprocess.getoutput(command_to_run)

	if len(result) == 0:
		twrite(f"[INFO] Successfully ran command '{command_to_run}'")
	else:
		twrite(f"[WARNING] Command '{command_to_run}' failed with result: {result}")




if __name__ == "__main__":
	P = argparse.ArgumentParser()
	P.add_argument("jobs", nargs="+",
		help="List of JOBIDs or identifying substrings to update")
	# P.add_argument("jobs", nargs="+", help="sequence of job IDs or identifying job name substrings to update. Substrings must contain a UID")
	P.add_argument("-f", "--force", action="store_true",
		help="force the command to be applied to all jobs")
	P.add_argument("--dry_run", action="store_true",
		help="print the commands that would be run, but do not actually run them")

	# Sensible things you could change with scontrol ... about a job.
	# Basically, I think it's overall better to force this to be done with keyword
	# arguments, and the number of things we update is small enough that hardcoding
	# them isn't an issue. However, this *DOES* break the 'scu is just like scontrol
	# update' paradigm.
	P.add_argument("--hold", action="store_true",
		help="Hold the specified jobs")
	P.add_argument("--release", action="store_true",
		help="Release the specified jobs")
	P.add_argument("--dependency", default=None,
		help="Update jobs to new dependency, in format like 'afterok:12345'")
	P.add_argument("--account", default=None,
		help="Update jobs to new account")
	P.add_argument("-p", "--partition", "--partitions", default=None,
		help="Update jobs to comma-separated list of partitions")
	P.add_argument("--time", default=None,
		help="Update jobs to new time limit, in format HH:MM:SS")

	# Configuration options for how SCU should behave
	P.add_argument("--match_partition_to_time", type=UtilsBase.truthy_type, default=True,
		help="If updating time, match partitions to the new time limit.")
	P.add_argument("--interac", action="store_true",
		help="Allow adding interactive parititions to partition lists when updating a time limits. Only applies if --match_partition_to_time is also set.")
	args = P.parse_args()

	args = args_with_sanitized_partitions(args)

	job2info = Utils.get_slurm_status(cur_user=True)

	##################################################################################
	# First, check that the UIDs in job2info are unique. If they are not, it is
	# dangerous to match on them
	##################################################################################
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
	##################################################################################
	##################################################################################
	##################################################################################

	

		

	



	for jobid in job_substr2jobid_to_update.values():
		_ = update_job(job_info=job2info[jobid], args=args, verbose=True)

		# if args.partition is None:
		# 	from ClusterInfo2 import Partition
		# 	partitions = Partition.get_partitions_from_sinfo()
		# 	time2partitions = defaultdict(list)
		# 	for p in partitions:
		# 		time2partitions[p.time].append(p.name)


		# # TimeLimit and Partition updates handled specially
		# if args.time is None and args.cmd.startswith("TimeLimit"):
		# 	args.time = args.cmd.split("TimeLimit=")[-1]
		# 	args.cmd = None

		# # Check: does the job request a full node?



		
		
