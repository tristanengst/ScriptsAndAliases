import argparse
from collections import defaultdict
from functools import cache
import os
import os.path as osp
import subprocess
import sys

import ClusterInfo2
import MachineInfo
import Utils
import UtilsBase
from UtilsBase import twrite, tqdm

def find_partitions_for_job(*, time_limit, full_node, gpu_type=None, args):
	"""Returns a list of the best partitions for a given time limit.
	
	Args:
	time_limit	-- time limit in seconds (float or int) or a string formatted that can
					be parsed into seconds
	full_node	-- whether the job requests a full node
	gpu			-- list of strings identifying GPUs requested, or None if no GPU
					requirement. Can be used to filter out partitions that don't have
					the requested GPU type.
					
					CLUSTER PARTICULARITIES:
					- KILLARNEY: Jobs request H100 or L40s GPUs, corresponding to
						gpubase_h100_bX and gpubase_l40s_bX partitions. If no specific
						GPU type is requested, then the scheduler *defaults* to the
						appropriate gpubase_l40s_bX partition. This means that it
						might be optimal to request no specific GPU type, and then
						update it to have partitions for both GPU types. Note that
						H100 GPUs are billed at a higher rate, so the 'give the job
						the most ways to run' heuristic won't necessarily be optimal.
					- TAMIA: TBD???
	args 		-- argparse Namespace
	"""
	if isinstance(time_limit, str) and any(char in time_limit for char in [":", "H", "M", "S", "D"]):
		time_limit = UtilsBase.time_to_seconds(time_limit)

	# Get the eligible partitions by filtering by time limit, interactiveness,
	# full-node-ness, and gpu type.
	partitions = UtilsBase.flatten([ps for t,ps in ClusterInfo2.time2partition_names().items() if t >= time_limit])
	partitions = [p for p in partitions if not "interac" in p]
	partitions = [p for p in partitions if full_node or not "bynode" in p]
	partitions = [p for p in partitions if gpu_type is None or gpu_type in p]
	partition2time = {p: t for t,ps in ClusterInfo2.time2partition_names().items() for p in ps if p in partitions}
	
	min_time = min([partition2time[p] for p in partitions], default=float("inf"))
	partitions = [p for p in partitions if partition2time[p] == min_time]
	return partitions

def expand_partitions_to_true_partitions(partitions, verbose=False):
	"""Returns [partitions] expanded to be the true partitions on the cluster."""
	def partition_to_matches(p):
		true_partitions = [tp.name for tp in ClusterInfo2.get_all_partitions()]
		matches = [tp for tp in true_partitions if p in tp]
		if len(matches) == 0:
			twrite(f"[WARNING] partition={p} does not match any true partitions={true_partitions} -> not expanding")
			return [p]
		elif len(matches) > 1:
			twrite(f"[WARNING] partition={p} matches multiple true partitions={matches} among {true_partitions} -> expanding to both")
			return matches
		else:
			return matches

	partition_list = partitions.split(",") if isinstance(partitions, str) else partitions
	partition2expanded = {p: partition_to_matches(p) for p in partition_list}
	expanded = set(UtilsBase.flatten(partition2expanded.values()))
	return ",".join(expanded) if isinstance(partitions, str) else expanded

def job_info_to_time_limit_full_node(job_info):
	"""Returns a dictionary giving the the time limit and whether a full node is
	requested for a given job_info.
	"""
	from Sqb2 import job_info_with_formatted_resources
	job_info = job_info_with_formatted_resources(job_info)
	time_limit = job_info.time_limit

	# Heuristic: a job requires a full node when it requests a full node's worth of
	# resources. This is indicated either (a) requesting multiple nodes, (b) requesting --mem=0, or (c) requesting
	if any(["bynode" in p.name for p in ClusterInfo2.get_all_partitions()]):
		if "nodes" in vars(job_info) and not job_info.nodes is None and int(job_info.nodes) > 1:
			full_node = True	
		elif "mem" in vars(job_info) and not job_info.mem is None and int(UtilsBase.remove_nonnumeric(job_info.mem)) == 0:
			full_node = True
		elif "gres" in vars(job_info) and not job_info.gres is None and "gpus" in vars(job_info) and not job_info.gpus is None:
			gpu_type = MachineInfo.str_to_gpu_type(job_info.gres)
			if gpu_type is None:
				twrite(f"[WARNING] couldn't parse gres={job_info.gres} for job {job_info.jobid} to a GPU type -> assume inelligible for a full node")
				full_node = False
			else:
				node_config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
				gpus_per_node = node_config[gpu_type].gpus_per_node if gpu_type in node_config else float("inf")
				full_node = (float(job_info.gpus) >= gpus_per_node)
		else:
			full_node = False
	else:
		full_node = False

	return dict(time_limit=time_limit, full_node=full_node)

# def args_with_sanitized_partitions(args):
# 	"""Returns [args] with partition inputs sanitized."""
# 	def update_partition(*, partition, avail_partitions):
# 		if not partition in avail_partitions and f"gpubase_{partition}" in avail_partitions:
# 			twrite(f"[INFO] Mapping partition={partition} -> gpubase_{partition}")
# 			return f"gpubase_{partition}"
# 		elif not partition in avail_partitions:
# 			twrite(f"[WARNING] Partition '{partition}' is not in available partitions={avail_partitions}. This may be fine if it is a CPU-only one, but otherwise might not work....")
# 			return partition
# 		else:
# 			return partition
		
# 	if args.partition is None:
# 		return args

# 	partitions = [update_partition(partition=p, avail_partitions=get_partitions()) for p in args.partition.split(",")]
# 	partitions = ",".join(partitions)
# 	return UtilsBase.updated_namespace(args, partition=partitions)

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
		elif update == "added_partitions":
			return "AddedPartitions"
		else:
			return update.capitalize()

	kv_scontrol_updates = ["dependency", "account", "partition", "time", "added_partitions"]
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
	elif cmd_prefix == "hold" and job_info.state in ["RUNNING"] and not args.force:
		# I actually don't recall the exact semantics of holding running jobs, so this feels safer
		twrite(f"[WARNING] Job {job_info.jobid} is not running, cannot hold. Use --force to override.")
		return

	# If updating a time limit, then optionally also update the partition to match.
	# This means that partitions should be specified specifically with the intent of
	# keeping them, ie. with --added_partitions or +p, rather than with --partitions
	if ("TimeLimit" in update2value
		and (args.added_partitions or not args.partition)
		and args.match_partition_to_time):
		
		time_limit_full_node = job_info_to_time_limit_full_node(job_info)
		new_partitions_for_time_limit_list = find_partitions_for_job(
			time_limit=update2value["TimeLimit"],
			full_node=time_limit_full_node["full_node"],
			args=args)
		new_partitions_for_time_limit = ",".join(new_partitions_for_time_limit_list)

		# If c
		if "AddedPartitions" in update2value:
			partitions_to_add = update2value["AddedPartitions"].split(",")
			new_partitions_for_time_limit_list = list(set(new_partitions_for_time_limit_list + partitions_to_add))
			new_partitions_for_time_limit = ",".join(new_partitions_for_time_limit_list)
		update2value["Partition"] = new_partitions_for_time_limit
	elif ("TimeLimit" in update2value and args.partition and args.match_partition_to_time):
		twrite(f"[WARNING] Can not simultaneously update TimeLimit and change partitions with --partition. To ensure the job is queued on a particular partition, use --added_partitions or +p. Skipping....")
		return None
	
	elif "AddedPartitions" in update2value and not args.partition:
		if job_info.partition is None:
			raise NotImplementedError()
		else:
			current_partitions = job_info.partition.split(",")
		partitions_to_add = update2value["AddedPartitions"].split(",")
		new_partitions = list(set(current_partitions + partitions_to_add))
		new_partitions = ",".join(new_partitions)
		update2value["Partition"] = new_partitions
		update2value = {u: v for u,v in update2value.items()}
	elif "AddedPartitions" in update2value and args.partition:
		twrite(f"[WARNING] Can not use --added_partitions or +p with --partition since nonsensical. Skipping....")
		return None


	# args.match_partition_to_time and not "Partition" in update2value:
	# 	from Sqb2 import job_info_with_formatted_resources
	# 	from ClusterInfo2 import Partition

	# 	# Find the number of GPUs requested by the job as a lower bound/proxy for its
	# 	# node fraction. Then filter possible partitions by (1) whether they are
	# 	# interactive if that's disallowed, (2) whether their time limits are
	# 	# sufficient, (3) whether they are full-node partitions and the job requests
	# 	# at least a node of resources. Finally, any partition that is strictly
	# 	# contained within another partition is removed.
	# 	job_info = job_info_with_formatted_resources(job_info) 
	# 	partitions = get_partitions(interac=args.interac)
	# 	job_time_limit = UtilsBase.time_to_seconds(args.time)
	# 	partitions = [p for p in partitions if p.seconds >= job_time_limit]
	# 	partitions = [p for p in partitions if not "bynode" in p or (not job_info.gpus is None and p.max_total_gpus <= float(job_info.gpus))]
	# 	partitions = Partition.filter_partitions(partitions)
	# 	update2value |= dict(Partition=",".join([p for p in partitions]))

	# 	if verbose:
	# 		twrite(f"[INFO] Matching partitions to new time limit {args.time} for job {job_info.jobid}. Candidate partitions: {partitions}")
	# else:
	# 	twrite(match_partition_to_time=args.match_partition_to_time, interac=args.interac, time=args.time, update2value=update2value, verbose=verbose)


	valid_suffixs = ["Dependency", "Account", "Partition", "TimeLimit"]
	cmd_suffix = " ".join([f"{u}={v}" for u,v in update2value.items() if u in valid_suffixs])
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
	P = argparse.ArgumentParser(prefix_chars="-+")
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
	P.add_argument("+p", "++partition", "++partitions", dest="added_partitions", default=None,
		help="Add this comma-separated list of partitions")
	P.add_argument("--time", default=None,
		help="Update jobs to new time limit, in format HH:MM:SS")

	# Configuration options for how SCU should behave
	P.add_argument("--match_partition_to_time", type=UtilsBase.truthy_type, default=True,
		help="If updating time, match partitions to the new time limit")

	P.add_argument("--stagger", action="store_true",
		help="Stagger job updates. Occassionally useful")
	args = P.parse_args()
	
	##################################################################################
	# Sanitize and interpret [args]
	##################################################################################
	if len(args.jobs) >= 2 and args.jobs[0] == "hold" and not args.release:
		args.hold = True
	if len(args.jobs) >= 2 and args.jobs[0] == "release" and not args.hold:
		args.release = True
	if len(args.jobs) >= 2 and any([args.jobs[0].startswith(k) for k in ["TimeLimit=", "Account=", "Partition=", "Dependency="]]):
		num_removed_args = 0
		for j in args.jobs:
			if j.startswith("TimeLimit=") and not args.time:
				args = UtilsBase.updated_namespace(args, time=j.split("TimeLimit=")[-1])
				num_removed_args += 1
			elif j.startswith("Account=") and not args.account:
				args = UtilsBase.updated_namespace(args, account=j.split("Account=")[-1])
				num_removed_args += 1
			elif j.startswith("Partition=") and not args.partition:
				args = UtilsBase.updated_namespace(args, partition=j.split("Partition=")[-1])
				num_removed_args += 1
			elif j.startswith("Dependency=") and not args.dependency:
				args = UtilsBase.updated_namespace(args, dependency=j.split("Dependency=")[-1])
				num_removed_args += 1
			else:
				break
		args = UtilsBase.updated_namespace(args, jobs=args.jobs[num_removed_args:])
		if len(args.jobs) == 0:
			twrite(f"[WARNING] No job substrings provided after parsing updates from the positional arguments. Please provide job substrings as positional arguments or include the updates as flags like --time, --account, etc.")
			sys.exit(1)

	if not args.partition is None:
		args = UtilsBase.updated_namespace(args, partition=expand_partitions_to_true_partitions(args.partition, verbose=True))
	if not args.added_partitions is None:
		args = UtilsBase.updated_namespace(args, added_partitions=expand_partitions_to_true_partitions(args.added_partitions, verbose=True))
	##################################################################################
	##################################################################################
	##################################################################################
	
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

	iter_over = job_substr2jobid_to_update.values()
	iter_over = tqdm(iter_over, desc="Updating jobs", total=len(iter_over)) if args.stagger else iter_over

	for idx,jobid in enumerate(iter_over):
		if args.stagger and idx > 0:
			import random
			import time
			wait_time = int(random.gauss(60, 30))
			twrite(f"[INFO] Sleeping for seconds={wait_time} before updating jobid={jobid}...")
			time.sleep(wait_time)

		_ = update_job(job_info=job2info[jobid], args=args, verbose=True)
		
		
