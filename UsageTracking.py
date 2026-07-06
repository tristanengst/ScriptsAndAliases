import argparse
from datetime import datetime, timezone, timedelta
from functools import cached_property
import json
import os
import os.path as osp
import shutil
import subprocess
import zoneinfo

import MachineInfo
from MachineInfo import gpu_name2alias, gpu2info, gpu2vram, cluster2node2config
import UserConfig
import Utils
import UtilsBase
from UtilsBase import twrite

DEFAULT_SQUEUE_FORMAT_NAME2FIELD = dict(
    jobid="JobID",
    submit="SubmitTime",
    eligible="EligibleTime",
    stderr="StdErr",
    stdout="StdOut",
    tres_per_task="tres-per-task", # Can work as a fallback for gres
    tres_per_node="tres-per-node", # Can work as a fallback for gres
    tres_per_job="tres-per-job", # Can work as a fallback for gres
    num_tasks="NumTasks",
    comment="Comment",
    account="Account",
    host="NodeList",
    partition="Partition",
    jobname="Name",
    user="UserName",
    state="State",
    start="StartTime",
    end="EndTime",
    time_left="TimeLeft",
    time_limit="TimeLimit",
    nodes="NumNodes",
    reason="Reason",
    exclude="Exclude",
    dependency="Dependency",
    priority="Priority",
)

DEFAULT_SACCT_FORMAT_NAME2FIELD = dict(
    jobid="jobid",
    jobname="jobname",
    account="account",
    requested="reqtres",
    allocated="alloctres",
    submit="submit",
    eligible="eligible",
    start="start",
    end="end",
    state="state",
    partition="partition",
    priority="priority",
    user="user",
    elapsed="elapsedraw",
    exitcode="exitcode",
    failednode="failednode",
    timelimit="timelimitraw",
    submitline="submitline",
)

def get_slurm_accounts_str(*accounts):
    """Returns the string that is a command-line argument specifying to query only for
    the given accounts [accounts]. If [accounts] is empty or 'all', then returns an
    empty string indicating no restriction.
    """
    accounts = [] if accounts == ["all"] else accounts
    return ("--accounts=" + ",".join(accounts)) if accounts else ""

def get_slurm_user_str(*users):
    """Returns the string that is a command-line argument specifying to query only for
    the given users [users]. If [users] is empty or 'all', returns an empty string
    indicating no restriction.
    """
    users = [] if users == ["all"] else users
    return ("--user=" + ",".join(users)) if users else ""
def get_slurm_jobid_str(*jobids):
    """Returns the string that is a command-line argument specifying to query only for
    the given jobids [jobids]. If [jobids] is empty or 'all', returns an empty string
    indicating no restriction.
    """
    jobids = [] if jobids == ["all"] else jobids
    return ("--jobs=" + ",".join([str(j) for j in jobids])) if jobids else ""


def parse_resources(tres_str):
    """Returns a string of the form RESOURCE_TYPE1=COUNT1,RESOURCE_TYPE2=COUNT2,...
    for the given [tres_str] string as a dictionary mapping resource types to their
    counts. Resource types get standardized to a more useful form.
    
    If no '=' sign is present, returns an empty dictionary.

    A bunch of post-processing is carried out here.
    """
    resource_counts = [x.strip() for x in tres_str.split(",") if "=" in x]
    result = dict()
    for kv in resource_counts:
        k, v = kv.split("=", 1)    
        if v == "":
            continue

        if k == "node":
            result["nodes"] = int(v)
        elif k == "gres/gpu" and v.isdigit():
            result["gpus_per_node"] = int(v)
        elif k == "gres/gpu" and ":" in v:
            result["gpus_per_node"] = int(v.split(":")[-1])
            result["gpu_name"] = v.split(":")[0] if len(v.split(":")) > 2 else result.get("gpu_name", None)
        elif k.startswith("gres/gpu:"):
            result["gpu_name"] = k.split(":")[1]
            result["gpus_per_node"] = int(v)
        elif k == "billing":
            result["billing"] = int(v) 
        elif k == "cpu":
            result["cpus"] = int(v)
        elif k == "mem":
            # At least for Nibi, requesting in 'G' or 'M' is equivalent to requesting in base-2 index 'GiB'
            result["mem"] = UtilsBase.unit_conversion(v, source="GiB" if v.endswith("G") else ("MiB" if v.endswith("M") else None), target="GiB") # base-2 indexed
        else:
            twrite(f"[WARNING] jobid={self.jobid}: Unrecognized resource type in {tres_str} (either alloctres={self.alloctres} or reqtres={self.reqtres}): {k}={v}")

    return result


# TODO: how to handle cases where the job uses part of multiple nodes?
# TODO: what if a job has multiple nodes in a list of resources that wasn't on a per-job basis? Can this happen?
def resources_to_rgus(*, gpus, gpu_alias, cpus, mem, nodes=1, billing=None, cluster=Utils.get_cluster_type()):
    """Returns the number of RGUs consumed by a job with the given resources.

    If [billing] is provided, we return billing / 1000.

    Otherwise, we go through the nominal RGU calculation. Empirically, this can
    predict billing for many but not all jobs. For example, the documented
    rgus_per_gpu on ComputeCanada for H143s is 6.1, CCDB shows 5.23, and SLURM seems
    to use 5.23 (5.227?) for the computation when this is the relevant variable. And,
    I am not sure how the billing for memory bundles works, because it does not quite
    seem to match the documentation when tested.
    
    Args:
    gpus        -- number of GPUs requested/allocated
    gpu_alias   -- the alias of the GPU type requested/allocated
    cpus        -- number of CPUs requested/allocated
    mem         -- memory in GiB
    nodes       -- number of nodes
    billing     -- the reported biling for the job
    cluster     -- cluster type
    """
    if not billing is None:
        return billing / 1000
    rgus_per_gpu = gpu2info[gpu_alias].rgus_per_gpu if gpu_alias in gpu2info else 1
    gpu_rgus = rgus_per_gpu * gpus
    cpu_rgus = cpus / MachineInfo.cluster2resources_per_rgu[cluster].cpus_per_rgu
    mem_rgus = mem / MachineInfo.cluster2resources_per_rgu[cluster].mem_per_rgu
    rgus = max(gpu_rgus, cpu_rgus, mem_rgus) * nodes
    return rgus

def formatted_date_time(date_time, tz="America/Vancouver"):
    """Returns string-like [date_time] in YYYY-MM-DDTHH:MM format, or "N/A" if it
    can't be parsed. If [tz] is not None, the time is converted to that timezone.
    """
    valid_date_time = False

    # If [start_time] starts with four numbers, these are the year and are removed.
    if len(date_time) > 4 and date_time[0:4].isnumeric():
        date_time = date_time[5:]  # Remove the year and the dash after it
    
    if date_time[0].isnumeric():
        date_time_chars = []
        for c in date_time:
            if c.isnumeric() or c in "-:T":
                date_time_chars.append(c)
            else:
                twrite(f"Unexpected character in start time: {c}")
                break
        date_time = "".join(date_time_chars)
        date_time = date_time[:-3].replace("T", "-") # Exclude seconds
        valid_date_time = True
    elif date_time.startswith("N/A"):
        date_time = "N/A"
    else:
        assert 0, f"Unexpected start time format: {date_time}"

    if valid_date_time and tz:
        from zoneinfo import ZoneInfo
        target_tz = ZoneInfo("America/Vancouver")
        dt = datetime.strptime(date_time, "%m-%d-%H:%M")
        dt = dt.astimezone(target_tz)  # Convert to local timezone
        date_time = dt.strftime("%m-%d-%H:%M")
    return date_time

class UncomputableProperty:
    """Represents something that can't be computed because we've already tried and
    failed in all the ways to do it.
    """
    __slots__ = ()
    def __repr__(self): return self.__class__.__name__
UNCOMPUTABLE = UncomputableProperty()


jobid2squeue_calls = dict()

class JobData:
    """Generic class representing a job. Subclasses are used to construct from various
    ways of getting the data.

    Args:
    jobid                   -- jobid
    jobname                 -- jobname
    account                 -- account of the job
    submit                  -- submit time in YYYY-MM-DDTHH:MM:SS format
    eligible                -- eligible time in YYYY-MM-DDTHH:MM:SS format or something else if job isn't eligible
    start                   -- start time in YYYY-MM-DDTHH:MM:SS format or something else if start time isn't available
    end                     -- end time in YYYY-MM-DDTHH:MM:SS format  or something else if end time isn't available
    state                   -- state of the job
    partition               -- partition(s) of the job
    priority                -- priority of the job
    user                    -- user who submitted the job
    """
    def __init__(self, *, jobid, jobname, account, submit, eligible, start, end, state, partition, priority, user, **kwargs):
        self.jobid = jobid
        self.jobname = jobname
        self.account = account
        self.submit = submit
        self.eligible = eligible
        self.start = start
        self.end = end
        self.state = state
        self.partition = partition
        self.priority = priority
        self.user = user

        # Set the cached property value directly in the instance's __dict__
        kwargs_that_are_cached_properties = {k: v for k,v in kwargs.items() if isinstance(getattr(self.__class__, k, None), cached_property)}
        for k,v in kwargs_that_are_cached_properties.items():
            self.__dict__[k] = copy.deepcopy(v)  

    def __repr__(self): return self.__str__()

    def meta_str(self):
        """Returns a string representation of the job's metadata."""
        return f"jobid={self.jobid}, uid={self.uid}, jobname={self.jobname}, state={self.state}, user={self.user}, partition={self.partition}"
    def timestamps_str(self):
        """Returns a string representation of the job's timestamps."""
        return f"submit={self.submit}, eligible={self.eligible}, start={self.start}, end={self.end}"
    def duration_str(self):
        """Returns a string representation of the job's durations."""
        return f"queue_time={self.queue_time}, run_time={self.run_time}, remaining_time={self.remaining_time}, time_limit_seconds={self.time_limit_seconds}"
    def resources_str(self):
        return f"nodes={self.nodes}, gpus={self.gpus}, gpu_alias={self.gpu_alias}, cpus={self.cpus}, mem={self.mem}, nodes={self.nodes}, rgus={self.rgus}"
    def usage_str(self):
        return f"rgu_days_used={self.rgu_days_used}, rgu_days_left={self.rgu_days_left}, rgu_days={self.rgu_days}"

    def __str__(self): return f"{self.__class__.__name__}({self.meta_str()}, {self.timestamps_str()}, {self.duration_str()}, {self.resources_str()}, {self.usage_str()})"

    ##################################################################################
    # Attributes surrounding the trackable resources used by the job
    ##################################################################################
    @cached_property
    def gpus_per_node(self): return self.resources.get("gpus", 0)
    @cached_property
    def gpus(self): return self.gpus_per_node * self.nodes
    @cached_property
    def gpu_alias(self): return gpu_name2alias.get(self.resources.get("gpu_name"), self.resources.get("gpu_name"))
    @cached_property
    def cpus(self): return self.resources.get("cpus", 1)
    @cached_property
    def mem(self): return self.resources.get("mem", 0) # In GiB
    @cached_property
    def nodes(self): return self.resources.get("nodes", 1)
    ##################################################################################
    ##################################################################################
    ##################################################################################




    ##################################################################################
    # Attributes surrounding resource consumption in RGUs. Many of these are dynamic,
    # while others are RATES of usage.
    ##################################################################################
    @cached_property
    def rgus_computed(self):
        """Number of RGUs allocated by the job. Considers the number GPUs,CPUs,and RAM
        requested resources (reqtres), for jobs that never/haven't run, and the
        allocated resources for those which have run (alloctres).
        """
        return resources_to_rgus(gpus=self.gpus_per_node, gpu_alias=self.gpu_alias,
            cpus=self.cpus, mem=self.mem, nodes=self.nodes, billing=None)

    @cached_property
    def rgus_billing(self):
        """Number of RGUs allocated by the job. The billing of a job divided by 1000
        seems to be this?
        """
        return self.billing / 1000

    @cached_property
    def rgus(self): return self.rgus_computed

    @property
    def rgu_days_used(self):
        """Returns the number of RGU-days used by the job. For completed jobs, the
        number that was actually consumed. For others, assumes the job starts and uses
        all of the allocated time.
        """
        return self.rgus * max(0, self.run_time / (60 * 60 * 24))

    @property
    def rgu_days_left(self):
        """Returns the number of RGU-days left for the job. For completed jobs, this is
        zero. For others, assumes the job starts and uses all of the allocated time.
        """
        return self.rgus * max(0, self.remaining_time / (60 * 60 * 24))

    @cached_property
    def rgu_days(self):
        """Returns the theoretical maximum RGU usage of the job, assuming it runs for
        its full allocated time.
        """
        return self.rgus * self.time_limit_seconds / (60 * 60 * 24)
    ##################################################################################
    ##################################################################################
    ##################################################################################

    ##################################################################################
    # Attributes surrounding durations of job's time whatnot. These are dynamic!
    ##################################################################################
    @property
    def queue_time(self):
        """Returns the queue time in seconds. Jobs that are ineligible have zero. Jobs
        that are curently queuing have their queue time so far returned.
        """
        if not self.eligible or not self.eligible[0].isnumeric():
            return 0
        else:
            queue_end = datetime.strptime(self.start, "%Y-%m-%dT%H:%M:%S") if self.start[0].isnumeric() else datetime.now()
            eligible = datetime.strptime(self.eligible, "%Y-%m-%dT%H:%M:%S")
            return (queue_end - eligible).total_seconds()

    @property
    def run_time(self):
        """Returns the run time in seconds. Jobs that have not started have zero."""
        if self.state in ["RUNNING", "COMPLETING"]:
            run_end = datetime.now()
        elif self.state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"] and self.end and self.end[0].isnumeric():
            run_end = datetime.strptime(self.end, "%Y-%m-%dT%H:%M:%S")
        else:
            run_end = None
        
        if self.start[0].isnumeric() and run_end:
            start = datetime.strptime(self.start, "%Y-%m-%dT%H:%M:%S")
            return (run_end - start).total_seconds()
        else:
            return 0

    @property
    def remaining_time(self):
        """Returns the time left for the job in seconds. Jobs that have yet to start
        have their fill time limit remaining, while those which have ended have zero
        remaining time regardless of how much time they used.
        """
        if self.state in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            return 0
        else:
            return self.time_limit_seconds - self.run_time

    @cached_property
    def eligible_time_dt(self):
        """Returns the eligible time as a datetime object, or None if it can't be
        parsed.
        """
        if self.eligible and self.eligible[0].isnumeric():
            return datetime.strptime(self.eligible, "%Y-%m-%dT%H:%M:%S")
        else:
            return None

    @property
    def status(self):
        """Returns a single word status for the job, maybe different from [state].

        See https://slurm.schedmd.com/job_state_codes.html
        """
        state_first_word = self.state.split()[0] if self.state else ""
        if state_first_word in ("BF", "BOOT_FAIL"):
            return "BOOT_FAIL"
        elif state_first_word in ("CA", "CANCELLED"):
            return "CANCELLED"
        elif state_first_word in ("CD", "COMPLETED"):
            return "COMPLETED"
        elif state_first_word in ("CF", "CONFIGURING"):
            return "CONFIGURING"
        elif state_first_word in ("CG", "COMPLETING"):
            return "COMPLETING"
        elif state_first_word in ("DL", "DEADLINE"):
            return "DEADLINE"
        elif state_first_word in ("F", "FAILED"):
            return "FAILED"
        elif state_first_word in ("NF", "NODE_FAIL"):
            return "NODE_FAIL"
        elif state_first_word in ("OOM", "OUT_OF_MEMORY"):
            return "OUT_OF_MEMORY"
        elif state_first_word in ("PD", "PENDING"):
            return "PENDING"
        elif state_first_word in ("PR", "PREEMPTED"):
            return "PREEMPTED"
        elif state_first_word in ("R", "RUNNING"):
            return "RUNNING"
        elif state_first_word in ("RD", "RESV_DEL_HOLD"):
            return "RESV_DEL_HOLD"
        elif state_first_word in ("RF", "REQUEUE_FED"):
            return "REQUEUE_FED"
        elif state_first_word in ("RH", "REQUEUE_HOLD"):
            return "REQUEUE_HOLD"
        elif state_first_word in ("RQ", "REQUEUED"):
            return "REQUEUED"
        elif state_first_word in ("RS", "RESIZING"):
            return "RESIZING"
        elif state_first_word in ("RV", "REVOKED"):
            return "REVOKED"
        elif state_first_word in ("S", "SUSPENDED"):
            return "SUSPENDED"
        elif state_first_word in ("SE", "SPECIAL_EXIT"):
            return "SPECIAL_EXIT"
        elif state_first_word in ("SI", "SIGNALING"):
            return "SIGNALING"
        elif state_first_word in ("SO", "STAGE_OUT"):
            return "STAGE_OUT"
        elif state_first_word in ("ST", "STOPPED"):
            return "STOPPED"
        elif state_first_word in ("TO", "TIMEOUT"):
            return "TIMEOUT"
        else:
            return state_first_word

    ##################################################################################
    ##################################################################################
    ##################################################################################


    ##################################################################################
    # Attributes computed from knowing the job's comment. Subclasses should figure out
    # how to set the comment attribute.
    ##################################################################################
    @cached_property
    def comment_dict(self):
        """Returns the job's comment as a dictionary if it can be parsed as JSON, or
        an empty dictionary otherwise.
        """
        _ = self.comment # Precompute the comment
        if isinstance(self.comment, str) and "{" in self.comment:
            c = self.comment.strip().replace("'", "\"")  # Replace single quotes with double quotes
            try:
                result = json.loads(c)
                return result if isinstance(result, dict) else json.loads(result) # Sometimes a double-load is needed.
            except json.JSONDecodeError as e:
                twrite(f"[WARNING] Failed to parse comment for jobid={self.jobid}: {self.comment}. Error: {e}")
            except Exception as e:
                twrite(f"[WARNING] Unexpected error while parsing comment for jobid={self.jobid}: {self.comment}. Error: {e}")
        return dict()

    @cached_property
    def uid(self): return self.comment_dict.get("uid", None)

    @cached_property
    def exp_name_full_path(self):
        """Returns the full path to the experiment folder indicated by 'exp_name' in
        the job's comment dictionary, or None if 'exp_name' is not present.
        """
        if "exp_name" in self.comment_dict:
            exp_name = self.comment_dict["exp_name"]
        elif "exp_name_trunc" in self.comment_dict and "uid" in self.comment_dict:
            exp_name = f"{self.comment_dict['exp_name_trunc']}*{self.comment_dict['uid']}*"
        else:
            return None

        found_exp_folders = UtilsBase.flatten([glob.glob(osp.join(s, exp_name)) for s in args.exp_search_dirs])
        if len(found_exp_folders) == 0:
            return None
        elif len(found_exp_folders) > 1:
            twrite(f"Multiple experiment folders found for exp_name={exp_name} with search_dirs={args.exp_search_dirs}: {found_exp_folders}")
            return None
        else:
            return found_exp_folders[0]

    # Not cached since dynamic
    @property
    def latest_checkpoint(self):
        """Returns the latest checkpoint file saved in the experiment folder indicated
        by 'exp_name' in the job's comment dictionary, 'exp_name' is there.
        """
        def checkpoint_is_valid(c):
            """Returns whether checkpoint [c] is valid for being considered the latest."""
            if any([c.endswith(ext) for ext in UserConfig.checkpoint_extensions]):
                if "" in UserConfig.checkpoint_prefixes and c[0].isdigit():
                    return True
                elif any([c.startswith(p) for p in UserConfig.checkpoint_prefixes if not p == ""]):
                    return True
                else:
                    return False
            else:
                return False
        
        if self.exp_name_full_path is None:
            return "no name found"
        else:
            checkpoints = [c for c in os.listdir(self.exp_name_full_path) if checkpoint_is_valid(c)]
            if len(checkpoints) == 0:
                return "no checkpoints"
            else:
                # Sort checkpoints by their modification time
                checkpoints = sorted(checkpoints, key=lambda c: osp.getmtime(osp.join(self.exp_name_full_path, c)), reverse=True)
                return checkpoints[0]

    # Not cached since dynamic
    @property
    def heartbeat(self):
        """Returns the latest heartbeat time saved in the experiment folder indicated
        by 'exp_name' in the job's comment dictionary, or a string indicating why it
        can't be found.
        """
        if self.exp_name_full_path is None:
            return "no name found"
        elif not osp.exists(self.exp_name_full_path):
            return "not found"
        elif not osp.exists(osp.join(self.exp_name_full_path, "heartbeat.txt")):
            return "no heartbeat"
        else:
            heartbeat_file = osp.join(self.exp_name_full_path, "heartbeat.txt")
            heartbeat = UtilsBase.load_file_lite(fpath=heartbeat_file).strip().split()
            heartbeat = f"{heartbeat[0]}T{heartbeat[1]}" # Matches a SLURM date-time format even though it came from Python
            return formatted_date_time(heartbeat, tz=None)
        
    ##################################################################################
    ##################################################################################
    ##################################################################################

    @staticmethod
    def from_sacct_line(line, delimiter="____", format_field_list=DEFAULT_SACCT_FORMAT_NAME2FIELD.values()):
        """Creates a JobDataSacct object from a line of sacct output."""
        line = UtilsBase.strip_right(line, delimiter)
        fields = line.split(delimiter)
        if not len(fields) == len(format_field_list):
            raise ValueError(f"Expected {len(format_field_list)} fields, got {len(fields)}: line={line}, fields={fields}")
        field_dict = dict(zip(format_field_list, fields))
        return JobDataSacct(**field_dict)

    @staticmethod
    def from_squeue_line(line, delimiter="____", format_name2field=DEFAULT_SQUEUE_FORMAT_NAME2FIELD):
        """Creates a JobDataSacct object from a line of squeue output."""
        fields = line.split(delimiter)
        property2value = dict(zip(format_name2field.keys(), fields))
        return JobDataSqueue(**property2value)


class JobDataSqueue(JobData):
    """Class representing a job with data sourced from squeue.

    Args:
    All arguments for superclass.
    stderr                  -- stderr file or N/A if it isn't used
    stdout                  -- stdout file or N/A if it isn't used
    tres_per_task           -- requested/allocated resources per task
    tres_per_node           -- requested/allocated resources per node
    tres_per_job            -- requested/allocated resources per job
    num_tasks               -- number of tasks
    host                    -- host(s) of the job
    time_left               -- time left in HH:MM:SS format or something else if it isn't available
    time_limit              -- time limit in HH:MM:SS format or something else if it isn't available
    nodes                   -- number of nodes
    reason                  -- reason for the job's state
    exclude                 -- excluded nodes
    dependency              -- job dependencies
    """
    def __init__(self, *,
        # Superclass arguments
        jobid, jobname, account, submit, eligible, start, end, state, partition, priority, user,
        # JobDataSqueue-specific arguments
        stderr=None, stdout=None, tres_per_task=None, tres_per_node=None, tres_per_job=None,
        num_tasks=1, comment="", host=None, time_left=None, time_limit=None, nodes=None,
        reason=None, exclude=None, dependency=None):
        super(JobDataSqueue, self).__init__(jobid=jobid, jobname=jobname, account=account, submit=submit,
            eligible=eligible, start=start, end=end,
            state=state, partition=partition,
            priority=priority, user=user)
        self.stderr = stderr
        self.stdout = stdout
        self.tres_per_task = tres_per_task
        self.tres_per_node = tres_per_node
        self.tres_per_job = tres_per_job
        self.num_tasks = int(num_tasks)
        self.comment = comment
        self.host = host
        self.time_left = time_left
        self.time_limit = time_limit
        self.nodes = int(nodes) if nodes and nodes.isnumeric() else 1
        self.reason = reason
        self.exclude = exclude
        self.dependency = dependency


        # Property describing whether the job can be found in squeue. Note that
        # technically this would be dynamic.
        self.in_squeue = True

    def __str__(self): return super(JobDataSqueue, self).__str__()

    @cached_property
    def time_limit_seconds(self):
        return UtilsBase.time_to_seconds(self.time_limit) if self.time_limit and self.time_limit[0].isnumeric() else 0

    @cached_property
    def resources(self):
        """Returns a dictionary of all resources allocated/requested."""
        per_job_resources = parse_resources(self.tres_per_job)
        per_node_resources = parse_resources(self.tres_per_node)
        per_node_resources = {k: v * self.nodes for k,v in per_node_resources.items()}
        per_task_resources = parse_resources(self.tres_per_task)
        per_task_resources = {k: v * self.num_tasks * self.nodes for k,v in per_task_resources.items()}

        result = per_job_resources | per_node_resources | per_task_resources

        # Check that all sources either don't specify a resource type or all specify
        # the same value for it
        for k in result.keys():
            values = [d[k] for d in [per_job_resources, per_node_resources, per_task_resources] if k in d]
            if len(set(values)) > 1:
                twrite(f"[WARNING] jobid={self.jobid}: Inconsistent resource counts for {k}: {values}. Using the maximum value.")
                result[k] = max(values)

        return result

    # For SQUEUE jobs, we need to ask sacct for the billing
    @cached_property
    def billing(self):
        """Returns the billing for the job by querying sacct. If no sacct data is
        found (very weird!), returns 0.
        """
        sacct_jobs = get_sacct_data(jobids=[self.jobid], users=[self.user])
        sacct_jobs = [sj for sj in sacct_jobs if sj.submit == self.submit]
        if len(sacct_jobs) == 0:
            twrite(f"[WARNING] jobid={self.jobid}: No sacct data found for this job. Can't compute billing.")
            return 0
        else:
            return sacct_jobs[0].billing


class JobDataSacct(JobData):
    """Class representing a job with data sourced from sacct.
    
    Args:
    All arguments for superclass.

    alloctres               -- allocated resources
    reqtres                 -- requested resources
    elapsedraw              -- elapsed time in SECONDS
    exitcode                -- exit code
    failednode              -- failed node
    timelimitraw            -- time limit in MINUTES
    submitline              -- command used to submit the job
    """
    def __init__(self, *,
        # Superclass arguments
        jobid, jobname, account, submit, eligible, start, end, state, partition, priority, user,
        # JobDataSacct-specific arguments
        alloctres=None, reqtres=None, elapsedraw=None, exitcode=None, failednode=None,
        timelimitraw=None, submitline=None, comment_from_submitline=None, comment=None, in_squeue=None):
        super(JobDataSacct, self).__init__(jobid=jobid, jobname=jobname, account=account, submit=submit, eligible=eligible,
            start=start, end=end, state=state, partition=partition,
            priority=priority, user=user)
        self.alloctres = alloctres
        self.reqtres = reqtres
        self.elapsedraw = int(elapsedraw)
        self.exitcode = exitcode
        self.failednode = failednode
        self.timelimitraw = int(timelimitraw)
        self.submitline = submitline

        if not comment_from_submitline is None:
            self.comment_from_submitline = comment_from_submitline
        if not comment is None:
            self.comment = comment
        if not in_squeue is None:
            self.in_squeue = in_squeue

    @cached_property
    def in_squeue(self):
        """Returns whether the job can be found in squeue. Note that this is dynamic.
        
        I haven't verified if the heuristic is totally perfect, but it's better than
        spamming squeue.
        """
        if self.status in ["COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"]:
            return False
        else:
            twrite(f"[INFO] jobid={self.jobid}: job think's it's in squeue with state={self.state}              status={self.status}")
            return True

    def __str__(self): return super(JobDataSacct, self).__str__()

    @cached_property
    def time_limit_seconds(self): return self.timelimitraw * 60

    # Semantically I think this is right, but empirically it doesn't *quite* agree
    # with computing this by subtracting the start time from the current time. It's
    # easiest to just have one way of doing this though.
    # @cached_property
    # def run_time(self):
    #     assert self.elapsedraw == int(super(JobDataSacct, self).run_time), f"elapsedraw={self.elapsedraw} != run_time={super(JobDataSacct, self).run_time}"
    #     return self.elapsedraw

    @cached_property
    def resources(self):
        """Returns a dictionary of all resources allocated/requested."""
        tres_str = self.alloctres if self.alloctres else self.reqtres
        return parse_resources(tres_str)
    
    # For SACCT jobs, we get the billing from the resources dictionary.
    @cached_property
    def billing(self): return self.resources["billing"]

    @cached_property
    def comment_from_submitline(self):
        """Tries to return the job's comment string from the job's submitline. If the
        submitline is not a SLURM script or the comment is not set, returns None.
        """
        submitline_parts = self.submitline.split()
        submitline_slurm_script = [sp for sp in submitline_parts if any(sp.endswith(ext) for ext in [".sh", ".slurm", ".sbatch"])]
        submitline_slurm_script = [sp for sp in submitline_slurm_script if osp.exists(sp)]

        if len(submitline_slurm_script) == 0 or len(submitline_slurm_script) > 1:
            return UNCOMPUTABLE
        else:
            comment_line = subprocess.run(f"grep -E '^#SBATCH --comment=' {submitline_slurm_script[0]}", shell=True, capture_output=True, text=True)
            if comment_line.returncode == 0:
                comment = UtilsBase.strip_left(comment_line.stdout.strip(), "#SBATCH --comment=")
                comment = UtilsBase.strip_left(UtilsBase.strip_right(comment, "\""), "\"")
                comment = UtilsBase.strip_left(UtilsBase.strip_right(comment, "\'"), "\'")
                return comment
            else:
                return UNCOMPUTABLE

    @cached_property
    def comment(self):
        """Returns the job's comment as a string or an UncomputableProperty if it
        can't be computed.
        """
        if self.in_squeue:
            global jobid2squeue_calls
            jobid2squeue_calls[self.jobid] = jobid2squeue_calls.get(self.jobid, 0) + 1
            maybe_squeue_job = get_squeue_data(jobids=[self.jobid], users=[self.user])
            if maybe_squeue_job and maybe_squeue_job[0].submit == self.submit:
                return maybe_squeue_job[0].comment
        if not self.comment_from_submitline is UNCOMPUTABLE:
            return self.comment_from_submitline
        else:
            return UNCOMPUTABLE


def get_sacct_data(*, accounts=[], users=[], jobids=[], starttime="2026-07-01", endtime="now", sacct_args=""):
    """Returns a list of Namespaces describing jobs."""
    accounts_str = get_slurm_accounts_str(*accounts)
    user_str = get_slurm_user_str(*users)
    user_str = " -a " if not user_str else user_str
    jobids_str = get_slurm_jobid_str(*jobids)

    format_field_list = DEFAULT_SACCT_FORMAT_NAME2FIELD.values()
    format_str = "--format=" + ",".join([f"{x}%0" for x in format_field_list])

    delimiter_str = "____"

    cmd = f"sacct -X --noheader --parsable --delimiter={delimiter_str} {accounts_str} {user_str} {jobids_str} --starttime={starttime} --endtime={endtime} {format_str} {sacct_args}"

    # twrite(f"[INFO] Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not result.returncode == 0:
        return []
    result = result.stdout.strip()

    job_datas = []
    for line in result.splitlines():
        try:
            job_data = JobData.from_sacct_line(line, delimiter=delimiter_str, format_field_list=format_field_list)
            job_datas.append(job_data)
        except Exception as e:
            twrite(f"[WARNING] get_sacct_data: Failed to parse line: {line}. Error: {e}")
    
    return job_datas

def get_squeue_data(*, accounts=[], users=[], jobids=[], **squeue_kwargs):
    """Returns a list of Namespaces describing jobs."""
    accounts_str = get_slurm_accounts_str(*accounts)
    user_str = get_slurm_user_str(*users)
    jobids_str = get_slurm_jobid_str(*jobids)

    delimiter_str = "____"
    format_name2field = DEFAULT_SQUEUE_FORMAT_NAME2FIELD
    format_str = f"--Format=\"" + f"{delimiter_str},".join([f"{x}:0" for x in format_name2field.values()]) + f"\""
    cmd = f"squeue --noheader {accounts_str} {user_str} {jobids_str} {format_str}"

    # twrite(f"[INFO] Running command: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if not result.returncode == 0:
        return []

    job_datas = []
    for line in result.stdout.strip().splitlines():
        try:
            job_data = JobDataSqueue.from_squeue_line(line, delimiter=delimiter_str, format_name2field=format_name2field)
            job_datas.append(job_data)
        except Exception as e:
            twrite(f"[WARNING] get_squeue_data: Failed to parse line: {line}. Error: {e}")
            raise e
        
    return job_datas

def get_weekly_usage_target(user=os.getlogin()):
    """Returns the weekly usage target in GPU-days for the given user."""
    if not "usage_targets_csv" in UserConfig.__dict__:
        return None
    elif not osp.exists(UserConfig.usage_targets_csv):
        twrite(f"[WARNING] Usage targets CSV file not found: {UserConfig.usage_targets_csv}")
        return None
    else:
        import csv
        with open(UserConfig.usage_targets_csv, "r") as f:
            reader = csv.DictReader(f)
            username_commitments = list(reader)
            lab_usage_target = sum([float(row["h100_per_day"]) for row in username_commitments])
            user_usage_target =  sum([float(row["h100_per_day"]) for row in username_commitments if row["username"] == user]) # Assume one row

            return dict(lab_usage_target=lab_usage_target, user_usage_target=user_usage_target)

def get_usage_so_far(user=os.getlogin(), accounts=UserConfig.cluster2accounts[Utils.get_cluster_type()], start_dt="last_tuesday"):
    """Returns the usage so far in GPU-days for the given user and accounts.

    Args:
    user                    -- user to query for jobs
    accounts                -- list of accounts to query for jobs
    start_dt                -- start date to query for jobs. One of a harcoded value,
                                a YYYY-MM-DD string, or a datetime object.
    
    """
    job_datas = get_sacct_data(accounts=accounts, starttime=start_dt, endtime="now")

    lab_rgu_days_used = sum([jd.rgu_days_used for jd in job_datas])
    lab_rgu_days_queued = sum([jd.rgu_days for jd in job_datas if jd.status == "PENDING"])
    user_rgu_days_used = sum([jd.rgu_days_used for jd in job_datas if jd.user == user])
    user_rgu_days_queued = sum([jd.rgu_days for jd in job_datas if jd.user == user and jd.status == "PENDING"])

    rgus_per_gpu = gpu2info[cluster2node2config[Utils.get_cluster_type()]["default"].gpu_alias].rgus_per_gpu

    return dict(lab_gpus_used=lab_rgu_days_used / rgus_per_gpu,
        lab_gpus_queued=lab_rgu_days_queued / rgus_per_gpu,
        others_gpus_used=(lab_rgu_days_used - user_rgu_days_used) / rgus_per_gpu,
        others_gpus_queued=(lab_rgu_days_queued - user_rgu_days_queued) / rgus_per_gpu,
        user_gpus_used=user_rgu_days_used / rgus_per_gpu,
        user_gpus_queued=user_rgu_days_queued / rgus_per_gpu)

def get_usage_progress_data(user=os.getlogin(),
    accounts=UserConfig.cluster2accounts[Utils.get_cluster_type()],
    period_start_str="last_tuesday", period_end_str="next_tuesday"):
    """Returns a dictionary of usage progress data for the given user and accounts."""

    def prepare_period(p):
        if p == "last_tuesday":
            vancouver_tz = zoneinfo.ZoneInfo("America/Vancouver")
            today = datetime.now(vancouver_tz).date()
            days_ago = max(1,(today.weekday() - 1) % 7)
            last_tuesday_date = today - timedelta(days=days_ago)
            last_tuesday = datetime(last_tuesday_date.year, last_tuesday_date.month, last_tuesday_date.day, 16, 30, tzinfo=vancouver_tz) 
            last_tuesday = last_tuesday.astimezone(None) # Convert to local timezone
            return last_tuesday.isoformat(timespec="seconds")
        elif p == "next_tuesday":
            vancouver_tz = zoneinfo.ZoneInfo("America/Vancouver")
            today = datetime.now(vancouver_tz).date()
            days_ahead = (1 - today.weekday()) % 7
            next_tuesday_date = today + timedelta(days=days_ahead)
            next_tuesday = datetime(next_tuesday_date.year, next_tuesday_date.month, next_tuesday_date.day, 16, 30, tzinfo=vancouver_tz) 
            next_tuesday = next_tuesday.astimezone(None) # Convert to local timezone
            return next_tuesday.isoformat(timespec="seconds")
        elif isinstance(p, datetime):
            return p.isoformat(timespec="seconds")
        elif isinstance(p, str):
            return p
        else:
            raise ValueError(f"Invalid period: {p}")

    period_start_str = prepare_period(period_start_str)
    period_end_str = prepare_period(period_end_str)

    usage_targets = get_weekly_usage_target(user=user)
    usage_so_far = get_usage_so_far(user=user, accounts=accounts, start_dt=period_start_str)
    usage_so_far = {k: v / 7 for k,v in usage_so_far.items()} # Convert GPU-days to GPU-weeks

    lab_usage_remaining = usage_targets["lab_usage_target"] - usage_so_far["lab_gpus_used"]
    lab_usage_remaining_unqueued = lab_usage_remaining - usage_so_far["lab_gpus_queued"]
    others_usage_remaining = lab_usage_remaining - usage_so_far["user_gpus_used"]
    others_usage_remaining_unqueued = others_usage_remaining - usage_so_far["others_gpus_queued"]
    user_usage_remaining = usage_targets["user_usage_target"] - usage_so_far["user_gpus_used"]
    user_usage_remaining_unqueued = user_usage_remaining - usage_so_far["user_gpus_queued"]

    # We can imagine these stats as a bar that looks like the following.
    # ||<-- others used -->||<-- user used -->||<-- others queued -->||<-- user queued -->||<-- user remaining unqueued -->||<-- others remaining unqueued -->||
    

    return dict(
        lab_usage_target=usage_targets["lab_usage_target"],
        user_usage_target=usage_targets["user_usage_target"],
        user_used = usage_so_far["user_gpus_used"],
        user_queued = usage_so_far["user_gpus_queued"],
        user_remaining_unqueued = user_usage_remaining_unqueued,
        others_used = usage_so_far["others_gpus_used"],
        others_queued = usage_so_far["others_gpus_queued"],
        others_remaining_unqueued = others_usage_remaining_unqueued,
    )



def get_usage_process_data_bar(user=os.getlogin(),
    accounts=UserConfig.cluster2accounts[Utils.get_cluster_type()],
    period_start_str="last_tuesday", period_end_str="next_tuesday"):
    """Returns a string representing the usage progress bar for the given user and accounts."""
    usage_progress_data = get_usage_progress_data(user=user, accounts=accounts, period_start_str=period_start_str, period_end_str=period_end_str)
    cols = max(shutil.get_terminal_size().columns - 10, 10)

    col_keys = ["user_used", "user_queued", "user_remaining_unqueued", "others_used", "others_queued", "others_remaining_unqueued"]
    bar_key2cols = {k: int(cols * (v / usage_progress_data["lab_usage_target"])) for k,v in usage_progress_data.items() if k in col_keys}
    cols = sum(bar_key2cols.values())

    # for c in UtilsBase.color2value:
        # print(UtilsBase.colorize(f"{c}: ========================", c))

    bar_key2color = dict(
        user_used="green",
        user_queued="lightblue",
        user_remaining_unqueued="red",
        others_used="green4",
        others_queued="blue2",
        others_remaining_unqueued="orange"
    )
    bar_key2str = dict(
        user_used=f"{user}: {usage_progress_data['user_used']:.1f} used",
        user_queued=f"{user}: {usage_progress_data['user_queued']:.1f} queued",
        user_remaining_unqueued=f"{user}: {usage_progress_data['user_remaining_unqueued']:.1f} unused",
        others_used=f"others: {usage_progress_data['others_used']:.1f} used",
        others_queued=f"others: {usage_progress_data['others_queued']:.1f} queued",
        others_remaining_unqueued=f"others: {usage_progress_data['others_remaining_unqueued']:.1f} unused"
    )
    bar_key2chars = dict()
    for k,v in bar_key2str.items():
        if v.endswith(" used"):
            char = "="
        elif v.endswith(" queued"):
            char = "="
        elif v.endswith(" unused"):
            char = "="
        else:
            char = "?"

        # Pad [v] with equals signs on both sides to fill up the allocated columns.
        # If there's room, remove one of the equals signs on each side of the string
        if len(v) <= bar_key2cols[k]-3:
            remaining_chars = bar_key2cols[k] - len(v)
            left_pad = remaining_chars // 2
            right_pad = remaining_chars - left_pad
            right_pad, left_pad = min(right_pad, left_pad), min(right_pad, left_pad)
            bar_key2chars[k] = char * left_pad + f"{v}" + char * right_pad
        elif len(v.split()[-1]) <= bar_key2cols[k]-3:
            remaining_chars = bar_key2cols[k] - len(v.split()[-1]) - 2
            left_pad = remaining_chars // 2
            right_pad = remaining_chars - left_pad
            right_pad, left_pad = min(right_pad, left_pad), min(right_pad, left_pad)
            bar_key2chars[k] = char * left_pad + f"{v.split()[-1]}" + char * right_pad
        else:
            bar_key2chars[k] = char * bar_key2cols[k]

        if len(bar_key2chars[k]) >= 2:
            bar_key2chars[k] = "|" + bar_key2chars[k][1:-1] + "|"

    bar_key2chars_colored = {k: UtilsBase.colorize(bar_key2chars[k], bar_key2color[k]) for k in bar_key2chars.keys()}
    return "".join([bar_key2chars_colored[k] for k in col_keys]) + "\n" + ", ".join([UtilsBase.colorize(v, bar_key2color[k]) for k,v in bar_key2str.items()])














def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--squeue", action="store_true", help="Use squeue instead of sacct to get job data. This may be faster but may not have all the information."
    )
    P.add_argument("--sacct", action="store_true", help="Use sacct instead of squeue to get job data. This may be slower but may have more information."
    )
    P.add_argument("-u", "--user", default=[os.getlogin()], nargs="+", type=UtilsBase.comma_separated_list_to_list,
        help="User(s) to query for jobs, default is the current user. 'all' for all users under --accounts")
    P.add_argument("-a", "--accounts", nargs="+", default=UserConfig.cluster2accounts[Utils.get_cluster_type()], type=UtilsBase.comma_separated_list_to_list,
        help="List of accounts to query for the current user. If --all_users is set, queries for all users on these accounts.")

    P.add_argument("--starttime", "--start", default="2026-07-01", type=str,)
    P.add_argument("--endtime", "--end", default="now", type=str,)
    
    args, unparsed_args = P.parse_known_args()

    args.sacct_args = " ".join(unparsed_args)
    return args

if __name__ == "__main__":
    args = get_args()

    print(get_usage_process_data_bar())

    
    # if args.squeue:
    #     job_datas = get_squeue_data(accounts=args.accounts, users=args.user)
    # elif args.sacct:
    #     job_datas = get_sacct_data(accounts=args.accounts, users=args.user, starttime=args.starttime, endtime=args.endtime)
    # else:
    #     pass
    
    # for jd in job_datas:
        # twrite(jd)
        # twrite(f"[INFO] Total squeue calls made", squeue_calls=jobid2squeue_calls.get(jd.jobid, 0), num_jobs=len(job_datas))


    # twrite(max_squeue_calls_per_job=max(jobid2squeue_calls.values()) if jobid2squeue_calls else 0, total_squeue_calls=sum(jobid2squeue_calls.values()), num_jobs=len(job_datas))