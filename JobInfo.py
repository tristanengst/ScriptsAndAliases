"""File containing utilities to get arbitrary job information. A bit like a better
version of squeue.
"""
import argparse
from collections import defaultdict
import glob
from functools import cached_property, lru_cache
import json
import os
import os.path as osp
import subprocess

import MachineInfo
import Utils
from Utils import get_cluster_type, is_cc, is_solar
import UtilsBase
from UtilsBase import twrite
from UserConfig import cluster2accounts

# Basically it's easier to compute this once and have it ready than to have to compute
# it for possibly many different JobInfo objects
node2config_true = MachineInfo.cluster2node2config[get_cluster_type()] if get_cluster_type() in MachineInfo.cluster2node2config else dict()
node2config_true = {n: argparse.Namespace(**c) for n,c in node2config_true.items()}

def as_int(s):
    """Returns string [s] as an integer. Can handle suffixes like G, M, etc, in which
    case the result is returned with the 'base' type; eg. G multiplies by 1024 ** 3.
    """
    s = s.strip()
    numeric = UtilsBase.first_numeric_substring(s)
    result = int(numeric)
    if len(numeric) == len(s):
        return result
    else:
        suffix2mul = dict(T=1024**4, G=1024**3, M=1024**2, K=1024)
        suffix = s[len(numeric):].strip().upper()
        if suffix in suffix2mul:
            return result * suffix2mul[suffix]
        else:
            raise ValueError(f"Unexpected suffix {suffix} in string {s}")

class JobInfo:
    """Class representing all information about a job. I cache a lot of the
    computations used to either get additional or just processed input data, and since
    job state can change, this means that the information could get old. Therefore,
    this should just be to have a snapshot of the job's current state.
    
    Args:
    jobid      -- the jobid of the job
    raw        -- dictionary mapping keys to their raw string values as output by
                    squeue. These are stored with an underscore prefix, allowing us to
                    choose between accessing the attributes before or after processing
    """
    def __init__(self, raw=dict()):
        self.raw = raw
        self.__dict__.update(**{f"_{k}": v for k,v in raw.items()}) # Raw values stored for posterity

        # Many keys require no or just type-casting processing
        int_properties = ["jobid", "nodes", "cpus", "num_tasks", ""]
        str_properties = ["name", "user", "state", "reason", "host"]
        list_properties = ["host"]

        raw_int_key2value = {k: int(v) for k,v in raw.items() if k in int_properties}
        self.__dict__.update(**raw_int_key2value)

        raw_str_key2value = {k: v for k,v in raw.items() if k in str_properties}
        self.__dict__.update(**raw_str_key2value)

    def __str__(self):
        return f"{self.__class__.__name__}(" + ", ".join([f"{k}={v}" for k,v in self.__dict__.items() if not k.startswith("_")]) + ")"
    def __repr__(self):
        return self.__str__()
        
    @cached_property
    def pending(self): return self.state in ["PENDING", "CONFIGURING"]

    @cached_property
    def running(self): return self.state in ["RUNNING", "COMPLETING"]

    @cached_property
    def could_run(self): return self.state in ["PENDING", "CONFIGURING", "RUNNING", "COMPLETING"]

    @cached_property
    def comment(self):
        if "comment" in self.raw and not self.raw["comment"] is None:
            c = self.raw["comment"].strip()
            if c.startswith("{") and c.endswith("}"):
                c = c.replace("'", "\"")  # Replace single quotes with double quotes
                try:
                    return json.loads(c)
                except json.JSONDecodeError:
                    return dict()
            else:
                return dict()
        else:
            return dict()

    @cached_property
    def uid(self): return self.comment.get("uid", None)

    @cached_property
    def req_tres(self):
        """Gets the ReqTRES information as a dictionary."""
        try:
            reqtres = self.get_from_scontrol("ReqTres")
            reqtres_kv = [rq.split("=", maxsplit=1) for rq in reqtres.split(",")] if reqtres else []
            
            # It's possible that gres/gpu will appear twice, once with just the number
            # of GPUs total, and once with the GPU type and total number. We'd rather
            # have the latter, so we select the one with more information by ensuring
            # it comes last in the list and overwrites the previous values when after
            # converting to a dict.
            reqtres_kv = sorted(reqtres_kv, key=lambda kv: len(kv[0]) + len(kv[1]))
            reqtres_k2v = {kv[0]: kv[1] for kv in reqtres_kv}
            return reqtres_k2v
        except KeyError:
            return dict()

    @cached_property
    def alloc_tres(self):
        """Gets the AllocTRES information as a dictionary. PENDING jobs are unlikely
        to have this set, so just return an empty dictionary in that case.
        """
        try:
            alloctres = self.get_from_scontrol("AllocTres")
            if alloctres in ["N/A", "(null)"]:
                return dict()

            alloctres_kv = [aq.split("=", maxsplit=1) for aq in alloctres.split(",")] if alloctres else []
            alloctres_kv = sorted(alloctres_kv, key=lambda kv: len(kv[0]) + len(kv[1]))
            alloctres_k2v = {kv[0]: kv[1] for kv in alloctres_kv}
            return alloctres_k2v
        except KeyError:
            return dict()
    
    @cached_property
    def scontrol_data(self):
        """Returns the entire scontrol show JOB output as a dictionary mapping keys to
        values, and with some post-processing.
        """
        single_line_keys = ["Comment", "StdErr", "StdIn", "StdOut", "WorkDir", "Command"]
        dict_keys = ["AllocTres", "ReqTres", "TresPerNode", "TresPerTask"]
        list_keys = ["NodeList", "ReqNodeList", "ExcNodeList"]

        cmd = f"scontrol show job {self.jobid}"
        output = subprocess.getoutput(cmd)
        output = output.strip()
        output_lines = [o.strip() for o in output.splitlines()]
        
        result = dict()
        for oline in output_lines:
            # These are on one line, but are also files so could be more variable, so
            # we just take the whole line as the value. And, these are the only lines
            # for which we expect to be able to have a space within the value,
            # excepting the JobName key
            if oline.split("=", maxsplit=1)[0] in single_line_keys:
                k, v = oline.split("=", maxsplit=1)
                result[k.strip()] = v.strip()            
            # Otherwise, there could be multiple key-value pairs on the line
            else:
                for o in (o.strip() for o in oline.split()):
                    if "=" in o:
                        k, rest = o.split("=", maxsplit=1)
                        if k in list_keys and rest in ["", "(null)"]:
                            result[k.strip()] = []
                        elif k in list_keys:
                            result[k.strip()] = rest.strip().split(",")
                        elif k in dict_keys and not rest in ["", "(null)"]:
                            rest_split_by_comma = rest.strip().split(",")
                            rest_split_by_comma = [r for r in rest_split_by_comma if "=" in r]
                            # This sort ensures that for the AllocTRES and ReqTRES
                            # keys, we get the gres/gpu key with the GPU type and
                            # number if possible when we convert to a dictionary
                            rest_split_by_comma = sorted(rest_split_by_comma, key=lambda s: len(s))
                            rest_k2v = [r.split("=", maxsplit=1) for r in rest_split_by_comma]
                            rest_k2v = {rk.strip(): rv.strip() for rk,rv in rest_k2v}
                            result[k.strip()] = rest_k2v
                        elif k in dict_keys:
                            result[k.strip()] = dict()
                        else:
                            result[k.strip()] = rest.strip()
                    else:
                        pass

        ##############################################################################
        # POST-PROCESSING
        ##############################################################################
        return result

    @lru_cache(maxsize=None)
    def get_dict_from_scontrol(self, *keys, **kv_keys):
        """Returns a dictionary mapping each key in [keys] to a value obtained
        from scontrol. Keys that aren't found are returned with a None value.

        Args:
        keys    -- list of keys to get information for. Missing keys cause an error
        kv_keys -- like keys but with a default fallback value given if not found
        """
        keys = [k for k in keys if not k in kv_keys]
        keys_k2v = {k: self.scontrol_data[k] for k in keys if k in self.scontrol_data}
        if len(keys_k2v) < len(keys):
            missing_keys = [k for k in keys if not k in keys_k2v]
            twrite(jobid=self.jobid, scontrol_data=self.scontrol_data)
            twrite("AAAA")
            raise KeyError(f"Could not find keys {missing_keys} for job {self.jobid} in scontrol output: {self.scontrol_data}")
        keys_with_fallback_k2v = {k: self.scontrol_data.get(k, v) for k,v in kv_keys.items()}
        return keys_k2v | keys_with_fallback_k2v
    
    def get_from_scontrol(self, *keys, **kv_keys):
        """Like get_dict_from_scontrol but returns a single value if only one key is
        requested, and otherwise returns a dictionary mapping keys to values.
        """
        assert len(keys) + len(kv_keys) == 1, f"Exactly one key must be requested, but got {len(keys)} keys and {len(kv_keys)} kv_keys"
        k = keys[0] if keys else list(kv_keys.keys())[0]
        if k in self.scontrol_data:
            return self.scontrol_data[k]
        elif k in kv_keys:
            return kv_keys[k]
        else:
            raise KeyError(f"Could not find key {k} for job {self.jobid} in scontrol output: {self.scontrol_data}")

    @cached_property
    def gpu_num_type(self):
        """Returns the total number of GPUs the job will allocate."""
        def process_resource_key2val_to_gpu_type_num(resource_key2val):
            gpu_gres = {k: v for k,v in resource_key2val.items() if k in ["gres/gpu", "gres/mig"]}
            if len(gpu_gres) == 0:
                return None, 0
            elif len(gpu_gres) > 1:
                raise NotImplementedError(f"jobid={self.jobid} | resource_key2val={resource_key2val}")
            else:
                gres_gpu = list(gpu_gres.values())[0]
                sep = "=" if "=" in gres_gpu else ":"
                gpu_possible_type_and_num = gres_gpu.split(sep)
                gpu_num = gpu_possible_type_and_num[-1]
                gpu_type = gpu_possible_type_and_num[0] if len(gpu_possible_type_and_num) > 1 else None
                return gpu_type, int(gpu_num)

        def compute_unknown_gpu_type():
            """Try and figure out what the job's actualy GPU type is. Assumes that the
            job does actually request at least one GPU.
            """
            if node2config_true:
                if Utils.is_solar() and self.host:
                    gpu_type = node2config_true.get(self.host, dict()).get("gpu_alias", None)
                    return gpu_type
                elif Utils.is_solar() and not self.host in node2config_true or self.host is None:
                    return "default_gpu"
                elif len({n.gpu_alias for n in node2config_true.values()}) == 1:
                    return list({n.gpu_alias for n in node2config_true.values()})[0]
                elif Utils.get_cluster_type() == "killarney":
                    return "l40s"
            
            for key in ["AllocTRES", "ReqTRES", "TresPerNode", "TresPerTask"]:
                alloc_gres = {k: v for k,v in self.get_dict_from_scontrol(key).items() if k in ["gres/gpu", "gres/mig"]}
                gpu_type, _ = process_resource_key2val_to_gpu_type_num(alloc_gres)
                if not gpu_type is None:
                    return gpu_type

            return "default_gpu"

        gres_gpu_type, gres_gpu_num = None, None
        tres_per_node_gpu_type, tres_per_node_gpu_num = None, None
        tres_per_job_gpu_type, tres_per_job_gpu_num = None, None
        tres_per_task_gpu_type, tres_per_task_gpu_num = None, None

        if not self._gres is None and not self._gres == "N/A":
            gres_processed = self._gres.replace(":", "=") # Replace colons with equals signs for easier parsing
            gres_key2val = [g.strip().split("=", maxsplit=1) for g in gres_processed.split(",")]
            gres_key2val = {k: v for (k,v) in gres_key2val}
            gres_gpu_type, gres_gpu_num = process_resource_key2val_to_gpu_type_num(gres_key2val)
            gres_gpu_num = gres_gpu_num * self.num_full_nodes
            twrite(gres_gpu_type=gres_gpu_type)

        # if not self._tres_per_node is None and not self._tres_per_node == "N/A":
        #     tres_per_node_processed = self._tres_per_node.replace(":", "=") # Replace colons with equals signs for easier parsing
        #     tres_per_node_key2val = [g.strip().split("=", maxsplit=1) for g in tres_per_node_processed.split(",")]
        #     tres_per_node_key2val = {k: v for (k,v) in tres_per_node_key2val}
        #     tres_per_node_gpu_type, tres_per_node_gpu_num = process_resource_key2val_to_gpu_type_num(tres_per_node_key2val)
        #     tres_per_node_gpu_num = tres_per_node_gpu_num * self.num_full_nodes

        # if not self._tres_per_job is None and not self._tres_per_job == "N/A":
        #     tres_per_job_processed = self._tres_per_job.replace(":", "=") # Replace colons with equals signs for easier parsing
        #     tres_per_job_key2val = [g.strip().split("=", maxsplit=1) for g in tres_per_job_processed.split(",")]
        #     tres_per_job_key2val = {k: v for (k,v) in tres_per_job_key2val}
        #     tres_per_job_gpu_type, tres_per_job_gpu_num = process_resource_key2val_to_gpu_type_num(tres_per_job_key2val)

        # if not self._tres_per_task is None and not self._tres_per_task == "N/A":
        #     tres_per_task_processed = self._tres_per_task.replace(":", "=") # Replace colons with equals signs for easier parsing
        #     tres_per_task_key2val = [g.strip().split("=", maxsplit=1) for g in tres_per_task_processed.split(",")]
        #     tres_per_task_key2val = {k: v for (k,v) in tres_per_task_key2val}
        #     tres_per_task_gpu_type, tres_per_task_gpu_num = process_resource_key2val_to_gpu_type_num(tres_per_task_key2val)
        #     tres_per_task_gpu_num = tres_per_task_gpu_num * self.num_full_nodes * self.num_tasks

        ##############################################################################
        # Now compute the actual GPU number
        ##############################################################################
        gpu_nums = set([gres_gpu_num, tres_per_node_gpu_num, tres_per_job_gpu_num, tres_per_task_gpu_num])
        gpu_nums = [n for n in gpu_nums if n]
        if len(gpu_nums) == 1:
            gpu_num = int(gpu_nums[0])
        elif len(gpu_nums) > 1:
            raise NotImplementedError(f"jobid={self.jobid} | gpu_nums={gpu_nums} | gres_gpu_num={gres_gpu_num}, tres_per_node_gpu_num={tres_per_node_gpu_num}, tres_per_job_gpu_num={tres_per_job_gpu_num}, tres_per_task_gpu_num={tres_per_task_gpu_num}")
        else:
            return 0, None

        ##############################################################################
        # Now compute the actual GPU type
        ##############################################################################
        twrite(jobid=self.jobid, gpu_types=[gres_gpu_type, tres_per_node_gpu_type, tres_per_job_gpu_type, tres_per_task_gpu_type])

        gpu_types = set([gres_gpu_type, tres_per_node_gpu_type, tres_per_job_gpu_type, tres_per_task_gpu_type])
        gpu_types = [t for t in gpu_types if t]
        if len(gpu_types) == 1:
            return int(gpu_num), gpu_types[0]
        elif len(gpu_types) > 1:
            raise NotImplementedError(f"jobid={self.jobid} | gpu_types={gpu_types} | gres_gpu_type={gres_gpu_type}, tres_per_node_gpu_type={tres_per_node_gpu_type}, tres_per_job_gpu_type={tres_per_job_gpu_type}, tres_per_task_gpu_type={tres_per_task_gpu_type}")
        else:
            gpu_type = compute_unknown_gpu_type()
            return int(gpu_num), gpu_type
    
    @cached_property
    def num_tasks(self):
        """Returns the number of tasks the job will allocate."""
        return int(self.get_from_scontrol(NumTasks=1))

    @cached_property
    def gpu_num(self): return self.gpu_num_type[0]

    @cached_property
    def gpu_type(self): return self.gpu_num_type[1]

    @cached_property
    def mem(self):
        """Returns the amount of memory the job will allocate (per node) in G
        (power-of-two). NOTE: this is better/more accurately sourced from scontrol.
        """
        req_mem = self.get_from_scontrol(ReqTRES=dict()).get("mem", 0)
        alloc_mem = self.get_from_scontrol(AllocTRES=dict()).get("mem", 0)
        mem = max(req_mem, alloc_mem)
        return as_int(mem) / (1024 ** 3)

    @cached_property
    def num_full_nodes(self):
        """Returns the number of full nodes the job will allocate. This is better/more
        accurately sourced from scontrol.
        """
        # This could maybe be smarter, but probably fine for now
        return self.nodes

    @cached_property
    def cpus(self):
        """Returns the number of CPUs the job will allocate."""



    @cached_property
    def node_frac(self):
        """Returns the (improper) fraction of nodes the job is trying to allocate."""
        









default_key_list = ["jobid", "user", "state", "start_time", "time_left", "time_limit", "gres",
        "nodes", "name", "reason", "account", "partition", "host", "exclude",
        "comment", "submit_time", "eligible_time", "stderr", "stdout", "uid",
        "partition", "dependency", "tres_per_task", "tres_per_node", "tres_per_job", "num_tasks"]
default_key2sq_format = dict()

def get_jobid2jobinfo(user=None, account=None, partition=None, verbose=False, keys=default_key_list, key2sq_format=default_key2sq_format):
    """Gets all jobs for [user] and [account] and returns a dictionary mapping jobid to JobInfo.
    
    Args:
    --- THESE ALLOW FILTERING OUT JOBS TO GET INFO FOR ---------------------------
    NOTE: It's better to filter minimally.
    ------------------------------------------------------------------------------
    user --             -- if None, get jobs for all users, if a list, get jobs
                            for all users in the list, if a string, get jobs for
                            that user
    account             -- if None, get jobs for all accounts, if a list, get jobs
                            for all accounts in the list, if a string, get jobs
                            for that account
    partition           -- If None, get info for all partitions. If a
                            comma-separated string or list, interpret as union of
                            partitions to get info for. If otherwise a string,
                            interpret as a single partition to get info for
    
    
    verbose             -- whether to print the squeue command and its output
    keys                -- which keys to include in the output. See
    key2sq_format
    """
    def find_required_keys(k, keys=[]):
        """Recursively finds all required keys for key [k]."""
        req_keys = key2required_keys[k]
        extra = [find_required_keys(kr) for kr in req_keys if not kr == k and not kr in keys]
        extra = UtilsBase.flatten(extra) if extra else extra
        return req_keys + extra

    ##################################################################################
    # Process arguments and handle recursion as needed
    ##################################################################################
    partition = None if partition is None else (partition.split(",") if isinstance(partition, str) else partition)

    # Process [account] to the correct list
    if account == "avail_accounts":
        account = cluster2info[get_cluster_type()].accounts
    elif isinstance(account, list) and "avail_accounts" in account:
        account = [a for a in account if not a == "avail_accounts"] + cluster2info[get_cluster_type()].accounts
    else:
        account = account

    # If either [account] or [user] is a list, we need to get the result recursively
    if isinstance(account, list):
        account2job_infos = {a: get_jobid2jobinfo(user=user, account=a, partition=partition, verbose=verbose, keys=keys, key2sq_format=key2sq_format) for a in account}
        return {j: info for a in account for j,info in account2job_infos[a].items()}
    elif account == "avail_accounts":
        account = cluster2info[get_cluster_type()].accounts

    if isinstance(user, list):
        user2job_infos = {u: get_jobid2jobinfo(user=u, account=account, partition=partition, verbose=verbose, keys=keys, key2sq_format=key2sq_format) for u in user}
        return {j: info for u in user for j,info in user2job_infos[u].items()}
    elif user == "cur_user":
        user = os.getenv("USER")
    ##################################################################################
    ##################################################################################
    ##################################################################################

    ##################################################################################
    # Compute the information we need to get information for for each requested key
    ##################################################################################
    
    # Computing values for some keys requires other keys to get included
    key2required_keys = defaultdict(list, dict(
        uid=["comment"],
    ))

    extra_keys = list(set(UtilsBase.flatten([find_required_keys(k, keys=keys) for k in keys])))
    keys += extra_keys

    user_str = "-u $USER" if user else ""
    account_str = f"-A {account}" if account else ""
    partition_str = f"-p {','.join(partition)}" if partition else ""
    # sep = "  |_||_|||_|||||  "
    sep = ","

    key2sq_format_o = dict(
        jobid=f"%i",
        user=f"%u",
        state=f"%T",
        start_time=f"%S",
        time_left=f"%L",
        time_limit=f"%l",
        gres=f"%b",
        nodes=f"%D",
        name=f"%j",
        reason=f"%r",
        account=f"%a",
        partition=f"%P",
        host=f"%N",
        exclude=f"%x",
        comment=f"%k",
        dependency=f"%E",
    )

    key2sq_format_O = dict(
        jobid="JOBID:10",
        submit_time="SubmitTime:64",
        eligible_time="EligibleTime:64",
        stderr="StdErr:1024",
        stdout="StdOut:1024",
        tres_per_task="tres-per-task:64", # Can work as a fallback for gres
        tres_per_node="tres-per-node:64", # Can work as a fallback for gres
        tres_per_job="tres-per-job:64", # Can work as a fallback for gres
        priority="PriorityLong:16",
    )

    key2sq_format_o = {k: v for k,v in key2sq_format_o.items() if k in keys} | {k: v for k,v in key2sq_format.items() if not ":" in v}
    key2sq_format_O = {k: v for k,v in key2sq_format_O.items() if k in keys} | {k: v for k,v in key2sq_format.items() if ":" in v}

    ##################################################################################
    ##################################################################################
    ##################################################################################

    ##################################################################################
    # Now, get the information for lowercase-o keys
    ##################################################################################
    sq_format_str = sep.join(key2sq_format_o.values())
    sq_cmd = f"squeue {user_str} {account_str} {partition_str} -h -o \"{sq_format_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()

    if verbose:
        print(f"Running command: {sq_cmd}")
        print(f"Output:\n{sq}")
    
    if sq == "":
        twrite(f"[INFO] No jobs found for cur_user={cur_user}, account={account}", cur_user=cur_user, account=account, verbose=verbose)
        return dict()

    jobs = sq.split("\n")
    jobs = [j.strip().split(sep) for j in jobs]
    infos = [dict(list(zip(key2sq_format_o.keys(), j))) for j in jobs]
    job2info = {info["jobid"]: info for info in infos}
    # job2info = {j: info | dict(comment=try_parse_comment(info["comment"])) for j,info in job2info.items()} if "comment" in keys else job2info
    # job2info = {j: info | dict(uid=info["comment"].get("uid", None)) for j,info in job2info.items()} if "uid" in keys else job2info

    # Possible early exit if no -O formatting keys are needed
    # if len(key2sq_format_O) == 0 or list(key2sq_format_O.keys()) == ["jobid"]:
    #     result = {j: argparse.Namespace(**info) for j,info in job2info.items()}
    
    sep = "  |_||_|||_|||||  "
    sq_key_str = sep.join(key2sq_format_O.values())
    sq_cmd = f"squeue {user_str} {account_str} {partition_str} -h -O \"{sq_key_str}\""
    sq = subprocess.getoutput(sq_cmd).strip()
    if verbose:
        print(f"\n\n\nRunning command: {sq_cmd}")
        print(f"Output:\n{sq}")

    jobs = sq.split("\n")
    jobs = [j.strip().split(sep) for j in jobs]
    jobs = [[j1.strip() for j1 in j] for j in jobs]
    infos = [dict(list(zip(key2sq_format_O.keys(), j))) for j in jobs]
    job2info_ = {info["jobid"]: info for info in infos}
    job2info = {j: info1 | job2info_[j] for j,info1 in job2info.items() if j in job2info_}
    return {j: JobInfo(info) for j,info in job2info.items()}


def expand_nodes(nodes_str):
    """Returns a sorted list of node names from [nodes_str]. Example:
        
    'rack02-10,rack03-[13-14]' -> ['rack02-10', 'rack03-13', 'rack03-14']
    """
    if isinstance(nodes_str, list):
        return [expand_nodes(n) for n in nodes_str]
    elif isinstance(nodes_str, str):
        command = f"scontrol show nodes {nodes_str} | grep 'NodeName='"
        output = subprocess.getoutput(command)
        nodes = [ln.split("NodeName=")[1].split()[0] for ln in output.splitlines() if "NodeName=" in ln]
        return sorted(nodes)
    else:
        raise ValueError(f"Unexpected type for nodes_str: {type(nodes_str)}")





# def get_jobid2jobinfo_sacct(user=None, account=None, partition=None, verbose=False, keys=default_key_list, key2sq_format=default_key2sq_format):
#     """Gets all jobs for [user] and [account] and returns a dictionary mapping jobid
#     to JobInfo. Uses sacct instead of squeue, so can get information about completed
#     jobs, but is also much slower.
#     """


#     # Hack because Solar's SLURM is different? As of 2026-02-06, maybe no longer needed?
#     if is_solar() and False:
#         jobs = [j.strip().split()[:len(key2sq_format_O.values())] for j in jobs]
#     elif is_solar():

        

    
    
    
#     job2info = {j: argparse.Namespace(**info) for j,info in job2info.items()}
#     return job2info

    
        
        
        
#         job2info = get_slurm_status(cur_user=(user is None), account=account, verbose=verbose, keys=keys, key2sq_format=key2sq_format)
#         return {j: JobInfo(j, **vars(info)) for j,info in job2info.items()}




# def get_slurm_status(cur_user=False, account=None, verbose=False,
#     keys=["jobid", "user", "state", "start_time", "time_left", "time_limit", "gres",
#         "nodes", "name", "reason", "account", "partition", "host", "exclude",
#         "comment", "submit_time", "eligible_time", "stderr", "stdout", "uid",
#         "partition", "dependency", "tres_per_task", "tres_per_node", "tres_per_job", "num_tasks"],
#     key2sq_format=dict()
#     ):
#     """Returns a dictionary describing the entire state what's running. Strings are
#     not processed or reformatted in any way.

#     Args:
#     cur_user        -- whether to get results for only the current user's jobs
#     account         -- if not None, only show jobs for this account
#     keys            -- which keys to include in the output. See
#                     https://slurm.schedmd.com/squeue.html for details, but some
#                     'uid' is custom
#     key2sq_format   -- Additional key2sq_format entries to use with -o formatting in
#                         addition to those used in [keys]. Keys whose values have
#                         colons in them (eg. 'SubmitTime:64') get -O formatting

#     Notes:
#     - Some keys must generally be included for sensible results, eg. 'jobid'
#     - Some keys require others. In particular, 'uid' requires 'comment'
#     """
#     import json
#     def try_parse_comment(c):
#         """Tries to parse the comment [c]."""
#         c = c.strip()
#         if c.startswith("{") and c.endswith("}"):
#             c = c.replace("'", "\"")  # Replace single quotes with double quotes
#             try:
#                 return json.loads(c) | dict(comment=c)
#             except json.JSONDecodeError:
#                 return dict(comment=c)
#         else:
#             return dict(comment=c)

#     # On ComputeCanada, get jobs by account
#     if account is None and is_cc():
#         accounts = cluster2info[get_cluster_type()].accounts
#         result = dict()
#         for account in accounts:
#             result |= get_slurm_status(cur_user=cur_user, account=account)
#         return result

    

    

    

    
    
    

    























# def is_apex():
#     """Returns whether a user is a member of APEX lab. This can confer extra
#     functionality that can't be put online. Hardcoded.
#     """
#     import grp
#     groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]

#     if is_cc() and any([g in ["rrg-keli", "def-keli", "aip-keli"] for g in groups]):
#         return True
#     elif is_solar() and any([g in ["cs-apex"] for g in groups]):
#         return True
#     elif (osp.exists("/NAS/ScriptsAndAliasesExtra")
#         or osp.exists("/localscratch/ScriptsAndAliasesExtra")
#         or osp.exists(osp.expanduser("~/ScriptsAndAliasesExtra"))):
#         return True
#     else:
#         return False

# def get_cluster_type():
#     """Returns a string for special host types, or None if they are not recognized."""
#     h = os.uname()[1]
#     if "nibi" in h:
#         return "nibi"
#     elif h.startswith("vulcan") or h.startswith("rack"):
#         return "vulcan"
#     elif h.startswith("klogin") or h.startswith("kn"):
#         return "killarney"
#     elif h.startswith("tamia") or h.startswith("tc") or h.startswith("tg"):
#         return "tamia"
#     elif "trig" in h or "trillium" in h:
#         return "trillium"
#     elif h.startswith("fc") or h.startswith("login"):
#         return "fir"
#     elif h.startswith("rorqual") or h.startswith("rq") or h.startswith("rg") or h.startswith("rl"):
#         return "rorqual"
#     elif h.startswith("narval") or h.startswith("ng"):
#         return "narval"
#     elif h.startswith("cedar") or h.startswith("cdr"):
#         return "cedar"
#     elif h.startswith("beluga") or h.startswith("bg"):
#         return "beluga"
#     elif h.startswith("gra-") or h.startswith("gra") or h.startswith("gr"):
#         return "graham"
#     elif h.startswith("cs-star") or h.startswith("cs-v"):
#         return "solar"
#     else:
#         h_ = "-".join(h.split("-")[:2])
#         return os.environ.get("CLUSTER_TYPE", "cs-apex")

# def is_solar(): return get_cluster_type() in ["solar"]
# def is_cc(): return get_cluster_type() in ["nibi", "narval", "cedar", "beluga", "graham", "rorqual", "trillium", "fir", "vulcan", "killarney", "tamia"]
# def is_slurm(): return is_solar() or is_cc()
# def is_workstation(): return not is_solar() and not is_cc()

# cluster2info = dict(
#     nibi=dict(accounts=cluster2accounts.get("nibi",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     trillium=dict(accounts=cluster2accounts.get("trillium",[]), conf_file=None, partitions_startswith=["compute", "compute_full_node"]),
#     fir=dict(accounts=cluster2accounts.get("fir",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     rorqual=dict(accounts=cluster2accounts.get("rorqual",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     narval=dict(accounts=cluster2accounts.get("narval",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     cedar=dict(accounts=cluster2accounts.get("cedar",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     beluga=dict(accounts=cluster2accounts.get("beluga",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     vulcan=dict(accounts=cluster2accounts.get("vulcan",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     killarney=dict(accounts=cluster2accounts.get("killarney",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase", "interac"]),
#     tamia=dict(accounts=cluster2accounts.get("tamia",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["gpubase"]),
#     solar=dict(accounts=cluster2accounts.get("solar",[]), conf_file="/etc/slurm/slurm.conf", partitions_startswith=["cs-gpu-research"]),
# )
# cluster2info = {k: argparse.Namespace(**v) for k,v in cluster2info.items()}



# def jobid2info_to_uid2jobids(job2info=None):
#     """Converts a job2info dictionary to a uid2jobids dictionary."""
#     from collections import defaultdict
#     job2info = job2info if job2info else get_slurm_status(cur_user=True)

#     uid2jobids = defaultdict(list)
#     for jobid,info in job2info.items():
#         if not info.uid is None:
#             uid2jobids[info.uid].append(jobid)
#     return dict(uid2jobids)

# def get_project_dir(def_or_rrg=None):
#     """Returns a path to the user's group's project directory."""
#     if is_solar():
#         return UserConfig.cluster2project_dirs["solar"][0]
#     elif is_cc():
#         import grp
#         groups = [grp.getgrgid(g).gr_name for g in os.getgroups()]
#         project_dirs = [g for g in groups if osp.exists(osp.join("/project", g))]
#         if len(project_dirs) == 0:
#             raise ValueError(f"Could not find a project directory in the user's groups: {groups}")
#         elif len(project_dirs) == 1:
#             return project_dirs[0]
#         elif len(project_dirs) > 1 and not def_or_rrg is None:
#             project_dirs = [p for p in project_dirs if p.startswith(def_or_rrg)]
#         else:
#             raise NotImplementedError()
#     else:
#         raise NotImplementedError()
        
if __name__ == "__main__":
    jobid2info = get_jobid2jobinfo(user="cur_user", verbose=True)
    twrite(jobid2info)
    twrite(**{j: info.gpu_num_type for j,info in jobid2info.items()})




