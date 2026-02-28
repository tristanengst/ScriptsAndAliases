"""Uses the sinfo command to get information about the current cluster."""
import argparse
from collections import defaultdict
import copy
from functools import cached_property, cache
import os
import subprocess
import sys

import MachineInfo
import Utils
import UtilsBase
from UtilsBase import twrite, colorize

@cache
def get_all_partitions(*, verbose=False): return get_partitions_from_sinfo(verbose=verbose)

def get_partitions_from_sinfo(nodes=[], verbose=False):
    """Returns a list of Partition objects representing the partitions on the cluster."""
    key2si_format_O = dict(name="PartitionName:32",
        time_limit="Time:32",
        priority_tier="PriorityTier:32",
        priority_job_factor="PriorityJobFactor:32", )

    si_format_strs = ",".join(key2si_format_O.values())
    si_cmd = f"sinfo -O '{si_format_strs}'"
    if verbose:
        print(f"[INFO] Running command: {si_cmd}")

    si = subprocess.getoutput(si_cmd)
    if verbose:
        print(f"[INFO] Got sinfo output:\n{si}")
    si_lines = [line for line in si.strip().split("\n") if len(line.strip()) > 0]
    si_lines = si_lines[1:] # Remove the header line
    si_lines = [line.split() for line in si_lines]
    si_lines = [[ll.strip() for ll in line] for line in si_lines]
    line_dicts = [dict(zip(key2si_format_O.keys(), line)) for line in si_lines]

    partitions = list()
    for ld in line_dicts:
        partition = Partition(name=ld["name"],
            time_limit=ld["time_limit"],
            priority_tier=ld["priority_tier"], 
            priority_job_factor=ld["priority_job_factor"],
            nodes=nodes,)
        partitions.append(partition)
    partitions = [p for p in partitions if not p.name.startswith("cpu")]
    return partitions

@cache
def time2partition_names():
	"""Returns a dict mapping time limits to lists of partitions for which that time
	limit is the maximum. Cached because it's unlikely to change. The time limits are
	in seconds. Note that interactive partitions are included by default!
	"""
	if Utils.is_solar():
		return {21600: ["cs-gpu-research-debug"], 604800: ["cs-gpu-research"]}
	elif Utils.is_cc():
		partitions = get_partitions_from_sinfo()
		time2partitions = defaultdict(list)
		for p in partitions:
			time2partitions[p.seconds].append(p.name)
		return dict(time2partitions)
	else:
		raise NotImplementedError()

class Node:
    def __init__(self, name, partitions, states,
        mem_total, mem_alloc,
        cpus_total, cpus_alloc,
        gres_total, gres_alloc):
        
        self.name = name
        self.partitions = partitions
        self.states = states

        idle_states = ["IDLE", "DYNAMIC_NORM"]

        if any([s in self.states for s in ["INVALID_REG", "DOWN", "DRAIN", "NOT_RESPONDING", "MAINT", "FAIL", "POWER_SAVE", "REBOOT", "RESERVED"]]):
            self.state = "down"

        # I have a hunch that GPUs which list 'shard' as a resource that could be
        # allocated are empirically not actually allocatable by real jobs
        elif (Utils.get_cluster_type() == "vulcan"
            and "IDLE" in self.states
            and all([s in idle_states for s in self.states])
            and not any([g.endswith("_shard") for g in gres_total.keys()])):
            self.state = "free"
        elif (not Utils.get_cluster_type() == "vulcan"
            and "IDLE" in self.states
            and all([s in idle_states for s in self.states])):
            self.state = "free"
        else:
            self.state = "avail"

        self.mem_total = int(mem_total)
        self.mem_alloc = int(mem_alloc)
        self.cpus_alloc = int(cpus_alloc)
        self.cpus_total = int(cpus_total)

        # Sort from most to least VRAM
        self.gres_total = {k: gres_total[k] for k in sorted(gres_total.keys(), key=lambda g: MachineInfo.gpu2vram[g], reverse=True)}
        self.gres_alloc = {k: gres_total[k] for k in sorted(gres_alloc.keys(), key=lambda g: MachineInfo.gpu2vram[g], reverse=True)}
        
        # If the cluster is Vulcan, count apparently free L40s GPUs as used if there
        # are also L40s_shard GPUs that could be allocated on them
        if Utils.get_cluster_type() == "vulcan" and "l40s_shard" in self.gres_total:
            self.gres_alloc["l40s"] = self.gres_total["l40s"]

        self.set_resource_availability()
        
        # Note that this is different from [gres_alloc] since it accounts for CPU and memory usage
        self.gres_used = {gpu: self.gres_total[gpu] - self.gres_avail[gpu] for gpu in self.gres_total.keys()}
        self.gres_state = ",".join([f"{gpu}={self.gres_avail[gpu]}/{self.gres_total[gpu]}" for gpu in self.gres_total.keys()])
        self.mem_state = f"{int(self.mem_total - self.mem_alloc)}/{int(self.mem_total)}GB"
        self.cpus_state = f"{self.cpus_total - self.cpus_alloc}/{self.cpus_total}"

    def set_resource_availability(self):
        """Determines how used each resource (GPU type) on the node is. Resources are
        considered in order from most to least VRAM.
        """
        self.gres_avail = dict()

        avail_cpus = self.cpus_total - self.cpus_alloc
        avail_memory = self.mem_total - self.mem_alloc

        for gpu_type,total in self.gres_total.items():
            lookup_name = self.name if Utils.is_solar() else gpu_type

            alloc = self.gres_alloc[gpu_type] if gpu_type in self.gres_alloc else 0
            avail_gpus = total - alloc

            req_cpus_per_gpu = MachineInfo.cluster2node2config[Utils.get_cluster_type()][lookup_name]["cpus_per_gpu"]
            req_mem_per_gpu = MachineInfo.cluster2node2config[Utils.get_cluster_type()][lookup_name]["mem_per_gpu"]

            cpu_limited_gpus = int(avail_cpus // req_cpus_per_gpu)
            mem_limited_gpus = int(avail_memory // req_mem_per_gpu)

            free_gpus = min(avail_gpus, cpu_limited_gpus, mem_limited_gpus)
            self.gres_avail[gpu_type] = free_gpus

            avail_cpus = avail_cpus - (free_gpus * req_cpus_per_gpu)
            avail_memory = avail_memory - (free_gpus * req_mem_per_gpu)
    
    def __str__(self):
        kv = dict(
            name=self.name,
            partitions=",".join(self.partitions),
            states=",".join(self.states),
            state=self.state,
            mem_state=self.mem_state,
            cpus_state=self.cpus_state,
            gres_state=self.gres_state

        )

        def format_dict(d): return "{" + ", ".join([f"{k}={v}" for k,v in d.items()]) + "}"
        kv = {k: (format_dict(v) if isinstance(v, dict) else v) for k,v in kv.items()}
        kv_str = ", ".join([f"{k}={v}" for k,v in kv.items()])

        return f"{self.__class__.__name__}({kv_str})"

    def __repr__(self): return self.__str__()

def tres_to_gres_used(tres, node_name="default"):
    """Returns a dictionary of used GRES objects from an AllocTRES or CfgTRES string.
    Note that this only populates GPU-related information, as it's easier to parse CPU
    and RAM information from other scontrol show node fields.
    """
    tres = UtilsBase.strip_left(tres, "AllocTRES=")
    tres = UtilsBase.strip_left(tres, "CfgTRES=")
    gres2count = dict()
    for resource in tres.split(","):
        if resource.startswith("gres/gpu") or resource.startswith("gres/shard"):
            is_shard = "gres/shard" in resource

            # Both of these can appear and should be removed
            resource = UtilsBase.strip_left(resource, "gres/gpu:")
            resource = UtilsBase.strip_left(resource, "gres/gpu")
            resource = UtilsBase.strip_left(resource, "gres/shard:")
            resource = UtilsBase.strip_left(resource, "gres/shard")

            if resource.startswith("=") and node_name and Utils.is_solar():
                count = int(resource[1:])
                gpu_name = MachineInfo.cluster2node2config[Utils.get_cluster_type()][node_name]["gpu_name"]
            elif resource.startswith("=") and Utils.get_cluster_type() == "trillium":
                count = int(resource[1:])
                gpu_name = MachineInfo.cluster2node2config[Utils.get_cluster_type()]["default"]["gpu_name"]
            elif not resource.startswith("=") and not Utils.is_solar():
                gpu_name, count = resource.split("=")
                count = int(count)
            else:
                continue

            gpu_name = f"{gpu_name}_shard" if is_shard else gpu_name
            gpu_type = MachineInfo.gpu_name2alias[gpu_name]
            gres2count[gpu_type] = count
            
    return gres2count

def get_nodes_from_scontrol_data(args):
    """Returns a list of Node objects representing the nodes on the cluster,
    using scontrol instead of sinfo.
    """
    cmd = "scontrol show nodes"
    data = subprocess.getoutput(cmd)
    node_infos = data.split("NodeName=")[1:]

    nodes = list()

    for n in node_infos:
        node_name = n.split()[0]
        entries = UtilsBase.flatten([e.split() for e in n.splitlines()])
        node_info = dict(name=node_name, partitions=[])

        for e in entries:
            if e.startswith("Partitions="):
                node_info["partitions"] = e.split("=")[1].split(",")
            elif e.startswith("State="):
                node_info["states"] = e.split("=")[1].split()[0].split("+")
            elif e.startswith("RealMemory="):
                node_info["mem_total"] = int(e.split("=")[1]) / 1024
            elif e.startswith("AllocMem="):
                node_info["mem_alloc"] = int(e.split("=")[1]) / 1024
            elif e.startswith("CPUAlloc="):
                node_info["cpus_alloc"] = int(e.split("=")[1])
            elif e.startswith("CPUTot="):
                node_info["cpus_total"] = int(e.split("=")[1])
            elif e.startswith("CfgTRES="):
                node_info["gres_total"] = tres_to_gres_used(e, node_name=node_name)
            elif e.startswith("AllocTRES"):
                node_info["gres_alloc"] = tres_to_gres_used(e, node_name=node_name)

        try:
            node = Node(**node_info)
        except Exception as e:
            print(entries)
            raise e
        nodes.append(node)
    
    nodes = [n for n in nodes if len(n.gres_total) > 0]
    return nodes




class Partition:
    """
    Args:
    name        -- the name of the partition
    time_limit  -- the time limit of the partition as a hhhHmmM string
    seconds     -- the time limit of the partition in seconds (int)
    hours       -- the time limit of the partition in hours (int)

    priority_tier       -- the priority tier of the partition
    priority_job_factor -- the priority job factor of the partition

    nodes       -- list of Node objects in this partition, likely set after construction
    children    -- list of partition names for child partitions
    """
    def __init__(self, *, name,
        time_limit: str | None = None, seconds: int | None = None, hours: int | None = None, 
        priority_tier=1, priority_job_factor=1,
        nodes=[], children=None):
        ##############################################################################
        # Ensure consistency between [time_limit], [seconds], and [hours]
        ##############################################################################
        time_limit_ = UtilsBase.time_to_seconds(time_limit) if time_limit else None
        seconds_ = UtilsBase.time_to_seconds(seconds) if seconds else None
        hours_ = UtilsBase.time_to_seconds(hours * 3600) if hours else None
        seconds_values = [t for t in [hours_, seconds_, time_limit_] if t is not None]
        if len(seconds_values) == 0:
            raise ValueError(f"At least one of time_limit, seconds, or hours must be provided, got: time_limit={time_limit}, seconds={seconds}, hours={hours}")
        elif len(set(seconds_values)) > 1:
            raise ValueError(f"time_limit, seconds, and hours must all represent the same time duration. Got time_limit={time_limit} seconds={seconds} hours={hours}")
        else:
            self.seconds = seconds_values[0]
            self.hours = UtilsBase.time_to_hours(self.seconds)
            self.time_limit = UtilsBase.time_to_pretty_str(self.seconds)
        ##############################################################################
        ##############################################################################
        ##############################################################################
        self.name = name

        self.priority_tier = UtilsBase.try_make_number(priority_tier)
        self.priority_job_factor = UtilsBase.try_make_number(priority_job_factor)

        self.nodes = [n for n in nodes if self.name in n.partitions]
        self.children = children if children else []

    def __str__(self):
        node_str = "" if not self.nodes else f", num_nodes={len(self.nodes)}"
        return f"Partition(name={self.name}, time={self.time_limit} prio_tier={self.priority_tier}, prio_factor={self.priority_job_factor},{node_str})"

    def __repr__(self): return self.__str__()

    @cached_property
    def max_total_gpus(self):
        """Returns the maximum number of total GPUs available on any node on the partition."""
        return max([sum(n.gres_total.values()) for n in self.nodes]) if len(self.nodes) > 0 else 0 

    @staticmethod
    def partition_with_children(*, partition, children):
        """Returns a copy of [partition] with its children set to [children]."""
        return Partition(name=partition.name,
            time_limit=partition.time_limit,
            nodes=partition.nodes,
            children=children)

    @staticmethod
    def contains(p1, p2):
        """Partition [p1] contains partition [p2] if the marginal benefit of queueing
        on [p2] and [p1] over just [p1] is zero. This is a bit heuristic.
        """
        return p1.seconds >= p2.seconds and all([n in p1.nodes for n in p2.nodes])

    @staticmethod
    def equivalent(p1, p2):
        """Returns if partitions [p1] and [p2] are equivalent, meaning that they
        contain each other."""
        return (Partition.contains(p1, p2) and Partition.contains(p2, p1)) or (p1.name == p2.name)

    @staticmethod
    def partition_better_than(p1, p2):
        """Returns if partition [p1] is better than partition [p2]."""
        result = (not Partition.equivalent(p1, p2)
            and Partition.contains(p1, p2)
            and (p1.priority_tier > p2.priority_tier or (
                p1.priority_tier == p2.priority_tier
                and p1.priority_job_factor >= p2.priority_job_factor))
            )
        return result

    @staticmethod
    def filter_partitions(partitions):
        """Returns a minimal list of partitions such that no partition can be removed
        without making it worse to queue on the remainder. Interactive partitions do
        not knock out other partitions however.
        """
        return [p1 for p1 in partitions if not any([Partition.partition_better_than(p2, p1) for p2 in partitions if not "interac" in p2.name])]
        

def partitions_nodes_to_resource(*, partitions, nodes, verbose=False, node2config):
    node2config = node2config if node2config else MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    time2resource2free = defaultdict(lambda: defaultdict(float))            # Unused resources
    time2resource2full_node_free = defaultdict(lambda: defaultdict(set))  # Unused resources
    
    time2resource2avail = defaultdict(lambda: defaultdict(float))           # Used resources
    time2resource2full_node_avail = defaultdict(lambda: defaultdict(set)) # Used resources
    
    time2resource2total = defaultdict(lambda: defaultdict(float))           # Total resources (includes offline nodes)
    time2resource2total_nodes = defaultdict(lambda: defaultdict(set))       # Total nodes (includes offline nodes)
    
    # Nodes seen so far in counting time/resource info. Otherwise we can accidentally double-count nodes in multiple partitions.
    time2resource2seen_nodes = defaultdict(lambda: defaultdict(set))        
    for p in partitions:
        for n in p.nodes:
            for gpu in n.gres_total.keys():

                if n.name in time2resource2seen_nodes[p.time_limit][gpu]:
                    continue

                resource_identifier = n.name if Utils.is_solar() else gpu
                if n.state == "down":
                    time2resource2total[p.time_limit][gpu] += n.gres_total[gpu]
                    time2resource2total_nodes[p.time_limit][gpu].add(n.name)
                elif n.state == "avail":
                    time2resource2total[p.time_limit][gpu] += n.gres_total[gpu]
                    time2resource2total_nodes[p.time_limit][gpu].add(n.name)
                    time2resource2avail[p.time_limit][gpu] += n.gres_total[gpu]
                    time2resource2full_node_avail[p.time_limit][gpu].add(n.name)
                    time2resource2free[p.time_limit][gpu] += n.gres_avail[gpu]
                elif n.state == "free":
                    time2resource2total[p.time_limit][gpu] += n.gres_total[gpu]
                    time2resource2total_nodes[p.time_limit][gpu].add(n.name)
                    time2resource2avail[p.time_limit][gpu] += n.gres_total[gpu]
                    time2resource2full_node_free[p.time_limit][gpu].add(n.name)
                    time2resource2free[p.time_limit][gpu] += n.gres_avail[gpu]
                    time2resource2full_node_free[p.time_limit][gpu].add(n.name)
                else:
                    pass

                time2resource2seen_nodes[p.time_limit][gpu].add(n.name)               

    if verbose:
        print("\n[INFO] Resource availability by partition time:")
        for time in sorted(time2resource2total.keys()):
            print(f"Max time: {UtilsBase.time_to_pretty_str(time*3600)}")
            for resource in sorted(time2resource2total[time].keys(), key=lambda g: MachineInfo.gpu2vram[g]):
                total = time2resource2total[time][resource]
                avail = time2resource2avail[time][resource]
                free = time2resource2free[time][resource]
                full_node_avail = len(time2resource2full_node_avail[time][resource])
                full_node_free = len(time2resource2full_node_free[time][resource])
                print(f"\tResource: {resource}:\t\ttotal={total}, avail={avail}, free={free}, full_node_avail={full_node_avail}, full_node_free={full_node_free}")

                if resource == "h100":
                    nodes_with_any_free = [n for n in nodes if n.gres_avail.get("h100", 0) > 0 and not n.state == "down"]
                    max_free = sum([n.gres_avail.get("h100", 0) for n in nodes_with_any_free])
                    print(f"\t\tNodes with any free H100s: {[n.name for n in nodes_with_any_free]} len={len(nodes_with_any_free)} max_free={max_free}")

            
        

    return argparse.Namespace(time2resource2free=time2resource2free,
        time2resource2full_node_free=time2resource2full_node_free,
        time2resource2avail=time2resource2avail,
        time2resource2full_node_avail=time2resource2full_node_avail,
        time2resource2total=time2resource2total,
        time2resource2total_nodes=time2resource2total_nodes)

def format_cluster_state(resource_states, nodes=None, printable_free_nodes=4):
    """Returns a string describing what resources are available on the cluster.

    The vertical axis lists different times, while the horizontal axis lists different
    GPU types. These are sorted so that the highest-VRAM GPUs available for the
    longest time are on the bottom right.

    Each entry is of the form (free/avail/total).
    For GPU types that can run full nodes (gpu_frac >= 1.0), if there are any full
    nodes free, at most three are listed next to the resource.

    There is colorization as follows:
    ...
    """
    node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]

    if Utils.is_solar():
        all_times = [max(resource_states.time2resource2total.keys())]
    else:
        # all_times = sorted(resource_states.time2resource2total.keys())
        all_times = sorted(resource_states.time2resource2total.keys(), key=UtilsBase.time_to_seconds)

    all_resources = sorted(set(UtilsBase.flatten([list(resource_states.time2resource2total[time].keys()) for time in all_times])),
        key=lambda g: MachineInfo.gpu2vram[g])
    all_resources = [r for r in all_resources if MachineInfo.gpu2info[r]["good"]]
    
    time2resource2str = defaultdict(lambda: defaultdict(str))
    time2resource2str["time"] = dict(time="\t\ttime") | {r: r for r in all_resources}

    for time in all_times:
        pretty_time = UtilsBase.time_to_pretty_str(time*3600) if isinstance(time, (int, float)) else time
        pretty_time = UtilsBase.strip_right(pretty_time, "00M")  # Remove minutes if zero
        pretty_time = UtilsBase.strip_left(pretty_time, "0")  # Remove minutes if zero
        time2resource2str[time]["time"] = pretty_time

        

        for resource in all_resources:
            total = int(resource_states.time2resource2total[time][resource])
            avail = int(resource_states.time2resource2avail[time][resource])
            free = int(resource_states.time2resource2free[time][resource])

            free_color = "lightblue" if free > 0.1 * avail else ("green" if free > 0 else "red")
            avail_total_color = "orange" if avail > 0.1 * total else "red"

            free = UtilsBase.colorize(f"({free}/", color=free_color)
            avail_total = UtilsBase.colorize(f"{avail}/{total})", color=avail_total_color)
            entry_str = free + avail_total

            if Utils.is_cc() and node2config[resource]["gpu_frac"] >= 1.0 and resource_states.time2resource2full_node_free[time][resource]:
                full_node_free = list(resource_states.time2resource2full_node_free[time][resource])
                num_free_nodes = len(full_node_free)
                full_node_free = full_node_free[:min(len(full_node_free), printable_free_nodes)]
                additional_free_node_str = f", ...{num_free_nodes} total" if num_free_nodes > len(full_node_free) else ""
                full_node_str = " nodes=(" + ",".join(full_node_free) + f"{additional_free_node_str})"
                full_node_str = UtilsBase.colorize(full_node_str, color="lightblue")
            elif Utils.is_solar() and resource_states.time2resource2full_node_free[time][resource]:
                full_node_free = list(resource_states.time2resource2full_node_free[time][resource])
                full_node_free = full_node_free[:min(len(full_node_free), printable_free_nodes)]
                full_node_free = [UtilsBase.strip_left(n, "cs-") for n in full_node_free]
                full_node_str = " nodes=(" + ",".join(full_node_free) + ")"
                full_node_str = UtilsBase.colorize(full_node_str, color="lightblue")
            else:
                full_node_str = ""

            if Utils.is_solar():
                time2resource2str[time][resource] = f"{resource.upper()}={entry_str}{full_node_str}"
            else:
                time2resource2str[time][resource] = f"{pretty_time}-{resource.upper()}={entry_str}{full_node_str}"


    # Ensure that each entry string is padded to the same *visual* length. Since they
    # are colored, we can't just use len().
    col2max_len = defaultdict(lambda: 0)
    for time in time2resource2str.keys():
        for col,resource in enumerate(["time"] + all_resources):
            entry_str = time2resource2str[time][resource]
            col2max_len[col] = max(col2max_len[col], len(UtilsBase.decolorize(entry_str)))

    # Now do the padding. 
    for time in time2resource2str.keys():
        for col,resource in enumerate(["time"] + all_resources):
            entry_str = time2resource2str[time][resource]
            len_colorized = len(entry_str)
            len_decolorized = len(UtilsBase.decolorize(entry_str))
            padding_needed = col2max_len[col] - len_decolorized
            
            entry_str = entry_str + " " * padding_needed
            time2resource2str[time][resource] = entry_str

    s = f"Free/Avail/Total"
    for idx,time in enumerate(all_times):
        new_line_chars = "\t" if idx == 0 else "\n\t\t\t\t"
        s += new_line_chars + "\t".join([f"{time2resource2str[time][resource]}" for resource in all_resources])
    return s

def get_str(printable_free_nodes=4):
    """Returns a string representation of the cluster info."""
    args = get_args(args=[])
    node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    nodes = get_nodes_from_scontrol_data(args)
    partitions = get_partitions_from_sinfo(nodes=nodes)
    cluster_info = Utils.cluster2info[Utils.get_cluster_type()]
    partitions = [p for p in partitions if any([p.name.startswith(pname) for pname in cluster_info.partitions_startswith])]
    resource_states = partitions_nodes_to_resource(partitions=partitions, nodes=nodes, verbose=False, node2config=node2config)
    s = format_cluster_state(resource_states, nodes=nodes, printable_free_nodes=printable_free_nodes)
    return s


def get_args(args=None):
    """Parses command-line arguments for this module."""
    P = argparse.ArgumentParser()
    P.add_argument("-v", "--verbose", action="store_true")
    P.add_argument("-vv", action="store_true")
    P.add_argument("-i", "--interac", action="store_true",)
    P.add_argument("-p", "--partitions", "--partition", nargs="+", default=None,
        help="If set, only consider these partitions.")
    P.add_argument("--gpus", nargs="+", default=["good"])
    P.add_argument("--gpu_counts", nargs="+", default=[1,2,4] if Utils.is_solar() else [1, "all"],
        help="List of GPU counts to consider. 'all' means all available GPUs on a node can be allocated together.")
    P.add_argument("-n", "--nodes", action="store_true",
        help="If set, show node-level information.")
    P.add_argument("--max_nodes_to_show", type=int, default=20 if Utils.is_solar() else (5 if Utils.get_cluster_type() == "nibi" else 2),
        help="Maximum number of nodes to show per resource in the summary.")
    P.add_argument("--aggregate", choices=[0,1], type=int, default=int(Utils.is_solar()),
        help="If set, aggregate resources in a useful way for summarization.")
    P.add_argument("--max_time", type=int, default=24,
        help="If set, only consider nodes with max time less than or equal to this value when summarizing resources.")
    P.add_argument("--show_resources", action="store_true",)
    args = P.parse_args(args)

    args.partitions = UtilsBase.flatten([p.split(",") for p in args.partitions]) if args.partitions else None
    if "good" in args.gpus:
        args.gpus += [g for g in MachineInfo.gpu2info.keys() if MachineInfo.gpu2info[g]["good"]]
    args.gpu_counts = [int(gc) if str(gc).isnumeric() else gc for gc in args.gpu_counts]

    return args

if __name__ == "__main__":
    args = get_args()
    node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    nodes = get_nodes_from_scontrol_data(args)
    for n in nodes:
        print(n)

    # Get partitions from the nodes, and filter them to include only partitions that
    # are documented in cluster info.
    partitions = get_partitions_from_sinfo(nodes=nodes)
    cluster_info = Utils.cluster2info[Utils.get_cluster_type()]
    partitions = [p for p in partitions if any([p.name.startswith(pname) for pname in cluster_info.partitions_startswith])]

    for p in partitions:
        print(f"Partition(name={p.name}, time_limit={p.time_limit}h)")
    
    resource_states = partitions_nodes_to_resource(partitions=partitions, nodes=nodes, verbose=True, node2config=node2config)
    s = format_cluster_state(resource_states)
    print(f"\n[INFO] Cluster resource state:\n{s}")
    