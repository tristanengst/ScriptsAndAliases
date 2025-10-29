"""Uses the sinfo command to get information about the current cluster."""
import argparse
from collections import defaultdict
import copy
import os
import subprocess
import sys

import MachineInfo
import Utils
import UtilsBase
from UtilsBase import twrite, colorize

class Node:
    def __init__(self, name, partitions, states,
        mem_total, mem_alloc,
        cpus_total, cpus_alloc,
        gres_total, gres_alloc):
        
        self.name = name
        self.partitions = partitions
        self.states = states

        if any([s in self.states for s in ["INVALID_REG", "DOWN", "DRAIN", "NOT_RESPONDING"]]):
            self.state = "down"
        elif all([s == "IDLE" for s in self.states]):
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

def tres_to_gres_used(tres, node_name="default"):
    """Returns a dictionary of used GRES objects from an AllocTRES or CfgTRES string.
    Note that this only populates GPU-related information, as it's easier to parse CPU
    and RAM information from other scontrol show node fields.
    """
    tres = UtilsBase.strip_left(tres, "AllocTRES=")
    tres = UtilsBase.strip_left(tres, "CfgTRES=")
    gres2count = dict()
    for resource in tres.split(","):
        if resource.startswith("gres/gpu"):
            # Both of these can appear and should be removed
            resource = UtilsBase.strip_left(resource, "gres/gpu:")
            resource = UtilsBase.strip_left(resource, "gres/gpu")

            if (resource.startswith("=")
                and node_name
                and (Utils.is_solar() or Utils.get_cluster_type() == "trillium")):
                count = int(resource[1:])
                gpu_name = MachineInfo.cluster2node2config[Utils.get_cluster_type()][node_name]["gpu_name"]
            elif not resource.startswith("=") and not (Utils.is_solar() or Utils.get_cluster_type() == "trillium"):
                gpu_name, count = resource.split("=")
                count = int(count)
            else:
                continue

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
        node_info = dict(name=node_name)

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
                node_info["gres_total"] = tres_to_gres_used(e)
            elif e.startswith("AllocTRES"):
                node_info["gres_alloc"] = tres_to_gres_used(e)

        node = Node(**node_info)
        nodes.append(node)
    
    nodes = [n for n in nodes if len(n.gres_total) > 0]
    return nodes

class Partition:
    """
    Args:
    name        -- the name of the partition
    time_limit  -- the time limit of the partition in hours
    nodes       -- list of Node objects in this partition, likely set after construction
    children    -- list of partition names for child partitions
    """
    def __init__(self, name, time_limit, nodes=[], children=None):
        self.name = name
        self.time_limit = time_limit
        self.time = UtilsBase.time_to_hours(time_limit)
        self.nodes = [n for n in nodes if self.name in n.partitions]
        self.children = children if children else []

    def __str__(self):
        node_str = "" if not self.nodes else f", num_nodes={len(self.nodes)}"
        return f"Partition(name={self.name}, time={UtilsBase.time_to_pretty_str(self.time*3600)},{node_str})"

    @staticmethod
    def get_partitions_from_sinfo(nodes=[]):
        """Returns a list of Partition objects representing the partitions on the cluster."""
        key2si_format_O = dict(name="PartitionName:32", time_limit="Time:32",)

        si_format_strs = ",".join(key2si_format_O.values())
        si_cmd = f"sinfo -O '{si_format_strs}'"
        # print(f"[INFO] Running command: {si_cmd}")

        si = subprocess.getoutput(si_cmd)
        # print(f"[INFO] Got sinfo output:\n{si}")
        si_lines = [line for line in si.strip().split("\n") if len(line.strip()) > 0]
        si_lines = si_lines[1:] # Remove the header line
        si_lines = [line.split() for line in si_lines]
        si_lines = [[ll.strip() for ll in line] for line in si_lines]
        line_dicts = [dict(zip(key2si_format_O.keys(), line)) for line in si_lines]

        partitions = list()
        for ld in line_dicts:
            time_limit = UtilsBase.time_to_hours(ld["time_limit"])
            partition = Partition(name=ld["name"], time_limit=time_limit, nodes=nodes)
            partitions.append(partition)
        partitions = [p for p in partitions if not p.name.startswith("cpu")]
        return partitions

    @staticmethod
    def partition_with_children(*, partition, children):
        """Returns a copy of [partition] with its children set to [children]."""
        return Partition(name=partition.name,
            time_limit=partition.time_limit,
            nodes=partition.nodes,
            children=children)

    @staticmethod
    def contains(p1, p2):
        """Returns if every job that could run on partition [p2] can run on [p1]."""
        return p1.time_limit >= p2.time_limit and all([n in p1.nodes for n in p2.nodes])

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
    args = P.parse_args(args)

    args.partitions = UtilsBase.flatten([p.split(",") for p in args.partitions]) if args.partitions else None
    if "good" in args.gpus:
        args.gpus += [g for g in MachineInfo.gpu2info.keys() if MachineInfo.gpu2info[g]["good"]]
    args.gpu_counts = [int(gc) if str(gc).isnumeric() else gc for gc in args.gpu_counts]

    return args

def partitions_nodes_to_resource(*, partitions, nodes, verbose=False, node2config):
    node2config = node2config if node2config else MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    time2resource2free = defaultdict(lambda: defaultdict(float))            # Unused resources
    time2resource2full_node_free = defaultdict(lambda: defaultdict(list))  # Unused resources
    time2resource2avail = defaultdict(lambda: defaultdict(float))           # Used resources
    time2resource2full_node_avail = defaultdict(lambda: defaultdict(list)) # Used resources
    time2resource2total = defaultdict(lambda: defaultdict(float))           # Total resources (includes offline nodes)
    time2resource2total_nodes = defaultdict(lambda: defaultdict(list))       # Total nodes (includes offline nodes)
    for p in partitions:
        for n in p.nodes:
            for gpu in n.gres_total.keys():
                resource_identifier = n.name if Utils.is_solar() else gpu

                time2resource2total[p.time_limit][gpu] += n.gres_avail[gpu]
                time2resource2total_nodes[p.time_limit][gpu].append(n.name)

                if n.state == "free":
                    time2resource2free[p.time_limit][gpu] += n.gres_avail[gpu]
                    if node2config[resource_identifier]["gpu_frac"] >= 1.0:
                        time2resource2full_node_free[p.time_limit][gpu].append(n.name)
                    
                    # Free resources are also available
                    time2resource2avail[p.time_limit][gpu] += n.gres_avail[gpu]
                    time2resource2full_node_avail[p.time_limit][gpu].append(n.name)
                elif n.state == "avail":
                    time2resource2avail[p.time_limit][gpu] += n.gres_avail[gpu]
                    time2resource2full_node_avail[p.time_limit][gpu].append(n.name)
                else:
                    pass # Everything is added to the total already

    if verbose:
        print("\n[INFO] Resource availability by partition time:")
        for time in sorted(time2resource2total.keys()):
            print(f"Max time: {UtilsBase.time_to_pretty_str(time*3600)}")
            for resource in time2resource2total[time].keys():
                total = time2resource2total[time][resource]
                avail = time2resource2avail[time][resource]
                free = time2resource2free[time][resource]
                full_node_avail = len(time2resource2full_node_avail[time][resource])
                full_node_free = len(time2resource2full_node_free[time][resource])
                print(f"\tResource: {resource}:\t\ttotal={total}, avail={avail}, free={free}, full_node_avail={full_node_avail}, full_node_free={full_node_free}")

    return argparse.Namespace(time2resource2free=time2resource2free,
        time2resource2full_node_free=time2resource2full_node_free,
        time2resource2avail=time2resource2avail,
        time2resource2full_node_avail=time2resource2full_node_avail,
        time2resource2total=time2resource2total,
        time2resource2total_nodes=time2resource2total_nodes)

def format_cluster_state(resource_states):
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
    all_times = sorted(resource_states.time2resource2total.keys())
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

            if node2config[resource]["gpu_frac"] >= 1.0 and resource_states.time2resource2full_node_free[time][resource]:
                full_node_free = resource_states.time2resource2full_node_free[time][resource]
                full_node_free = full_node_free[:min(len(full_node_free), 3)]
                full_node_str = " nodes=(" + ",".join(full_node_free) + ")"
                full_node_str = UtilsBase.colorize(full_node_str, color="lightblue")
            else:
                full_node_str = ""
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

def get_str():
    """Returns a string representation of the cluster info."""
    args = get_args(args=None)
    node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    nodes = get_nodes_from_scontrol_data(args)
    partitions = Partition.get_partitions_from_sinfo(nodes=nodes)
    partitions = [p for p in partitions if any([p.name.startswith(pname) for pname in Utils.cluster2info[Utils.get_cluster_type()].partitions_startswith])]
    resource_states = partitions_nodes_to_resource(partitions=partitions, nodes=nodes, verbose=False, node2config=node2config)
    s = format_cluster_state(resource_states)
    return s

if __name__ == "__main__":
    args = get_args()
    node2config = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    nodes = get_nodes_from_scontrol_data(args)
    for n in nodes:
        print(n)

    partitions = Partition.get_partitions_from_sinfo(nodes=nodes)
    partitions = [p for p in partitions if any([p.name.startswith(pname) for pname in Utils.cluster2info[Utils.get_cluster_type()].partitions_startswith])]

    for p in partitions:
        print(f"Partition(name={p.name}, time_limit={p.time_limit}h)")
    
    
    resource_states = partitions_nodes_to_resource(partitions=partitions, nodes=nodes, verbose=True)
    s = format_cluster_state(resource_states)
    print(f"\n[INFO] Cluster resource state:\n{s}")
    