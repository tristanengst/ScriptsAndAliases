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

def resource_infos_to_str(time2resource2info, show_nodes=0, max_nodes_to_show=5, good_gpus_only=False, max_time_to_show=None):
    """Returns a nice string representation of the resource info dictionaries."""
    strs = list()

    times = sorted(time2resource2info.keys(), key=lambda t: UtilsBase.time_to_hours(t))

    time2printed_resources = defaultdict(list)

    for time in times:
        resource2info = time2resource2info[time]
        resources = sorted(resource2info.keys(), key=lambda r: resource2info[r].vram)

        for resource in resources:
            info = resource2info[resource]
            if good_gpus_only and not any([r in MachineInfo.good_gpus for r in UtilsBase.strip_right(resource, "-node").split("/")]):
                # print(f"[DEBUG] Skipping resource {resource} because it's not good GPUs only")
                continue

            if (not max_time_to_show is None
                and info.max_time > UtilsBase.time_to_hours(max_time_to_show * 3600)
                and (any([resource in time2printed_resources[t] for t in times]) if Utils.get_cluster_type() == "fir" else False)):
                continue

            avail_color_scale = ["red"] + (["orange"] * 10) + ["yellow", "green"]
            avail_color = avail_color_scale[min(int((info.avail / info.total) * len(avail_color_scale)), len(avail_color_scale)-1)]
            avail_total_str = colorize(f"{info.avail}/{info.total})", avail_color)

            free_color = "red" if info.free == 0 else ("green" if (info.free / info.avail) <= 0.1 else "lightblue")
            free_str = colorize(f"({info.free}/", free_color)

            time_str = "" if Utils.is_solar() else f"{time}_"

            s = f"{time_str}{resource}={free_str}{avail_total_str}"

            if ((resource.endswith("-node") and show_nodes >= 1 and info.free_nodes)
                or (not resource.endswith("-node") and show_nodes >= 2 and info.free_nodes)):

                free_nodes = info.free_nodes[:max_nodes_to_show]
                free_nodes = [UtilsBase.strip_left(n, "cs-") for n in free_nodes] if Utils.is_solar() else free_nodes
                s += f" ({','.join(free_nodes)})"
                s += "..." if len(info.free_nodes) > max_nodes_to_show else ""

            strs.append(s)
            time2printed_resources[time].append(s)

    to_print = ""
    max_str_len = max([len(UtilsBase.decolorize(s)) + len("\t\t") for s in strs]) if len(strs) > 0 else 0
    print(max_str_len)
    for idx,(time,resource_strs) in enumerate(time2printed_resources.items()):
        to_print += "\n\t\t\t\t" if idx > 0 else ""
        to_print += "".join([r + (" " * (max_str_len - len(UtilsBase.decolorize(r)))) for r in resource_strs])
    
    s = f"Free/Avail/Total:\t{to_print}"
    return s

def aggregate_resource_infos(time2resource2info, args):
    """Returns a different resource info dictionary where resources are aggregated
    in a useful way.
    """
    agg_time2resource2info = defaultdict(lambda: defaultdict(lambda: argparse.Namespace(
        total=0, avail=0, free=0, vram=None, resources_per_node=None, max_time=None, free_nodes=[])))

    all_gpus = UtilsBase.flatten([info.gpu for time,resource2info in time2resource2info.items() for info in resource2info.values()])
    all_gpus = set(all_gpus)
    good_gpus = [g for g in all_gpus if MachineInfo.gpu2info[g]["good"]]
    good_gpus = sorted(good_gpus, key=lambda g: MachineInfo.gpu2vram[g])
    good_gpu_str = "/".join(good_gpus) if len(good_gpus) > 0 else "good-gpu"

    bad_gpus = [g for g in all_gpus if not MachineInfo.gpu2info[g]["good"]]
    bad_gpus = sorted(bad_gpus, key=lambda g: MachineInfo.gpu2vram[g])
    bad_gpu_str = "/".join(bad_gpus) if len(bad_gpus) > 0 else "bad-gpu"

    resourc2agg_resource = dict()
    for time,resource2info in time2resource2info.items():
        for resource,info in resource2info.items():
            resourc2agg_resource[resource] = f"{info.num_gpus}x" + (good_gpu_str if info.gpu in good_gpus else bad_gpu_str)
    
    for time,resource2info in time2resource2info.items():
        agg_resource2infos = defaultdict(list)
        for r,info in resource2info.items():
            agg_resource2infos[resourc2agg_resource[r]].append(info)

        for agg_resource,infos in agg_resource2infos.items():
        
            max_time = min([info.max_time for info in infos])
            vram = min([info.vram for info in infos if info.vram is not None])
            resources_per_node = min([info.resources_per_node for info in infos if info.resources_per_node is not None])
            free_nodes = list(set(UtilsBase.flatten([info.free_nodes for info in infos])))
            total = sum([info.total for info in infos])
            avail = sum([info.avail for info in infos])
            free = sum([info.free for info in infos])

            agg_time2resource2info[time][agg_resource].max_time = max_time
            agg_time2resource2info[time][agg_resource].vram = vram
            agg_time2resource2info[time][agg_resource].resources_per_node = resources_per_node
            agg_time2resource2info[time][agg_resource].free_nodes = free_nodes
            agg_time2resource2info[time][agg_resource].total = total
            agg_time2resource2info[time][agg_resource].avail = avail
            agg_time2resource2info[time][agg_resource].free = free
    return agg_time2resource2info


def node_list_to_resources_info(*, nodes, args):
    """Returns how many instances of each resource can be allocated within the given
    node list, separated by maximum time. If a certain resource could be counted
    towards multiple types of resources (eg. a single GPU could be counted towards
    full nodes or individual GPUs), it is counted towards both types.

    Args:
    node_list   -- list of Node objects
    args        -- Namespace giving [gpu_counts] ....
    """
    time2resource2info = defaultdict(lambda: defaultdict(lambda: argparse.Namespace(
        total=0, avail=0, free=0, num_gpus=None, gpu=None,
        vram=None, resources_per_node=None, max_time=None,
        free_nodes=[])))

    node2info = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
    gpu2info = {gpu: argparse.Namespace(**info) for gpu,info in MachineInfo.gpu2info.items()}

    for node in nodes:
        for gpu_count in args.gpu_counts:
            for gpu in node.gpu2count_total.keys():

                node2info_key = node.name if Utils.is_solar() else gpu

                if ((not gpu in args.gpus and not "/" in gpu)
                    or not gpu2info[gpu].ddp and (gpu_count == "all" or gpu_count > 1)):
                    # print(f"[DEBUG] Skipping GPU {gpu} on node {node.name} for gpu_count={gpu_count} node={node}")
                    continue
                else:
                    total = node.gpu2count_total[gpu]
                    avail = node.gpu2avail[gpu]

                if gpu_count == "all" or gpu_count == -1:
                    gpu_count = MachineInfo.cluster2node2config[Utils.get_cluster_type()][node2info_key]["gpus_per_node"]
                    resource_name = f"{gpu}-node"
                else:
                    resource_name = f"{gpu_count}x{gpu}" if gpu_count > 1 else gpu
                
                total = total // gpu_count
                avail = avail // gpu_count

                time2resource2info[node.max_time_str][resource_name].max_time = node.max_time
                time2resource2info[node.max_time_str][resource_name].gpu = gpu
                time2resource2info[node.max_time_str][resource_name].num_gpus = gpu_count
                time2resource2info[node.max_time_str][resource_name].vram = MachineInfo.gpu2vram[gpu] * gpu_count
                time2resource2info[node.max_time_str][resource_name].resources_per_node = gpu_count // total
                time2resource2info[node.max_time_str][resource_name].total += total
                time2resource2info[node.max_time_str][resource_name].free += avail
                time2resource2info[node.max_time_str][resource_name].avail += 0 if node.state == "down" else total
                time2resource2info[node.max_time_str][resource_name].free_nodes += [node.name] if avail > 0 else []

                # if resource_name.endswith("-node"):
                #     print(f"[DEBUG] Node {node.name} with resource {resource_name}: total={total}, avail={avail}, free_nodes={time2resource2info[node.max_time_str][resource_name].free_nodes}")

    # for t in time2resource2info.keys():
    #     for r in time2resource2info[t].keys():
    #         twrite(f"[DEBUG] Time {t}, resource={r} info={time2resource2info[t][r]}")
    
    return time2resource2info

class Node:
    """Class representing a node on the cluster.
    
    Args:
    name        -- the name of the node
    partition   -- partition the partition associated to the node
    max_time    -- the maximum time limit for jobs on the node (given the current partition)
    state       -- the overall state of the node
    cpu_state   -- the CPU state of the node
    gres        -- the GRES (e.g., GPUs) available on the node
    gres_used   -- the state of the GRES on the node
    free_mem    -- the free memory on the node
    memory      -- the total memory on the node
    partition2max_time -- (optional) dictionary mapping partitions the node can be allocated on to their max times
    """    
    def __init__(self, *, name, partition, max_time, state, cpu_state, gres, gres_used, free_mem, memory,
        partition2max_time=None, correct_parse=True, **kwargs):
        self.init_kwargs = dict(name=name, partition=partition, max_time=max_time, state=state, cpu_state=cpu_state,
            gres=gres, gres_used=gres_used, free_mem=free_mem, memory=memory, partition2max_time=partition2max_time)
        
        self.name = name
        self.partition = partition
        
        self.max_time = UtilsBase.time_to_hours(max_time)
        self.max_time_str = UtilsBase.time_to_pretty_str(self.max_time * 3600)
        self.max_time_str = UtilsBase.strip_right(self.max_time_str, "00M")
        self.max_time_str = UtilsBase.strip_left(self.max_time_str, "0")
        
    
        self.state_ = state
        self.cpu_state_ = cpu_state
        self.gres_ = gres
        self.gres_used_ = gres_used
        self.free_mem_ = free_mem
        self.memory_ = memory
        self.partition2max_time = partition2max_time if partition2max_time else {self.partition: self.max_time}

        self.set_state()
        self.set_avail_resource_fraction()

    def set_state(self):
        """Returns True if the node is available for scheduling."""
        down_strs = ["down", "drain", "fail", "unknown", "maint"]
        if any([ds in self.state_.lower() for ds in down_strs]):
            self.state = "down"
        elif self.state_.lower() == "idle":
            self.state = "idle"
        else:
            self.state = "alloc"

    def set_avail_resource_fraction(self):
        """Returns the fraction of the node that can be used."""
        cpus_used,self.avail_cpus,_,self.total_cpus = [int(x) for x in self.cpu_state_.split("/")]

        self.avail_memory = (int(self.free_mem_) / 1000) if self.free_mem_.isnumeric() else 0
        self.total_memory = (int(self.memory_) / 1000) if self.memory_.isnumeric() else 1

        def parse_gres(gres_str):
            """Parses a gres string into a list of resources and counts. Resources are
            named by their alias in MachineInfo.gpu_name2alias.
            """
            gres_str = gres_str.strip()
            gpu2count = dict()
            for gpu_name in gres_str.split(","):
                if not gpu_name.startswith("gpu:"):
                    continue
                gpu_name = UtilsBase.strip_left(gpu_name, "gpu:")
                gpu_name = gpu_name[:gpu_name.index("(")] if "(" in gpu_name else gpu_name
                gpu_name, count = gpu_name.split(":")
                gpu2count[MachineInfo.gpu_name2alias[gpu_name]] = int(count)
            return gpu2count

        self.gpu2count_used = parse_gres(self.gres_used_)
        self.gpu2count_total = parse_gres(self.gres_)

        # If there is only one kind of GPU on the node, then it's simple to compute
        # how free it is. However, if there are multiple kinds of GPUs, then it might
        # be possible to have more or less of some type be free. What we will do here
        # is to compute the fraction of the node that's free for the highest-VRAM
        # GPU type. If there are resources left over, continue with the next-highest
        # VRAM GPU type, and so on.
        node2info = MachineInfo.cluster2node2config[Utils.get_cluster_type()]
        sorted_gpus = sorted(self.gpu2count_total.keys(), key=lambda gpu: MachineInfo.gpu2vram[gpu], reverse=True)
        avail_cpus_ = self.avail_cpus
        avail_memory_ = self.avail_memory
        self.gpu2avail = dict()
        
        for gpu in sorted_gpus:
            resource_info = node2info[self.name] if Utils.is_solar() else node2info[gpu]
            
            avail_gpus = self.gpu2count_total[gpu] - self.gpu2count_used[gpu]

            cpu_limited_gpus = int(avail_cpus_ // resource_info["cpus_per_gpu"])
            mem_limited_gpus = int(avail_memory_ // resource_info["mem_per_gpu"])
            self.gpu2avail[gpu] = min(avail_gpus, cpu_limited_gpus, mem_limited_gpus)
            avail_cpus_ -= self.gpu2avail[gpu] * resource_info["cpus_per_gpu"]
            avail_memory_ -= self.gpu2avail[gpu] * resource_info["mem_per_gpu"]
        
        # If the node is down, nothing on it is available
        _ = self.set_state()
        if self.state == "down":
            self.avail_cpus = 0
            self.avail_memory = 0
            self.gpu2avail = {k: 0 for k in self.gpu2count_total.keys()}

        
    def __str__(self):    
        kv = dict(name=self.name, partition=self.partition,
            max_time=self.max_time_str,
            state=self.state,
            avail_cpus=self.avail_cpus, total_cpus=self.total_cpus,
            avail_memory=f"{int(self.avail_memory)}GB", total_memory=f"{int(self.total_memory)}GB",
            gpu2avail=self.gpu2avail,
            gpu2count_total=self.gpu2count_total,
            gpu2count_used=self.gpu2count_used)
        
        def format_dict(d): return "{" + ", ".join([f"{k}={v}" for k,v in d.items()]) + "}"
        kv = {k: (format_dict(v) if isinstance(v, dict) else v) for k,v in kv.items()}
        kv_str = ", ".join([f"{k}={v}" for k,v in kv.items()])

        return f"{self.__class__.__name__}({kv_str})"

    @staticmethod
    def sanitize_node_list_across_partitions(node_list):
        """In a list of nodes, a given node could appear many times (ie. because it's
        in multiple partitions). We want to return a list so that each node appears
        only once, with the partition and max time information aggregated into a
        [partition2time] list. These nodes have their partition and maximum time set
        to the largest possible values.
        """
        name2nodes = defaultdict(list)
        for node in node_list:
            name2nodes[node.name].append(node)

        result_nodes = list()
        for name,nodes in name2nodes.items():
            partition2max_time = {node.partition: node.max_time for node in nodes}
            sorted_nodes = sorted(nodes, key=lambda n: n.partition)
            max_time_node = sorted(sorted_nodes, key=lambda n: n.max_time)[-1]
            node_kwargs = copy.deepcopy(max_time_node.init_kwargs) | dict(partition2max_time=partition2max_time)
            result_nodes.append(Node(**node_kwargs))
        return result_nodes

    # @staticmethod
    # def sanitize_node_list_across_gpus(node_list):
    #     """In a list of nodes there could be many GPU types represented. We would like
    #     to compress them, eg. have one GPU type for if the GPU is good or not.

    #     Assumes we have run sanitize_node_list_across_partitions() first.
    #     """
    #     all_gpus = UtilsBase.flatten([list(node.gpu2count_total.keys()) for node in node_list])
    #     all_gpus = set(all_gpus)
    #     good_gpus = [g for g in all_gpus if MachineInfo.gpu2info[g]["good"]]
    #     good_gpus = sorted(good_gpus, key=lambda g: MachineInfo.gpu2vram[g])
    #     good_gpu_str = "/".join(good_gpus) if len(good_gpus) > 0 else "good-gpu"

    #     bad_gpus = [g for g in all_gpus if not MachineInfo.gpu2info[g]["good"]]
    #     bad_gpus = sorted(bad_gpus, key=lambda g: MachineInfo.gpu2vram[g])
    #     bad_gpu_str = "/".join(bad_gpus) if len(bad_gpus) > 0 else "bad-gpu"

    #     resource2nodes = defaultdict(list)
    #     for node in node_list:
    #         for gpu in node.gpu2count_total.keys():
    #             resource_key = good_gpu_str if gpu in good_gpus else bad_gpu_str
    #             resource2nodes[resource_key].append(node)

    #     result_nodes = []
    #     for resource_key,nodes in resource2nodes.items():

    #         min_time = min([node.max_time for node in nodes])
    #         for node in nodes:
    #             new_node = copy.deepcopy(node)
    #             new_node.gpu2count_total = {resource_key: sum(node.gpu2count_total.values())}
    #             new_node.gpu2avail = {resource_key: sum(node.gpu2avail.values())}
    #             new_node.max_time = min_time
    #             new_node.max_time_str = UtilsBase.time_to_pretty_str(new_node.max_time * 3600)
    #             result_nodes.append(new_node)
    #     return result_nodes
            
def cluster_to_partition_str(args):
    """Returns the partitions of interest on the cluster."""
    if args.partitions:
        return ",".join(args.partitions)

    if Utils.get_cluster_type() == "trillium":
        result = ["compute_full_node"]
    elif Utils.is_solar():
        result = ["cs-gpu-research"]
    elif Utils.is_cc():
        result = [f"gpubase_bygpu_b{n}" for n in range(1,6)] + [f"gpubase_bynode_b{n}" for n in range(1,6)] + ["gpubackfill"]
        result += ["gpubase_interac", "interac"] if args.interac else []
    else:
        raise NotImplementedError()
    return ",".join(result)

def cluster_to_node_list_str():
    """Returns a string that can be passed to sinfo's --node_list option to give
    information for only nodes of interest on the current cluster.

    This includes all GPU nodes, interactive or otherwise.

    Use the command 'sinfo -O 'Gres:40   ,NodeList:200'' to find a short list of nodes
    with GPUs on them. I think you could also just use upper and lower bounds on this
    list too.
    """
    if Utils.is_solar():
        result = "cs-venus-[01-18]"
    elif Utils.get_cluster_type() == "trillium":
        result = "trig[0001-0063]"
    elif Utils.get_cluster_type() == "nibi":
        result = "g[01-36]"
    elif Utils.get_cluster_type() == "fir":
        result = "fc[10601-10607,10609-10620,10701-10720,10901-10920,11001-11020],fc[10101-10120,10201-10208,10210-10220,10402-10420,10501-10506,10508-10514,10516,10518-10520],fc[10209,10401,10507,10515,10517]"
    elif Utils.get_cluster_type() == "rorqual":
        result = "rg[31501-31503,31601-31609],rg[12501-12503,12601-12603,12701-12703,12801-12803,12901-12903,13001-13003,13101-13103,13201-13203,13301-13303,13401-13403,13501-13503,13601-13603],rg[21701-21708,21801-21807,31701-31703,31801-31803,31901-31903,32001-32003,32101-32103,32201-32203,32301-32303,32401-32403,32501-32503,32601-32603]"
    elif Utils.get_cluster_type() == "narval":
        result = "ng[10101-10104,10201-10204,10301-10304,10401-10404,10501-10504,10601-10610,10701-10712,10801-10808,10901-10906,11001-11006,11101-11106,20101-20104,20201-20204,20301-20303,30601-30605,30701-30712,30801-30811,30901-30912,31001-31006,31101-31104,31201-31205,31301-31305,31401-31402],ng20304,ng[20401-20404,20501,20504,20601-20604,30103-30104,30301-30302,30304,30402,30501-30504],ng[20502-20503,30101-30102,30201-30204,30303,30401,30403-30404]"
    else:
        raise NotImplementedError()

    # Replace all non-integer characters in [result] with spaces, leaving a string of
    # whitespace and integers. Then split on whitespace to get all the individual node
    # numbers. This isn't all the nodes in the cluster, but we do have a guaruntee
    # that it includes the smallest- and largest-indexed GPU nodes.
    # result_numbers = [c if c.isdigit() else " " for c in result]
    # result_numbers = "".join(result_numbers).split()
    # result_numbers = [int(rn) for rn in result_numbers]
    # min_node, max_node = min(result_numbers), max(result_numbers)
    # return MachineInfo.cluster2node_prefix[Utils.get_cluster_type()] + f"[{min_node}-{max_node}]"
    return result

def get_nodes_from_sinfo_data(args):
    """Returns a list of Node objects representing the nodes on the cluster."""
    key2si_format_O = dict(
        partition="PartitionName:32",
        max_time="Time:32",
        # state="StateComplete:128",
        name="NodeList:1024", # Name of the node
        cpu_state="CPUsState:32",
        gres="Gres:512",
        gres_used="GresUsed:512",
        free_mem="FreeMem:32",
        memory="Memory:32",
        allocated_mem="AllocMem:32",
    )

    key2si_format_O |= dict(state="StateComplete:32") if not Utils.is_solar() else dict()

    si_format_strs = ",".join(key2si_format_O.values())
    nodes_str = cluster_to_node_list_str()
    partitions_str = cluster_to_partition_str(args)
    si_cmd = f"sinfo -N --partition={partitions_str} --nodes={nodes_str}  -O '{si_format_strs}'"

    _ = twrite(f"[INFO] Running command: {si_cmd}", quiet=not args.verbose)
    si = subprocess.getoutput(si_cmd)
    _ = twrite(f"[INFO] Got sinfo output:\n{si}", quiet=not args.vv)

    si_lines = [line for line in si.strip().split("\n") if len(line.strip()) > 0]
    si_lines = si_lines[1:] # Remove the header line
    si_lines = [line.split() for line in si_lines]
    si_lines = [[ll.strip() for ll in line] for line in si_lines]
    
    if args.verbose:
        col2max_width = defaultdict(lambda: 0)
        for line in si_lines:
            for col_idx,entry in enumerate(line):
                col2max_width[col_idx] = max(col2max_width[col_idx], len(entry))

        si_line_strs = ["\t".join([entry.ljust(col2max_width[col_idx]+1) for col_idx,entry in enumerate(line)]) for line in si_lines]
        si_sanitized = "\n".join(si_line_strs)
        _ = twrite(f"[INFO] Sanitized sinfo output:\n{si_sanitized}")

    line_dicts = [dict(zip(key2si_format_O.keys(), line)) for line in si_lines]
    line_dicts = [ld | dict(correct_parse=all([k in ld for k in key2si_format_O.keys()])) for ld in line_dicts]
    line_dicts = [ld | dict(state="up") if Utils.is_solar() and "state" not in ld else ld for ld in line_dicts]

    # In practice, this might be more correct!
    line_dicts = [ld | dict(free_mem=str(int(ld["memory"]) - int(ld["allocated_mem"]))) for ld in line_dicts]

    nodes = [Node(**ld) for ld in line_dicts]
    nodes = Node.sanitize_node_list_across_partitions(nodes)
    return nodes
        

def get_resource_info_summary():
    args = get_args([])
    nodes = get_nodes_from_sinfo_data(args)
    time2resource2info = node_list_to_resources_info(nodes=nodes, args=args)
    time2resource2info = aggregate_resource_infos(time2resource2info, args) if args.aggregate else time2resource2info
    s = resource_infos_to_str(time2resource2info,
        show_nodes=2 if Utils.get_cluster_type() in ["nibi", "solar", "solar1"] else 1,
        max_nodes_to_show=args.max_nodes_to_show,
        good_gpus_only="good" in args.gpus,
        max_time_to_show=args.max_time)
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
    args = P.parse_args(args)

    args.partitions = UtilsBase.flatten([p.split(",") for p in args.partitions]) if args.partitions else None
    if "good" in args.gpus:
        args.gpus += [g for g in MachineInfo.gpu2info.keys() if MachineInfo.gpu2info[g]["good"]]
    args.gpu_counts = [int(gc) if str(gc).isnumeric() else gc for gc in args.gpu_counts]

    return args

if __name__ == "__main__":
    args = get_args()
    nodes = get_nodes_from_sinfo_data(args)

    if args.nodes:
        twrite("\n[INFO] Node information:\n" + "\n".join([str(node) for node in nodes]))

    time2resource2info = node_list_to_resources_info(nodes=nodes, args=args)

    time2resource2info = aggregate_resource_infos(time2resource2info, args) if args.aggregate else time2resource2info

    # print(time2resource2info)


    print("\n[INFO] Resource availability by max time:")
    for time,resource2info in time2resource2info.items():
        print(f"Max time: {time}")
        for resource,info in resource2info.items():
            print(f"\tResource: {resource}:\t\ttotal={info.total}, avail={info.avail}, free={info.free}, vram={info.vram}GB")
    s = resource_infos_to_str(time2resource2info,
        show_nodes=2,
        max_nodes_to_show=args.max_nodes_to_show,
        good_gpus_only="good" in args.gpus,
        max_time_to_show=args.max_time)
    print(f"\n[INFO] Summary:\n{s}")
    