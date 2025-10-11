import argparse
import math
import subprocess

import MachineInfo
import Utils
import UtilsBase

class Node:
    def __init__(self, node_name, state, gpu_type, fractional_gpus, gpus, cpus, memory, alloc_fractional_gpus, alloc_gpus, alloc_cpus, alloc_memory):
        self.node_name = node_name
        self.state = state
        self.gpu_type = gpu_type
        self.fractional_gpus = fractional_gpus
        self.gpus = gpus
        self.cpus = cpus
        self.memory = memory
        self.alloc_fractional_gpus = alloc_fractional_gpus
        self.alloc_gpus = alloc_gpus
        self.alloc_cpus = alloc_cpus
        self.alloc_memory = alloc_memory

        # Figure out how many fractions of the node are allocated. If the node has N
        # GPUs, there are N fractions. We take the maximum fraction over GPUs, CPUs,
        # and memory.
        self.fraction_allocated = max(
            (alloc_gpus / gpus) if gpus else 0,
            (alloc_cpus / cpus) if cpus else 0,
            (alloc_memory / memory) if memory else 0
        )
        self.fraction_allocated = math.ceil(self.fraction_allocated * gpus) / gpus
        
        self.fraction_avail = 1 - self.fraction_allocated
        self.can_allocate = not any([s in self.state for s in ["DOWN", "NOT_RESPONDING", "DRAIN", "RESERVED"]])
        self.possible_gpus = self.gpus if self.can_allocate else 0

        self.available = self.state == "IDLE"
        self.avail_gpus = self.fraction_avail * self.possible_gpus
        

    def __repr__(self):
        return f"{self.__class__.__name__}({self.node_name}, avail_gpus={self.avail_gpus}/{self.possible_gpus}, state={self.state}, gpu_type={self.gpu_type})"

    @staticmethod
    def print_cluster_stats():
        """Returns a dictionary with the total number of nodes, total GPUs, and
        available GPUs in the cluster.
        """
        return print(Node.cluster_stats_to_str())
    
    @staticmethod
    def cluster_stats_to_str():

        def cluster_stats_to_str_(gpu_type, node_list):
            total_nodes = len(node_list)
            total_gpus = sum([n.gpus for n in node_list if n.gpus is not None])

            avail_full_nodes = len([n for n in node_list if n.available])
            avail_gpus = sum([n.avail_gpus for n in node_list if n.available])

            possible_nodes = len([n for n in node_list if n.can_allocate])
            possible_gpus = sum([n.possible_gpus for n in node_list])

            avail_full_node_list = [n.node_name for n in node_list if n.available]
            avail_full_node_str = (f"(" + ", ".join(avail_full_node_list) + ")") if avail_full_node_list else ""
            return f"{gpu_type}=[AvailFullNodes={avail_full_nodes}/{total_nodes} {avail_full_node_str} AvailGPUs={avail_gpus}/{possible_gpus} PossibleNodes={possible_nodes}/{total_nodes} PossibleGPUs={possible_gpus}/{total_gpus}]"


        node_list = Node.get_node_list()
        gpu_types = set([n.gpu_type for n in node_list])
        gpu_type2node_list = {g: [n for n in node_list if n.gpu_type == g] for g in gpu_types}
        stats = [cluster_stats_to_str_(g, l) for g,l in gpu_type2node_list.items()]
        stats_str = "\t\t".join(stats)
        return stats_str





        

    @staticmethod
    def get_node_list():
        cmd = "scontrol show nodes"
        nodes = subprocess.getoutput(cmd)
        lines = nodes.split("\n")

        node_list = []

        node_name = None
        state = None
        gpu_type = None
        fractional_gpus = None
        gpus = None
        cpus = None
        memory = None
        alloc_fractional_gpus = 0
        alloc_gpus = 0
        alloc_cpus = 0
        alloc_memory = 0


        good_gpus = ["h100", "a100", "v100", "l40s", "a40", "a5000","nvidia_h100_80gb_hbm3_3g.40gb"]

        for line in lines:
            line = line.strip()
            if line.startswith("NodeName="):

                if not node_name is None and not gpu_type is None:
                    node_list.append({
                        "node_name": node_name,
                        "state": state,
                        "gpu_type": gpu_type,
                        "fractional_gpus": fractional_gpus,
                        "gpus": gpus,
                        "cpus": cpus,
                        "memory": memory,
                        "alloc_fractional_gpus": fractional_gpus,
                        "alloc_gpus": alloc_gpus,
                        "alloc_cpus": alloc_cpus,
                        "alloc_memory": alloc_memory
                    })

                node_name = line.split("=")[1].split()[0]
                state = None
                gpu_type = None
                fractional_gpus = None
                gpus = None
                cpus = None
                memory = None
                alloc_fractional_gpus = 0
                alloc_gpus = 0
                alloc_cpus = 0
                alloc_memory = 0

            elif line.startswith("Gres=") and Utils.get_cluster_type() == "trillium":
                if not any([g in line for g in good_gpus]):
                    print(f"no GPU for node={node_name}, line={line}")
                    continue
                gres = line.split("=")[1].split(",")
                gres = gres[0].split(":")
                gpu_type = gres[1]
                gpu = int(gres[2][0])

            
            elif line.startswith("State="):
                state = line.split("=")[1].split()[0]
            elif line.startswith("AllocTRES="):
                alloc_tres = line.replace("AllocTRES=", "")
                alloc_tres = alloc_tres.split(",")
                alloc_fractional_gpus = int("g." in line)
                for tres in alloc_tres:
                    if tres.startswith("gres/gpu") and fractional_gpus:
                        alloc_fractional_gpus = int(tres.split("=")[1])
                    elif tres.startswith("gres/gpu"):
                        alloc_gpus = int(tres.split("=")[1])
                    elif tres.startswith("cpu"):
                        alloc_cpus = int(tres.split("=")[1])
                    elif tres.startswith("mem"):
                        alloc_memory = tres.split("=")[1]
                        if alloc_memory.endswith("M"):
                            alloc_memory = float(alloc_memory[:-1]) // (1024)
                        elif alloc_memory.endswith("G"):
                            alloc_memory = float(alloc_memory[:-1])
                        elif alloc_memory.endswith("K"):
                            alloc_memory = float(alloc_memory[:-1]) // (1024 * 1024)
                        elif alloc_memory.endswith("T"):
                            alloc_memory = float(alloc_memory[:-1]) * 1024
                        else:
                            alloc_memory = float(alloc_memory)
            elif line.startswith("CfgTRES="):
                cfg_tres = line.replace("CfgTRES=", "")
                cfg_tres = cfg_tres.split(",")

                good_gpus = ["h100", "a100", "v100", "l40s", "a40", "a5000","nvidia_h100_80gb_hbm3_3g.40gb"]
                for tres in cfg_tres:
                    if tres.startswith("gres/gpu") and Utils.get_cluster_type() == "trillium":
                        gpus = float(tres.split("=")[1])
                    elif tres.startswith("gres/gpu") and len([t for t in cfg_tres if t.startswith("gres/gpu")]) == 1:
                        gpus = float(tres.split("=")[1])
                    elif tres.startswith("gres/gpu") and UtilsBase.strip_left(tres, "gres/gpu:").split("=")[0] in good_gpus:
                        gpus = float(tres.split("=")[1])
                        gpu_type = UtilsBase.strip_left(tres, "gres/gpu:").split("=")[0]
                        gpu_type = MachineInfo.gpu_name_to_type(gpu_type)
                    elif tres.startswith("cpu"):
                        cpus = float(tres.split("=")[1])
                    elif tres.startswith("mem"):
                        memory = tres.split("=")[1]
                        if memory.endswith("M"):
                            memory = float(memory[:-1]) // (1024)
                        elif memory.endswith("G"):
                            memory = float(memory[:-1])
                        elif memory.endswith("K"):
                            memory = float(memory[:-1]) // (1024 * 1024)
                        elif memory.endswith("T"):
                            memory = float(memory[:-1]) * 1024
                        else:
                            memory = float(memory)
            
        if not node_name is None and not gpu_type is None:
                node_list.append({
                    "node_name": node_name,
                    "state": state,
                    "gpu_type": gpu_type,
                    "fractional_gpus": fractional_gpus,
                    "gpus": gpus,
                    "cpus": cpus,
                    "memory": memory,
                    "alloc_fractional_gpus": alloc_fractional_gpus,
                    "alloc_gpus": alloc_gpus,
                    "alloc_cpus": alloc_cpus,
                    "alloc_memory": alloc_memory
                })

        node_list = [Node(**n) for n in node_list if n["gpus"] is not None]
        return node_list


if __name__ == "__main__":
    Node.print_cluster_stats()

    node_list = Node.get_node_list()
    for n in node_list:
        print(n)
        
            