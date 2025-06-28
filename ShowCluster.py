import argparse
import math
import subprocess

class Node:
    def __init__(self, node_name, state, gpu_type, gpus, cpus, memory, alloc_gpus, alloc_cpus, alloc_memory):
        self.node_name = node_name
        self.state = state
        self.gpu_type = gpu_type
        self.gpus = gpus
        self.cpus = cpus
        self.memory = memory
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

        self.available = self.fraction_avail == 1 and self.can_allocate
        self.avail_gpus = self.fraction_avail * self.possible_gpus
        

    def __repr__(self):
        return f"{self.__class__.__name__}(avail_gpus={self.avail_gpus})"

    @staticmethod
    def print_cluster_stats(node_list):
        """Returns a dictionary with the total number of nodes, total GPUs, and
        available GPUs in the cluster.
        """
        total_nodes = len(node_list)
        total_gpus = sum([n.gpus for n in node_list if n.gpus is not None])

        avail_full_nodes = len([n for n in node_list if n.available])
        avail_gpus = sum([n.avail_gpus for n in node_list if n.available])

        possible_nodes = len([n for n in node_list if n.can_allocate])
        possible_gpus = sum([n.possible_gpus for n in node_list])

        print(f"AvailFullNodes={avail_full_nodes}/{total_nodes} AvailGPUs={avail_gpus}/{total_gpus} PossibleNodes={possible_nodes}/{total_nodes} PossibleGPUs={possible_gpus}/{total_gpus}")

    @staticmethod
    def get_node_list():
        cmd = "scontrol show nodes"
        nodes = subprocess.getoutput(cmd)
        lines = nodes.split("\n")

        node_list = []

        node_name = None
        state = None
        gpu_type = None
        gpus = None
        cpus = None
        memory = None
        alloc_gpus = 0
        alloc_cpus = 0
        alloc_memory = 0

        for line in lines:
            line = line.strip()
            if line.startswith("NodeName="):

                if not node_name is None and not gpu_type is None:
                    node_list.append({
                        "node_name": node_name,
                        "state": state,
                        "gpu_type": gpu_type,
                        "gpus": gpus,
                        "cpus": cpus,
                        "memory": memory,
                        "alloc_gpus": alloc_gpus,
                        "alloc_cpus": alloc_cpus,
                        "alloc_memory": alloc_memory
                    })

                node_name = line.split("=")[1].split()[0]
                state = None
                gpu_type = None
                gpus = None
                cpus = None
                memory = None
                alloc_gpus = 0
                alloc_cpus = 0
                alloc_memory = 0

            elif line.startswith("Gres="):
                if not any([g in line for g in ["h100", "a100", "v100", "l40s", "a40", "a5000"]]):
                    continue
                gres = line.split("=")[1].split(",")
                _, gpu_type, gpus = gres[0].split(":")
                gpus = int(gpus)
            
            elif line.startswith("State="):
                state = line.split("=")[1].split()[0]
            elif line.startswith("AllocTRES="):
                alloc_tres = line.replace("AllocTRES=", "")
                alloc_tres = alloc_tres.split(",")
                for tres in alloc_tres:
                    if tres.startswith("gres/gpu"):
                        alloc_gpus = int(tres.split("=")[1])
                    elif tres.startswith("cpu"):
                        alloc_cpus = int(tres.split("=")[1])
                    elif tres.startswith("mem"):
                        alloc_memory = tres.split("=")[1]
                        if alloc_memory.endswith("M"):
                            alloc_memory = int(alloc_memory[:-1]) // (1024)
                        elif alloc_memory.endswith("G"):
                            alloc_memory = int(alloc_memory[:-1])
                        elif alloc_memory.endswith("K"):
                            alloc_memory = int(alloc_memory[:-1]) // (1024 * 1024)
                        else:
                            alloc_memory = int(alloc_memory)
            elif line.startswith("CfgTRES="):
                cfg_tres = line.replace("CfgTRES=", "")
                cfg_tres = cfg_tres.split(",")
                for tres in cfg_tres:
                    if tres.startswith("gres/gpu"):
                        gpus = int(tres.split("=")[1])
                    elif tres.startswith("cpu"):
                        cpus = int(tres.split("=")[1])
                    elif tres.startswith("mem"):
                        memory = tres.split("=")[1]
                        if memory.endswith("M"):
                            memory = int(memory[:-1]) // (1024)
                        elif memory.endswith("G"):
                            memory = int(memory[:-1])
                        elif memory.endswith("K"):
                            memory = int(memory[:-1]) // (1024 * 1024)
                        else:
                            memory = int(memory)
            
        if not node_name is None and not gpu_type is None:
                node_list.append({
                    "node_name": node_name,
                    "state": state,
                    "gpu_type": gpu_type,
                    "gpus": gpus,
                    "cpus": cpus,
                    "memory": memory,
                    "alloc_gpus": alloc_gpus,
                    "alloc_cpus": alloc_cpus,
                    "alloc_memory": alloc_memory
                })

        node_list = [Node(**n) for n in node_list if n["gpus"] is not None]
        return node_list


if __name__ == "__main__":
    Node.print_cluster_stats(Node.get_node_list())
        
            