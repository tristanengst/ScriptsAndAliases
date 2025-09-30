"""Code that will allocate a node but in an intelligent way."""
import argparse
from MachineInfo import cluster2node2config
import UtilsBase
import Utils

def args_to_gres_str(args):
    """Returns a --gres= string from [args]."""
    if args.gres is None and len(args.gpus) == 0:
        return ""
    elif not args.gres is None:
        _ = twrite(f"[INFO] args_to_gres_str() ignoring --gpus={args.gpus} and --gpus_type={args.gpu_type} since --gres={args.gres} is specified")
        return f"--gres={args.gres}"
    else:
        raise NotImplementedError()

def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--gpus", type=int, nargs="+", default=[],
        help="List of GPU indices to allocate (you do not actually get the indices)")
    P.add_argument("--ntasks-per-node", type=int, default=1,
        help="ntasks-per-node")
    P.add_argument("--mem", type=int, default=0,
        help="Amount of memory to request in GB. -1 or 0 scales by the number of CPUs")
    P.add_argument("--cpus-per-task", type=int, default=4,
        help="Number of CPUs (per task) to request")
    P.add_argument("--nodelist", default="default", choices=cluster2node2config[Utils.get_cluster_type()].keys(),
        help="Node to allocate. Can be interpreted in the general case sometimes")
    P.add_argument("--gres", default=None,
        help="If provided, overrides --gpus and --gpu_type")
    P.add_argument("--time", default="2:00:00",
        help="Time string to allocate")
    args, unparsed_args = P.parse_known_args()

    return args, unparsed_args

if __name__ == "__main__":
    args, unparsed_args = get_args()

    node = args.nodelist
    node2config = cluster2node2config[Utils.get_cluster_type()]

    if Utils.is_solar() and args.nodelist == "default":
        raise ValueError(f"On Solar/Solar1, must specify --nodelist")
    elif Utils.is_solar():
        cpus_per_node = node2config[node]["cpus_per_gpu"] * node2config[node]["gpus_per_node"]
        mem_per_node = node2config[node]["mem_per_gpu"] * node2config[node]["gpus_per_node"]

        num_cpus = args.ntasks_per_node * args.cpus_per_task
        cpu_frac = num_cpus / cpus_per_node
        mem = args.mem if args.mem > 0 else max(int(cpu_frac * mem_per_node), 1)

        gres_str = args_to_gres_str(args)
        mem_str = f" --mem={mem}G "
        cpu_str = f" --ntasks-per-node={args.ntasks_per_node} --cpus-per-task={args.cpus_per_task} "
        time_str = f" --time={args.time}"
        unparsed_args_str = " " + " ".join(unparsed_args) + " "

        s = f"srun -J \"interactive-bash\" --nodelist={args.nodelist} {cpu_str} {mem_str} {gres_str} {time_str} {unparsed_args_str} --pty bash"
        print(s)




    else:
        raise NotImplementedError()