"""Generates an interactive SLURM job. Basically, it's dumb to have to type out most
of the configuration when it's generally the same.

We will alias this as:

sallocb="python ~/.ScriptsAndAliases/GetInteractiveJob.py salloc "
srunb="python ~/.ScriptsAndAliases/GetInteractiveJob.py srun "
"""



# No reason to expose imports outside of main; there's literally no reason anything
# should try and import from this script.
if __name__ == "__main__":
    import argparse
    import subprocess

    import UserConfig
    import Utils
    import UtilsBase
    from UtilsBase import twrite

    # def get_all_possible_nodes():
    #     cmd = "scontrol show nodes | grep NodeName= "
    #     output = subprocess.getoutput(cmd)
    #     nodes = [UtilsBase.strip_left(n, "NodeName=") for n in output.split("\n")]
    #     nodes = {n.split()[0] for n in nodes}
    #     return nodes

    def get_default_account():
        """Returns the default account to use for the current cluster."""
        if Utils.is_solar():
            return None
        else:
            account = UserConfig.cluster2accounts.get(Utils.get_cluster_type(), [None])[0]
            if account is None:
                twrite(f"[WARNING] No default account found for cluster {Utils.get_cluster_type()}! Using no account.")
            return account

    def get_default_partition():
        """Returns the default partition to use for the current cluster."""
        if Utils.is_solar():
            account = UserConfig.cluster2accounts.get(Utils.get_cluster_type(), [None])[0]
            if account is None:
                twrite(f"[WARNING] No default account found for cluster {Utils.get_cluster_type()}! Using no partition.")
                return None
        else:
            return None

    P = argparse.ArgumentParser()
    P.add_argument("scommand", type=str, choices=["salloc", "srun"], help="Whether to use salloc or srun for the interactive job")
    P.add_argument("--mem", type=str, default="64G", help="Memory to allocate")
    P.add_argument("--ntasks-per-node", type=int, default=1, help="Number of tasks per node")
    P.add_argument("--cpus-per-task", type=int, default=1, help="Number of CPUs per task")
    P.add_argument("--gpus_per_node", "--gpus", nargs="*", default=None, help="Number of GPUs per node")
    P.add_argument("--time", type=str, default="1:00:00", help="Time limit for the job (e.g., '1:00:00' for 1 hour)")
    P.add_argument("-n", "--nodelist", type=str, const=None, nargs="?", help="Comma-separated list of nodes to use (e.g., 'node1,node2')")
    P.add_argument("--account", type=str, default=get_default_account(),
        help="Account to use for the job. By default, uses the first account listed for the cluster in UserConfig.py, or no account if none are listed.")
    P.add_argument("--partition", type=str, default=get_default_partition(), 
        help="Partition to use. In general, this should be left to be set automatically!")
    args, unparsed_args = P.parse_known_args()

    # Handle node specification
    if not args.nodelist is None and Utils.is_solar():
        from MachineInfo import MachineInfo
        nodelist = UtilsBase.flatten([n.split(",") for n in args.nodelist])
        solar_nodes = MachineInfo.cluster2node2config["solar"]
        req_node2substring_of_nodes = {n: [s for s in solar_nodes if n in s] for n in nodelist}
        nodelist = UtilsBase.flatten(req_node2substring_of_nodes.values())
        nodelist = sorted(set(nodelist))
        args.nodelist = ",".join(nodelist)
    elif not args.nodelist is None:
        nodelist = UtilsBase.flatten([n.split(",") for n in args.nodelist])
        args.nodelist = ",".join(nodelist)
    elif args.nodelist is None:
        args.nodelist = None

    # Expand --gpus-per-node to allow for GPU shorthands and non-specification.
    # Interpret the --gpus-per-node flag. If no GPU type is given, then assume any is
    # okay on ComputeCanada. On Solar, assume anything better than an A5000 is fine,
    # ie. rgu_multiplier >= 3.0.
    if args.gpus_per_node:
        args.gpus_per_node = UtilsBase.flatten([g.split(",") for g in args.gpus_per_node]) if args.gpus_per_node else None
        gpu_per_node_configs = []
        from MachineInfo import cluster2node2config, gpu_alias2name, gpu2info
        import Utils
        node2config = cluster2node2config[Utils.get_cluster_type()]
        for gpu_spec in args.gpus_per_node:
            if Utils.is_solar() and not ":" in gpu_spec:
                raise NotImplementedError()
            elif not Utils.is_solar() and not ":" in gpu_spec:
                gpu_count = gpu_spec
                gpu_per_node_configs += [(gpu_alias, gpu_count) for gpu_alias, gpu_config in node2config.items() if not gpu_alias == "default" and gpu_config["can_allocate"]]
            else:
                gpu_alias, gpu_count = gpu_spec.split(":")
                gpu_per_node_configs.append((gpu_alias, gpu_count))

        # Map GPU types to their full names.
        gpu_per_node_configs = [(gpu_alias2name.get(gpu_alias, gpu_alias), gpu_count) for gpu_alias, gpu_count in gpu_per_node_configs]
        args.gpus_per_node = ",".join([f"{gpu_type}:{gpu_count}" for gpu_type, gpu_count in gpu_per_node_configs])


    # Construct the salloc command
    account_str = f"--account={args.account}" if not args.account is None else ""
    partition_str = f"--partition={args.partition}" if not args.partition is None else ""
    ntasks_per_node_str = f"--ntasks-per-node={args.ntasks_per_node}"
    cpus_per_task_str = f"--cpus-per-task={args.cpus_per_task}"
    gpus_per_node_str = f"--gpus-per-node={args.gpus_per_node}" if not args.gpus_per_node is None else ""
    time_str = f"--time={args.time}"
    nodelist_str = f"--nodelist={args.nodelist}" if not args.nodelist is None else ""
    mem_str = f"--mem={args.mem}"

    all_options = [account_str, partition_str, ntasks_per_node_str, cpus_per_task_str, gpus_per_node_str, mem_str, time_str, nodelist_str] + unparsed_args
    all_options_str = " ".join(all_options).strip()
    command = f"{args.scommand} {all_options_str}"

    UtilsBase.twrite(f"[INFO] Running command:\n\t{command}")
    subprocess.run(command, shell=True)


