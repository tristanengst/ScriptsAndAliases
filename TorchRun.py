"""Updated, better version of CPU-managed DDP launch. The old ones either didn't do
the right thing or became bloated.

Syntax:
tpython_ddp .... --gpus GPU_INDICES or --strip_gpus GPU_INDICES

Equivalent to:
CUDA_VISIBLE_DEVICES=GPU_INDICES taskset -c CPU_INDICES python script.py ARGS

NOTE: Environment variables specified prior to the script will be preserved!
NOTE: If --wandb=online isn't specified, and the script is one of [WANDB_LOGGING_SCRIPTS], it will be added automatically. This is basically the only weird complexity.
"""
import argparse
import os
import os.path as osp
import sys
from UtilsBase import twrite
import SSHCommunication
import MachineInfo

# Scripts for which --wandb=online will be added if --wandb isn't specified
WANDB_LOGGING_SCRIPTS = ["TrainSSL2.py", "EvalKNN.py", "EvalLinear.py", "EvalFinetune.py", "EvalSegmentation2.py", "EvalSegmentation2.py",]

def get_local_hw_info():
    """Returns a dictionary of local hardware information."""
    num_cpus = os.cpu_count()
    import subprocess
    cmd = "nvidia-smi --query-gpu=uuid --format=csv,noheader"
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    gpus = [line for line in result.stdout.splitlines() if line.strip()]
    return argparse.Namespace(total_cpus=num_cpus, total_gpus=len(gpus))

server_gpu2cpu = {
    0: list(range(0, 6)) + list(range(64, 70)),
    1: list(range(6, 12)) + list(range(70, 76)),
    2: list(range(12, 18)) + list(range(76, 82)),
    3: list(range(18, 24)) + list(range(82, 88)),
    4: list(range(24, 32)) + list(range(88, 96)), # Gets extra CPUs: 30, 31, 94, 95
    5: list(range(32, 38)) + list(range(96, 102)),
    6: list(range(38, 44)) + list(range(102, 108)),
    7: list(range(44, 50)) + list(range(108, 114)),
    8: list(range(50, 56)) + list(range(114, 120)),
    9: list(range(56, 64)) + list(range(120, 128))} # Gets extra CPUs: 62, 63, 126, 127
server_gpu2cpu = {gpu: sorted(cpus) for gpu, cpus in server_gpu2cpu.items()}

def get_taskset_str(*, gpus):
    """Returns the string of CPU indices to feed to taskset for the specified GPUs."""
    machine_name = SSHCommunication.hostname_to_machine(SSHCommunication.get_hostname())

    # Servers use a specific map because we enable hyperthreading. Although it can be
    # computed, it's easier to just state it plainly for quick reference
    if machine_name in ["S1", "S2", "S3"]:
        gpu2cpu = server_gpu2cpu
    else:
        host_info = get_local_hw_info()
        cpus_per_gpu =  host_info.total_cpus // host_info.total_gpus
        gpu2cpu = {gpu_idx: list(range(gpu_idx * cpus_per_gpu, (gpu_idx+1) * cpus_per_gpu-1)) for gpu_idx in range(host_info.total_gpus)}

    cpu_range = sorted(set([c for gpu in gpus for c in gpu2cpu[gpu]]))
    cpu_ranges = []
    cur_cpu = cpu_range[0]
    for c in cpu_range:
        if c == cur_cpu + 1:
            cpu_ranges[-1].append(c)
        else:
            cpu_ranges.append([c])
        cur_cpu = c
        
    taskset_str = ",".join(f"{cr[0]}-{cr[-1]}" if len(cr) > 1 else f"{cr[0]}" for cr in cpu_ranges)
    taskset_str = f"taskset -c {taskset_str}"
    return taskset_str

def parse_gpu_indices_and_args_from_argv(*, tpython_ddp_args, argv):
    """Returns a (gpu_indices, new_argv) tuple, where [gpu_indices] is a list of GPU
    indices to run on and [new_argv] is the modified argv with the GPU indices
    replaced by 0,1,...,N-1 or removed if --strip_gpus was specified.
    """
    if tpython_ddp_args.strip_gpus is not None:
        strip_gpus_specified = True
        gpu_indices = tpython_ddp_args.strip_gpus
        new_argv = argv
    elif "--gpus" in argv:
        strip_gpus_specified = False
        gpu_start_idx = argv.index("--gpus")+1
        gpu_indices = []
        for arg in argv[gpu_start_idx:]:
            if arg.isdigit():
                gpu_indices.append(int(arg))
            else:
                break
        gpu_end_idx = gpu_start_idx + len(gpu_indices)
        new_argv = argv[:gpu_start_idx] + [str(gpu_idx) for gpu_idx in range(len(gpu_indices))] + argv[gpu_end_idx:]
    else:
        raise ValueError("Must specify either --gpus or --strip_gpus")

    ##################################################################################
    # Ensure that --wandb=online is set in [new_argv]
    ##################################################################################
    script_basename = osp.basename(new_argv[0]) if len(new_argv) > 0 else None
    if script_basename in WANDB_LOGGING_SCRIPTS:
        if not any(arg.startswith("--wandb") for arg in new_argv):
            new_argv.append("--wandb online")

    return gpu_indices, new_argv

if __name__ == "__main__":
    ##################################################################################
    # Parse arguments
    ##################################################################################
    P = argparse.ArgumentParser(allow_abbrev=False)
    P.add_argument("--strip_gpus", nargs="+", default=None, type=int,
        help="GPU indices to strip from the command. If specified, the GPUs will be removed naturally!")
    P.add_argument("--time", action="store_true",
        help="Convenient to specify here to avoid it propagating into the script")
    P.add_argument("--wandb_online", choices=[0, 1], default=1,
        help="If --wandb=online isn't set, and the script is one of [WANDB_LOGGING_SCRIPTS], set it.")
    P.add_argument("--taskset_debug", action="store_true",
        help="If set, will print the command instead of executing it")

    # Allow disabling/enabling the taskset -c functionality.
    taskset_group = P.add_mutually_exclusive_group()
    P.set_defaults(taskset=True)
    P.add_argument("--taskset", action="store_true",
        help="If set, will use taskset to bind the process to specific CPUs")
    P.add_argument("--no-taskset", "--no_taskset", action="store_false", dest="taskset",
        help="If set, will not use taskset to bind the process to specific CPUs")
    tpython_ddp_args, argv = P.parse_known_args()
    ##################################################################################
    ##################################################################################
    ##################################################################################

    gpu_indices, new_argv = parse_gpu_indices_and_args_from_argv(tpython_ddp_args=tpython_ddp_args, argv=argv)
    taskset_str = get_taskset_str(gpus=gpu_indices) if tpython_ddp_args.taskset else ""
    cuda_devices_str = ",".join([str(gpu_idx) for gpu_idx in gpu_indices])
    torchrun_str = f"torchrun --nproc_per_node={len(gpu_indices)} --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0"
    command = f"CUDA_VISIBLE_DEVICES={cuda_devices_str} {taskset_str} {torchrun_str} {' '.join(new_argv)}"

    if tpython_ddp_args.taskset_debug:
        twrite(f"[DEBUG] Would run command:\n{command}")
    else:
        twrite("-----------------------")
        twrite(f"[INFO] Running command: {command}")
        twrite("-----------------------")
        os.system(command)

