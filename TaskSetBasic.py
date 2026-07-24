import sys
import os
import os.path as osp
from TaskSet import get_cpus_from_gpus

from UtilsBase import twrite

def args_with_gpus_to_gpu_idx_and_idx_in_args(args):
    gpu_idx_in_args = args.index("--gpus") + 1
    gpu_idx = int(args[gpu_idx_in_args])
    return gpu_idx, gpu_idx_in_args

def args_with_strip_gpus_to_gpu_idx_and_idx_in_args(args):
    gpu_idx_in_args = args.index("--strip_gpus") + 1
    gpu_idx = int(args[gpu_idx_in_args])
    return gpu_idx, gpu_idx_in_args

if __name__ == "__main__":
    cwd = os.getcwd()
    args = sys.argv

    gpus_in_args = "--gpus" in args
    strip_gpus_in_args = "--strip_gpus" in args

    if gpus_in_args:
        gpu_idx, gpu_idx_in_args = args_with_gpus_to_gpu_idx_and_idx_in_args(args)
    else:
        gpu_idx, gpu_idx_in_args = None, None

    if strip_gpus_in_args:
        strip_gpu_idx, strip_gpu_idx_in_args = args_with_strip_gpus_to_gpu_idx_and_idx_in_args(args)
    else:
        strip_gpu_idx, strip_gpu_idx_in_args = None, None

    twrite(args=args, strip_gpus_in_args=strip_gpus_in_args, gpus_in_args=gpus_in_args, gpu_idx=gpu_idx, strip_gpu_idx=strip_gpu_idx)

    if gpus_in_args and strip_gpus_in_args and not (gpu_idx == strip_gpu_idx):
        raise ValueError("Cannot specify both --gpus and --strip_gpus with different gpu indices")
    elif gpus_in_args and strip_gpus_in_args and gpu_idx == strip_gpu_idx:
        twrite(f"[WARNING] Both --gpus and --strip_gpus specified with GPUINDEX={gpu_idx} -> proceed, but be less weird")
    elif not gpus_in_args and not strip_gpus_in_args:
        raise ValueError("Must specify either --gpus or --strip_gpus")

    true_gpu_index = gpu_idx if gpus_in_args else strip_gpu_idx

    if gpus_in_args:
        args[gpu_idx_in_args] = "0"

    if strip_gpus_in_args:
        args = args[:strip_gpu_idx_in_args-1] + args[strip_gpu_idx_in_args+1:]

    pre_script, script, post_script = [], None, None
    for a in args[1:]:
        if osp.exists(a) and a.endswith(".py") or a.endswith(".sh"):
            script = a
            post_script = args[args.index(a)+1:]
            break
        else:
            pre_script.append(a) # Stuff like env vars

    if script is None:
        raise ValueError("No valid script found in arguments")

    pre_script_str = " ".join(pre_script) if pre_script else ""
    post_script_str = " ".join(post_script) if post_script else ""

    if script.endswith(".sh"):
        start_cmd = "bash"
    elif script.endswith(".py"):
        start_cmd = "python"
    else:
        raise ValueError("Script must end with .sh or .py")

    cpu_str = get_cpus_from_gpus(gpus=[true_gpu_index])
    os.chdir(cwd) # Just to be safe, make sure we're in the same directory as the script was called from

    gpu_str = f"CUDA_VISIBLE_DEVICES={true_gpu_index}"
    command = f"{gpu_str} taskset -c {cpu_str} {pre_script_str} {start_cmd} {script} {' '.join(args[2:])}"

    # Now execute the command using the shell that called this script
    print(f"RUNNING COMMAND: {command}")
    os.system(command)

