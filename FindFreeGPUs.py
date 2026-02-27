"""Pretty prints information on available GPUs. Heuristically, GPUs that aren't
running a process with 'python' in the name are available.
"""
import argparse
import math
from multiprocessing import Pool
import subprocess

import MachineInfo
import Utils
from UtilsBase import twrite

# Sometimes this takes a bit, so tqdm is nice. But we can't assume it's installed.
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x): return x

def find_free_gpus(h):
    """Returns a dictionary containing which GPUs on host [h] are free or not, and a
    string describing who is using how many. Heuristically, a free GPU is one for
    which nvidia-smi doesn't show any processes including 'python' in their name.
    """
    def gpu_line_to_data(l):
        """Returns a Namespace containing the GPU index and process name given by line
        of nvidia-smi output [l].
        """
        if "No running processes found" in l:
            return None
        
        l = l.split()
        proc_id = proc_id=l[4]

        if l[1].isdigit() and proc_id.isdigit():
            return argparse.Namespace(gpu=int(l[1]), proc_name=l[6], )
        else:
            print(f"[WARNING] Unexpected nvidia-smi line format on host {h}: {l}")
            return None

    host_info = MachineInfo.get_updated_machine_info(h)

    if host_info.nvidia_smi_ok:
        lines = host_info.nvidia_smi.split("\n")
        eq_line_idxs = [idx for idx,l in enumerate(lines) if l.startswith("|=")]
        idx_of_last_eq_line = eq_line_idxs[-1]
        lines = lines[idx_of_last_eq_line+1:-1]

        gpu_proc_name = [gpu_line_to_data(l) for l in lines]
        gpu_proc_name = [gpn for gpn in gpu_proc_name if gpn is not None]
        gpu_proc_name = [(gpn.gpu, gpn.proc_name) for gpn in gpu_proc_name]
        gpu2proc_names = {idx: [pn for gpu,pn in gpu_proc_name if gpu == idx] for idx in range(host_info.total_gpus)}

        user2num_gpus = [f"{k}={host_info.user2num_gpus[k]}" for k in sorted(host_info.user2num_gpus.keys(), key=lambda k: host_info.user2num_gpus[k], reverse=True)]
        user2num_gpus = "(" + ", ".join(user2num_gpus) + ")" if len(user2num_gpus) > 0 else "()"
        return {idx: not any(["python" in pn for pn in proc_names]) for idx,proc_names in gpu2proc_names.items()}, user2num_gpus
    else:
        return {i: False for i in range(host_info.total_gpus)}, "()"

def find_free_solar_gpus():
    """Returns a string giving information about free GPUs on Solar."""
    node_infos = MachineInfo.SlurmNodeInfo.get_all_slurm_node_infos()

    for n in node_infos:
        print(n)
    
    four_large_gpu = sum([s.total_gpus // 4 for s in node_infos if s.total_gpus >= 4 and s.gpu_vram >= 40])
    four_large_gpu_free = sum([s.free_gpus_eff // 4 for s in node_infos if s.total_gpus >= 4 and s.gpu_vram >= 40])
    two_large_gpu = sum([s.total_gpus // 2 for s in node_infos if s.total_gpus >= 2 and s.gpu_vram >= 40])
    two_large_gpu_free = sum([s.free_gpus_eff // 2 for s in node_infos if s.total_gpus >= 2 and s.gpu_vram >= 40])

    four_med_gpu = sum([s.total_gpus // 4 for s in node_infos if s.total_gpus >= 4 and s.gpu_vram >= 24])
    four_med_gpu_free = sum([s.free_gpus_eff // 4 for s in node_infos if s.total_gpus >= 4 and s.gpu_vram >= 24])
    two_med_gpu = sum([s.total_gpus // 2 for s in node_infos if s.total_gpus >= 2 and s.gpu_vram >= 24])
    two_med_gpu_free = sum([s.free_gpus_eff // 2 for s in node_infos if s.total_gpus >= 2 and s.gpu_vram >= 24])

    four_gpu = sum([s.total_gpus // 4 for s in node_infos if s.total_gpus >= 4])
    four_gpu_free = sum([s.free_gpus_eff // 4 for s in node_infos if s.total_gpus >= 4])
    two_gpu = sum([s.total_gpus // 2 for s in node_infos if s.total_gpus >= 2])
    two_gpu_free = sum([s.total_gpus // 2 for s in node_infos if s.total_gpus >= 2])

    s = f"---- NUMBER OF RUN TYPES POSSIBLE ----\n"
    s += f"4xLarge GPU: {four_large_gpu_free}/{four_large_gpu}\n"
    s += f"2xLarge GPU: {two_large_gpu_free}/{two_large_gpu}\n"
    s += f"4xMedium GPU: {four_med_gpu_free}/{four_med_gpu}\n"
    s += f"2xMedium GPU: {two_med_gpu_free}/{two_med_gpu}\n"
    s += f"4xGPU: {four_gpu_free}/{four_gpu}\n"
    s += f"2xGPU: {two_gpu_free}/{two_gpu}\n"
    return s

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-c", "--current", "--current-machine-only", action="count", default=0,
        help="0: find information for all machines, 1: find information for all machines but print the current machine's info ASAP, 2: find information for only the current machine")
    P.add_argument("--hosts", type=str, nargs="*", default=MachineInfo.machine2info.keys(), choices=list(MachineInfo.machine2info.keys()),
        help="workstation/server names to check for free GPUs.")
    P.add_argument("--solar", type=int, default=2, choices=[0, 1, 2],
        help="0: always find information for workstations, 1: always find information for Solar, 2: find information for solar iff on solar")
    args = P.parse_args()

    if args.solar == 1 or (args.solar == 2 and Utils.is_solar()):
       print(find_free_solar_gpus())
    
    if args.solar == 0 or (args.solar == 2 and not Utils.is_solar()):
        args.hosts = MachineInfo.machine2info.keys() if len(args.hosts) == 0 else args.hosts
        args.hosts = [h for h in args.hosts if h not in MachineInfo.machines_cc and not h == "solar"]

        if args.current == 0:
            pass
        elif args.current == 1 or args.current == 2:
            current_machine = MachineInfo.to_machine_name(MachineInfo.get_current_machine())
            if current_machine in args.hosts:
                info = find_free_gpus(current_machine)
                print(f"Current machine {current_machine}: free={sum(info[0].values())}/{len(info[0])} IDs={[gpu_id for gpu_id,free in info[0].items() if free]} users={info[1]}")
            else:
                print(f"Current machine {current_machine} not in hosts to check, so not checking it.")
            
            args.hosts = [h for h in args.hosts if not h == current_machine]
        else:
            pass
        
        if args.current == 0 or args.current == 1:
            # Parallelizing this is ~5x faster
            with Pool(processes=min(16, len(args.hosts))) as p:
                infos = p.map(find_free_gpus, args.hosts, chunksize=math.ceil(len(args.hosts) / 16))
                machine2info = {h: info for h,info in zip(args.hosts, infos)}

            host2gpu2free = {h: gpu2free for h,(gpu2free, _) in machine2info.items()}
            host2users = {h: users_str for h,(_, users_str) in machine2info.items()}
            
            host2num_free = {h: sum(gpu2free.values()) for h,gpu2free in host2gpu2free.items()}
            host2total = {h: len(gpu2free) for h,gpu2free in host2gpu2free.items()}

            for h in sorted(host2gpu2free, key=lambda h: host2num_free[h], reverse=True):
                print(f"{h}: free={host2num_free[h]}/{host2total[h]} IDs={[gpu_id for gpu_id,free in host2gpu2free[h].items() if free]} users={host2users[h]}")
            
            print(f"Total free GPUs: {sum(host2num_free.values())}/{sum(host2total.values())}")