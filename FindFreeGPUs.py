"""Pretty prints information on available GPUs. Heuristically, GPUs that aren't
running a process with 'python' in the name are available.
"""
import argparse
import math
import subprocess
import MachineInfo
import Utils

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
        l = l.split()
        proc_id = proc_id=l[4]

        return argparse.Namespace(gpu=int(l[1]), proc_name=l[6], )

    host_info = MachineInfo.get_updated_machine_info(h)

    if host_info.nvidia_smi_ok:
        lines = host_info.nvidia_smi.split("\n")
        eq_line_idxs = [idx for idx,l in enumerate(lines) if l.startswith("|=")]
        idx_of_last_eq_line = eq_line_idxs[-1]
        lines = lines[idx_of_last_eq_line+1:-1]

        gpu_proc_name = [gpu_line_to_data(l) for l in lines]
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
    P.add_argument("--hosts", type=str, nargs="*", default=MachineInfo.machine2info.keys(), choices=list(MachineInfo.machine2info.keys()),
        help="workstation/server names to check for free GPUs.")
    P.add_argument("--solar", type=int, default=2, choices=[0, 1, 2],
        help="0: always find information for workstations, 1: always find information for Solar, 2: find information for solar iff on solar")
    args = P.parse_args()

    if args.solar == 1 or (args.solar == 2 and Utils.is_solar()):
       print(find_free_solar_gpus())
    
    if args.solar == 0 or (args.solar == 2 and not Utils.is_solar()):
        args.hosts = MachineInfo.machine2info.keys() if len(args.hosts) == 0 else args.hosts
        
        machine2info = {h: find_free_gpus(h) for h in tqdm(args.hosts)}
        host2gpu2free = {h: gpu2free for h,(gpu2free, _) in machine2info.items()}
        host2users = {h: users_str for h,(_, users_str) in machine2info.items()}
        
        host2num_free = {h: sum(gpu2free.values()) for h,gpu2free in host2gpu2free.items()}
        host2total = {h: len(gpu2free) for h,gpu2free in host2gpu2free.items()}

        for h in sorted(host2gpu2free, key=lambda h: host2num_free[h], reverse=True):
            print(f"{h}: free={host2num_free[h]}/{host2total[h]} IDs={[gpu_id for gpu_id,free in host2gpu2free[h].items() if free]} users={host2users[h]}")
        
        print(f"Total free GPUs: {sum(host2num_free.values())}/{sum(host2total.values())}")