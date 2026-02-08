import argparse
import os.path as osp
import subprocess

import MachineInfo
import Utils
from UtilsBase import twrite, tqdm_lite

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("update_on", nargs="*", default=None,
        help="Which machines to update on. If empty, interpreted as the current machine. If 'all', interpreted as every SSH-able machine. Other values are interpreted as giving the machine to update on.")
    args = P.parse_args()

    args.update_on = [MachineInfo.get_current_machine()] if args.update_on is None else args.update_on
    if "all" in args.update_on:
        args.update_on = MachineInfo.get_all_usable_ssh_names()
        twrite(f"[INFO] will update on all SSH-able machines with SSH names: {update_on}")
    else:
        update_on2ssh_name = {m: MachineInfo.to_ssh_name(m) for m in args.update_on}
        twrite(f"[INFO] will update machines using machine-to-SSH-name mapping: {update_on2ssh_name}")
        args.update_on = update_on2ssh_name.values()

    for u in tqdm_lite(args.update_on):
        twrite("-" * 80)
        twrite(f"Updating host={u}...")
        result = MachineInfo.run_command_on_machine(machine=u,
            command="bash -ic \"cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py ; source ~/.bashrc\"")
        twrite(f"Result of updating host={u}:\n{result}")