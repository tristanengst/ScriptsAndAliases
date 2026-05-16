import argparse
import os.path as osp
import subprocess

import SSHCommunication
import Utils
from UtilsBase import twrite, tqdm

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("update_on", nargs="*", default=None,
        help="Which machines to update on. If empty, interpreted as the current machine. If 'all', interpreted as every SSH-able machine. Other values are interpreted as giving the machine to update on.")
    args = P.parse_args()


    if args.update_on is None:
        args.update_on = [SSHCommunication.get_current_machine()]
        twrite(f"[INFO] No machines specified to update on, so defaulting to current machine: {args.update_on}")
    elif "all" in args.update_on:
        args.update_on = SSHCommunication.get_machine_name_to_hostname_map_all().values()
        twrite(f"[INFO] will update on all SSH-able machines with SSH names: {args.update_on}")
    else:
        pass

    for u in tqdm(args.update_on):
        twrite("-" * 80)
        twrite(f"Updating host={u}...")
        try:
            result = SSHCommunication.run_command_on_machine(machine=u,
                command="bash -ic \"cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py ; source ~/.bashrc\"",
                if_connect_error="HostInfoError",
                if_ssh_map_error="HostInfoError",
            )
        except Exception as e:
            twrite(f"Error updating host={u}: {e}")
            continue
        twrite(f"Result of updating host={u}:\n{result}")