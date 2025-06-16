import argparse
import os.path as osp
import subprocess

import MachineInfo

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--recursive_call", default=0, type=int, choices=[0, 1],
        help="If 1, then this script was made to run from another machine and should not attempt to update other machines.")
    args = P.parse_args()

for m in MachineInfo.machine2info:
    hostname = MachineInfo.machine_to_hostname(m)
    
    if MachineInfo.hostname_is_current_machine(hostname):
        print(f"Updating host={hostname} (current machine)...")
        if not osp.exists(osp.expanduser("~/.ScriptsAndAliases")):
            MachineInfo.run_command_on_machine(m, "git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases")
        
        MachineInfo.run_command_on_machine(m, "cd ~/.ScriptsAndAliases ; git pull ; python ~/.ScriptsAndAliases/WriteAliases.py")

        if osp.exists(osp.expanduser("~/.bashrc")):
            MachineInfo.run_command_on_machine(m, "source ~/.bashrc",)
        if osp.exists(osp.expanduser("~/.zshrc")):
            MachineInfo.run_command_on_machine(m, "source ~/.zshrc",)
    
    elif not args.recursive_call:
        print(f"Updating host={hostname}...")
        MachineInfo.run_command_on_machine(m, "python ~/.ScriptsAndAliases/UpdateScriptsAndAliases.py",)
    else:
        pass