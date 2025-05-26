import MachineInfo

for m in MachineInfo.machine2info:

    hostname = MachineInfo.machine_to_hostname(m)
    if MachineInfo.hostname_is_current_machine(hostname):
        print(f"Skipping updating scripts and aliases on current machine {m}")
        continue
    else:
        print(f"Updating scripts and aliases on {m}")
        MachineInfo.run_command_on_machine(m, "rm -rf ~/.ScriptsAndAliases ; git clone https://github.com/tristanengst/ScriptsAndAliases ~/.ScriptsAndAliases ; python ~/.ScriptsAndAliases/WriteAliases.py")