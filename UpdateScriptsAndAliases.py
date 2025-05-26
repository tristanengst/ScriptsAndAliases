import MachineInfo

for m in MachineInfo.machine2info:
    print(f"Updating scripts and aliases on {m}")
    MachineInfo.run_command_on_machine(m, "cd ~/.ScriptsAndAliases ; git pull")