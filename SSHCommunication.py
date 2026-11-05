"""File for allowing inter-machine communication via SSH.

A key challege is that figuring out what other machines are actually called is
surprisingly nontrivial given that we can't control how it's done incredibly weirdly
sometimes.

Hostnames of ComputeCanada clusters are public and stored here. Those of other machines
(workstations, Solar) live in the private submodule at Private/MachineHostnames.py, which
isn't in the public repo; without it, those machines are reached by their SSH names (eg.
~/.ssh/config aliases) instead.

Paramiko would be cleaner, but violates the no non-standard-library desiderata.
"""
import functools
import os
import os.path as osp
import socket
import subprocess

from UtilsBase import twrite

######################################################################################
# Machine names to hostnames. Public ones are here, and private ones are merged in from
# the private submodule if it's checked out.
######################################################################################
machine2hostname = dict(
    narval="narval.alliancecan.ca",
    trillium="trillium-gpu.alliancecan.ca",
    fir="fir.alliancecan.ca",
    rorqual="rorqual.alliancecan.ca",
    nibi="nibi.alliancecan.ca",
    vulcan="vulcan.alliancecan.ca",
    killarney="killarney.alliancecan.ca",
    killa="killarney.alliancecan.ca",
    tamia="tamia.alliancecan.ca",)
try:
    import Private.MachineHostnames
    machine2hostname |= Private.MachineHostnames.machine2hostname
    has_private_hostnames = True
except ImportError:
    has_private_hostnames = False

######################################################################################
######################################################################################
######################################################################################

class HostInfoError(Exception):
    """Custom exception for HostInfo-related errors."""
    pass

@functools.cache
def get_hostname():
    result = socket.getfqdn()
    if not "local" in result and "." in result:
        return result

    ##################################################################################
    # Annoying alternative methods to get the hostname in case the code above fails,
    # probably because the system isn't configured quite right in the first place. If
    # any fail, the original result is returned and we hope for the best.
    ##################################################################################
    # Method 1: read IP address from terminal and pass to gethostbyaddr()
    elif not "local" in result and not "." in result:
        ip_address = subprocess.getoutput("ip route get 1.1.1.1 | grep -oP 'src \K[0-9.]+'")
        socket_hostname = socket.gethostbyaddr(ip_address)[0]
        if not "local" in socket_hostname and "." in socket_hostname:
            return socket_hostname

    twrite(f"[WARNING] get_hostname() got unexpected result: {result}")
    return result

@functools.cache
def hostname_to_machine(h):
    """Returns the machine name corresponding to hostname [h]. If several machine names
    share [h] (eg. an alias), the first one in [machine2hostname] is returned.
    """
    for m,v in machine2hostname.items():
        if h == v:
            return m
    twrite(f"[WARNING] hostname_to_machine() couldn't find a machine for hostname={h}{'' if has_private_hostnames else ' (Private submodule not checked out?)'}")
    return None

@functools.cache
def get_machine_name(): return hostname_to_machine(get_hostname())

@functools.cache
def check_connection(machine, *, ssh_args=()):
    """Returns if [machine] can be SSHed to, passing [ssh_args] to SSH. [ssh_args] is a
    tuple because caching requires hashable function arguments.
    """
    cmd = f"ssh -o ConnectTimeout=1 -o BatchMode=yes {' '.join(ssh_args)} {machine} true"
    try:
        return subprocess.run(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except subprocess.TimeoutExpired:
        return False

@functools.cache
def to_ssh_name(machine):
    """Returns the best SSHable name of [machine]. This is ideally an alias defined
    by ~/.ssh/config, but if not, it will be the FQDN of the machine if it's known,
    or just [machine] if it's not.
    """
    hostname = machine2hostname.get(machine, machine) # FQDN
    
    ##################################################################################
    # Ideally, we can map the hostname to something in ~/.ssh/config, which may better
    # specify how to SSH to it. Otherwise, we can just use the hostname, since it
    # should be an FQDN. It may even be that [machine] is an alias in ~/.ssh/config,
    # in which case we can just use it directly.
    ##################################################################################
    ssh_conf_alias2hostname = get_ssh_conf_alias2hostname()
    if machine in ssh_conf_alias2hostname:
        return machine
    else:
        aliases = [a for a,h in ssh_conf_alias2hostname.items() if h == hostname]
        if aliases:
            # Handle the case of multiple aliases for the same hostname by returning the shortest one
            aliases = sorted(aliases, key=lambda x: len(x))
            return aliases[0]
        else:
            return hostname

def run_command_on_machine(*, machine, command, ssh_args=[], if_connect_error="error"):
    """Runs [command] on machine [m] and returns the output.
    
    Args:
    machine             -- machine name in [machine2hostname], or SSHable thing to run
                            command on
    command             -- command to run on machine
    ssh_args            -- additional arguments passed to SSH (probably none needed)
    if_connect_error    -- Something returned on connection error, or "HostInfoError"
                            to raise one
    """
    hostname = machine2hostname.get(machine, machine)
    if hostname == get_hostname():
        target = None
    else:
        target = to_ssh_name(machine)
        if check_connection(target, ssh_args=tuple(ssh_args)):
            pass
        elif if_connect_error == "HostInfoError":
            raise HostInfoError(f"Can't SSH to machine={machine} (hostname={hostname}) via {target}")
        else:
            return if_connect_error() if callable(if_connect_error) else if_connect_error

    cwd = os.getcwd()
    os.chdir("/") # This fixes an issue at one point; not sure why.
    try:
        return subprocess.getoutput(command if target is None else f"ssh {' '.join(ssh_args)} {target} '{command}'")
    finally:
        os.chdir(cwd)

@functools.cache
def get_ssh_conf_alias2hostname():
    """Returns a dict mapping aliases (Host) to hostnames (HostName) by parsing the
    ~/.ssh/config file on the current machine.
    """
    if not osp.exists(osp.expanduser("~/.ssh/config")):
        return dict()
    
    with open(osp.expanduser("~/.ssh/config"), "r") as f:
        lines = f.readlines()
        lines = [l.strip().split("#", 1)[0] for l in lines] # Remove comments
    
    ssh_machine2hostname = dict()
    cur_hosts = None
    for l in lines:
        if l.startswith("Host "):
            cur_hosts = l.strip().split()[1:]
            cur_hosts = [h for h in cur_hosts if not h.startswith("*")]
        elif l.startswith("HostName ") and not cur_hosts is None:
            for h in cur_hosts:
                ssh_machine2hostname[h] = l.split()[1]
            cur_hosts = None
        else:
            continue
        
    return ssh_machine2hostname

if __name__ == "__main__":
    twrite(machine2hostname=machine2hostname)
    twrite(current_machine=get_machine_name())
