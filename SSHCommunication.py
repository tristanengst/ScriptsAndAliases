"""File for allowing inter-machine communication via SSH.

A key challege is that figuring out what other machines are actually called is
surprisingly nontrivial given that we can't control how it's done incredibly weirdly
sometimes.

Paramiko violates the non-standard-library requirement.
"""
import argparse
import base64
from collections import defaultdict
import functools
import json
import hashlib
import os
import os.path as osp
import socket
import subprocess
import sys
import time

import UtilsBase
from UtilsBase import twrite

######################################################################################
# Super basic encryption scheme for storing SSH info publically on GitHub. This make
# sharing the info among machines that should have access maximally easy without
# storing it in plaintext.
######################################################################################

def encrypt(key, to_encrypt):
    """Encrypts a string using a key and returns a base64 string."""
    key_bytes = hashlib.sha256(key.encode().strip().lower()).digest()
    data_bytes = to_encrypt.encode()
    # XOR each byte of data with the key hash
    processed = bytes(data_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data_bytes)))
    return base64.b64encode(processed).decode()

def decrypt(key, encrypted):
    """Decrypts a base64 string using a key."""
    try:
        key_bytes = hashlib.sha256(key.encode().strip().lower()).digest()
        data_bytes = base64.b64decode(encrypted)
        # XORing the encrypted data with the same key restores the original
        processed = bytes(data_bytes[i] ^ key_bytes[i % len(key_bytes)] for i in range(len(data_bytes)))
        return processed.decode()
    except Exception:
        return None

@functools.cache
def read_encrypted_machine_to_hostname_info(fpath=osp.join(osp.dirname(__file__), "ssh_info.enc")):
    """Returns the SSH info encryped at [fpath] and returns it as a dict. If this
    fails, returns an empty dict.
    """
    hostname = socket.getfqdn()
    hostname_parts = hostname.split(".", 1)
    if len(hostname_parts) == 1 or hostname_parts[1] == "local":
        return dict() # Can't figure out the domain name, so just return an empty dict
    else:
        prefix = hostname_parts[0].split("-")
        prefix[-1] = prefix[-1][-1] # Keep only last character
        key = "-".join(prefix) + hostname_parts[1]
        encrypted_info = UtilsBase.load_file_lite(fpath).strip()
        decrypted_info = decrypt(key, encrypted_info)
        # Need to call json.loads() twice. asdfgh
        result = json.loads(json.loads(decrypted_info)) if decrypted_info else dict()
        return result

def write_encrypted_machine_to_hostname_info(info, fpath=osp.join(osp.dirname(__file__), "ssh_info.enc")):
    """Writes the SSH info in [info] to [fpath] in an encrypted format."""
    hostname = socket.getfqdn()
    hostname_parts = hostname.split(".", 1)
    if len(hostname_parts) == 1 or hostname_parts[1] == "local":
        raise ValueError("Can't figure out the domain name, so can't write encrypted info.")
    else:
        def map_to_alphabet_modulo(text):
            result =[(ord(char.upper()) - ord('A') + 1) % 2 for char in text if char.isalpha()][-4:]
            return "".join([str(num) for num in result])
            
        prefix = hostname_parts[0].split("-")
        prefix[-2] = map_to_alphabet_modulo(prefix[-2])
        prefix[-1] = prefix[-1][-1] # Keep only last character
        key = "-".join(prefix) + hostname_parts[1]

        info_str = json.dumps(info)
        encrypted_info = encrypt(key, info_str)
        _ = UtilsBase.atomic_save_lite(data=encrypted_info, fpath=fpath)

        with open(fpath, "w") as f:
            f.write(encrypted_info)

######################################################################################
######################################################################################
######################################################################################

######################################################################################
# Some data can be stored in plaintext since it's not super sensitive
######################################################################################

machine2hostname = dict(
    narval="narval.alliancecan.ca",
    trillium="trillium-gpu.alliancecan.ca",
    fir="fir.alliancecan.ca",
    rorqual="rorqual.alliancecan.ca",
    nibi="nibi.alliancecan.ca",
    vulcan="vulcan.alliancecan.ca",
    killarney="killarney.alliancecan.ca",
    tamia="tamia.alliancecan.ca",)

machine2node_prefixes = dict(
    apex=["cs-a"],
    solar=["cs-s", "cs-v", "cs-b"],
    narval=["narval"],
    trillium=["trillium", "trig"],
    fir=["fc", "login"],
    rorqual=["rorqual", "rq", "rg", "rl"],
    nibi=["nibi", "g"],
    vulcan=["vulcan", "rack"],
    killarney=["klogin", "kn"],
    tamia=["tamia", "tc", "tg"],
)

######################################################################################
######################################################################################
######################################################################################

class HostInfoError(Exception):
    """Custom exception for HostInfo-related errors."""
    pass

@functools.cache
def get_hostname(): return socket.getfqdn()

@functools.cache
def hostname_to_machine(h):
    """Returns the machine name corresponding to hostname [h]."""
    def get_machine_from_machine_to_hostname_map(m2h):
        found_hostnames = []
        for k,v in m2h.items():
            _ = found_hostnames.append(k) if h == v else None
        if found_hostnames:
            return sorted(found_hostnames, key=lambda x: len(x))[-1]
        else:
            return None

    machine_from_known_m2h = get_machine_from_machine_to_hostname_map(machine2hostname)
    if not machine_from_known_m2h is None:
        return machine_from_known_m2h
    decrypted_m2h = read_encrypted_machine_to_hostname_info()
    machine_from_decrypted_m2h = get_machine_from_machine_to_hostname_map(decrypted_m2h)
    return machine_from_decrypted_m2h if not machine_from_decrypted_m2h is None else None

@functools.cache
def get_machine_name(): return hostname_to_machine(get_hostname())

@functools.cache
def machine_to_hostname(m):
    """Returns the hostname corresponding to machine [m]."""
    # First, try reading data from stuff we can store in plaintext
    node_prefix = get_hostname().split(".", 1)[0]
    machine_type = None
    for mtype, node_prefixes in machine2node_prefixes.items():
        for np in node_prefixes:
            if node_prefix.startswith(np):
                machine_type = mtype
                break
    if not machine_type is None and machine_type in machine2hostname:
        return machine2hostname[machine_type]

    ssh_info = read_encrypted_machine_to_hostname_info()
    return ssh_info[m] if m in ssh_info else None

@functools.cache
def to_hostname(x, allow_if_can_ssh=True):
    """Returns the hostname corresponding to [x], or [x] if's ssh-able and
    [allow_if_can_ssh] is True, or None.
    """
    if x in machine2hostname:
        return machine2hostname[x]
    decrypted_m2h = read_encrypted_machine_to_hostname_info()
    if x in decrypted_m2h:
        return decrypted_m2h[x]
    elif allow_if_can_ssh and check_connection(x):
        return x
    else:
        return None

@functools.cache
def hostname_is_current_machine(h):
    """Returns if hostname [h] corresponds to the current machine."""
    return (socket.getfqdn() == h) or (os.uname().nodename == h)

@functools.cache
def check_connection(machine):
    """Returns if [machine] can be SSHed to."""
    cmd = f"ssh -o ConnectTimeout=1 -o BatchMode=yes {machine} exit"
    result = subprocess.getoutput(cmd)
    return result == ""

def run_command_on_machine(*, machine, command, ssh_args=[], if_connect_error="error", if_ssh_map_error="error"):
    """Runs [command] on machine [m] and returns the output.
    
    Args:
    machine             -- machine or SSHable thing to run command on
    command             -- command to run on machine
    ssh_args            -- additional arguments passed to SSH (probably none needed)
    if_connect_error    -- Something returned on connection error
    if_ssh_map_error    -- Something returned if we can't find the hostname
    """
    cwd = os.getcwd()
    os.chdir("/") # This fixes an issue at one point; not sure why.
    
    target_hostname = to_hostname(machine, allow_if_can_ssh=False)
    if target_hostname is None:
        if if_ssh_map_error == "HostInfoError":
            raise HostInfoError(f"SSHable name for machine={machine} unknown")
        else:
            return if_ssh_map_error() if callable(if_ssh_map_error) else if_ssh_map_error
    elif hostname_is_current_machine(target_hostname):
        result = subprocess.getoutput(command)
        os.chdir(cwd)
        return result
    elif not check_connection(target_hostname):
        if if_connect_error == "HostInfoError":
            raise HostInfoError(f"Found hostname={target_hostname} for machine={machine}, but can't SSH")
        else:
            return if_connect_error() if callable(if_connect_error) else if_connect_error
    else:
        ssh_args_str = " ".join(ssh_args)
        command_to_run = f"ssh {ssh_args_str} {target_hostname} '{command}'"
        result = subprocess.getoutput(command_to_run)
        os.chdir(cwd)
        return result

@functools.cache
def get_machine_name_to_hostname_map_by_ssh_conf():
    """Returns a dict mapping machine names to hostnames by parsing the
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

@functools.cache
def get_machine_name_to_hostname_map_all(include_cc=True):
    """Returns a dict mapping machine names to hostnames by using the known
    structure of hostnames in the CC domain.
    """
    result = get_machine_name_to_hostname_map_by_ssh_conf() | machine2hostname
    result = result if include_cc else {m: h for m,h in result.items() if not h.endswith("alliancecan.ca")}
    return result

@functools.cache
def get_all_usable_ssh_names(include_cc=True):
    """Returns a list of all machine names that we can SSH to."""
    machine2hostname = get_machine_name_to_hostname_map_all()
    machine2hostname = machine2hostname if include_cc else {m: h for m,h in machine2hostname.items() if h.endswith("alliancecan.ca")}
    machine2hostname = {m: h for m,h in machine2hostname.items() if check_connection(h)}
    return machine2hostname.values()

if __name__ == "__main__":

    def test_encryption():
        """Tests the encryption and decryption functions."""
        # to_encrypt = json.dumps(get_machine_name_to_hostname_map_all())
        # _ = write_encrypted_info(to_encrypt)
        decrypted_info = read_encrypted_info()
        print(decrypted_info)

    print(get_machine_name_to_hostname_map_all())
    test_encryption()



