"""File for allowing inter-machine communication via SSH.

A key challege is that figuring out what other machines are actually called is
surprisingly nontrivial given that we can't control how it's done incredibly weirdly
sometimes.

NOMENCLATURE:
1. A "machine" is a system we want to log into. Therefore, it might be a cluster or a
    just a single node. This is the useful thing we actually want to type.
2. An "ssh name" is something we can SSH to, but it's not immediately clear what the
    corresponding machine is or even hostname is.
2. A "hostname" is the actual hostname of the machine. If we were to SSH into it, it
    would *definitely* work providing that SSH was enabled, etc. Conversely, SSHing
    to a machine would work only with the appropriate ~/.ssh/config file.
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

######################################################################################
# Super basic encryption scheme for storing SSH info publically on GitHub. We only 
# keep hostnames here; nothing super interesting.
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
    """Reads the encripted SSH info from [fpath] and returns it as a dict."""
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
        return json.loads(decrypted_info) if not decrypted_info is None else dict()

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

def to_hostname(x):
    """Converts [x], which can be a machine name, ssh name, or hostname, to a hostname."""
    if x in machine2hostname:
        return machine2hostname[x]
    decrypted_m2h = read_encrypted_machine_to_hostname_info()
    if x in decrypted_m2h:
        return decrypted_m2h[x]
    return x # Assume it's already a hostname

def hostname_is_current_machine(h):
    """Returns if [hostname] corresponds to the current machine."""
    return socket.getfqdn() == h

def check_connection(machine):
    cmd = f"ssh -o ConnectTimeout=1 -o BatchMode=yes {machine} exit"
    result = subprocess.getoutput(cmd)
    return result == ""

def run_command_on_machine(*, machine, command, ssh_args=[], if_connect_error="error", if_ssh_map_error="error", **ssh_kwargs):
    """Runs [command] on machine [m] and returns the output."""
    cwd = os.getcwd()
    os.chdir("/") # Not sure why this fixes an issue. Need to change back to the normal directory after running the command
    
    target_hostname = to_hostname(machine)
    if hostname_is_current_machine(target_hostname):
        result = subprocess.getoutput(command)
        os.chdir(cwd)
        return result
    


    
    
    if os.uname().nodename == to_hostname(machine) or machine is None:
        result = subprocess.getoutput(command)
        os.chdir(cwd)
        return result
    
    
    ssh_name = to_ssh_name(machine)
    if ssh_name is None:
        if if_ssh_map_error == "HostInfoError":
            raise HostInfoError(f"Could not find SSH name for machine {machine}. Please check your ~/.ssh/config file.")
        else:
            return if_ssh_map_error() if callable(if_ssh_map_error) else if_ssh_map_error
    else:
        if not check_connection(ssh_name):
            return if_connect_error() if callable(if_connect_error) else if_connect_error

        ssh_args_str = " ".join(ssh_args)
        command_to_run = f"ssh {ssh_args_str} {ssh_name} '{command}'"
        result = subprocess.getoutput(command_to_run)
        os.chdir(cwd)
        return result




if __name__ == "__main__":

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

    def get_machine_name_to_hostname_map_all():
        """Returns a dict mapping machine names to hostnames by using the known
        structure of hostnames in the CC domain.
        """
        return get_machine_name_to_hostname_map_by_ssh_conf() | machine2hostname


    def test_encryption():
        """Tests the encryption and decryption functions."""
        # to_encrypt = json.dumps(get_machine_name_to_hostname_map_all())
        # _ = write_encrypted_info(to_encrypt)
        decrypted_info = read_encrypted_info()
        print(decrypted_info)







    print(get_machine_name_to_hostname_map_all())

    test_encryption()



