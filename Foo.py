import os
import MachineInfo

from MachineInfo import *

if machine_to_ssh_name(os.uname()[1]) == "S1":
    print("AAAA")
else:
    print("BBBB")


import argparse
P = argparse.ArgumentParser()
P.add_argument("--file",)
args = P.parse_args()

f = args.file


def compress_user(f, cwd=osp.expanduser("~")):
    
    f_abspath = osp.abspath(osp.expanduser(f))

    f_relpath = osp.relpath(f_abspath, cwd)

    print(f"{f} -> {f_abspath} -> {f_relpath}")


compress_user(f)