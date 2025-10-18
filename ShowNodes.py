"""Nicer way of showing nodes."""
import argparse
import os
import os.path as osp
import subprocess
import sys

import Utils
from UtilsBase import twrite

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-n", "--nodes", nargs="+", default=[], help="If set, only show these nodes.")
    args = P.parse_args()

    if len(args.nodes) == 0:
        twrite(subprocess.getoutput("scontrol show nodes"))
        sys.exit(0)

    node2scontrol_info = dict()
    for n in args.nodes:
        cmd = f"scontrol show node {n}"
        output = subprocess.getoutput(cmd)
        node2scontrol_info[n] = output
    for n,info in node2scontrol_info.items():
        twrite(info)
    