"""Prints a list of jobs in the queue. If on ComputeCanada, limits to just the
rrg-keli and def-keli accounts.
"""
import os
import subprocess

if os.uname().nodename == "cs-star" or os.uname().nodename.startswith("cs-venus"):
    s = "squeue -O 'NodeList:.10,JobArrayID:.6,State:.9,tres-per-node:.12,Account:.15,Partition:.16,Name:.165,TimeLeft:.12,Reason:5' --sort N"
else:
    s = "squeue -A rrg-keli_gpu -O 'JobArrayID:11,UserName:6,State:9,tres-per-node:17,TimeLeft:12,Reason:20,Name:.160'; squeue -A def-keli_gpu  -O 'JobArrayID:11,UserName:6,State:9,tres-per-node:17,TimeLeft:12,Reason:20,Name:.160'"

print(subprocess.getoutput(s))