import argparse
import os

P = argparse.ArgumentParser()
P.add_argument("cmd", help="scontrol command like TimeLimit=8:00:00")
P.add_argument("jobs", nargs="+", help="sequence of job ids to update")
args = P.parse_args()
for j in args.jobs:
	os.system(f"scontrol update job {j} {args.cmd}")
