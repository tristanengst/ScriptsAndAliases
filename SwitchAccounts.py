import argparse
import os

P = argparse.ArgumentParser()
P.add_argument("--job", nargs="+", default=[])
P.add_argument("--account", choices=["def", "rrg"], default="def")
args = P.parse_args()

account_to_account_str = {
    "def": "def-keli_gpu",
    "rrg": "rrg-keli_gpu"
}

for j in args.job:
    s = f"scontrol update job {j} Account={account_to_account_str[args.account]}"
    os.system(s)