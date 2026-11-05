#!/usr/bin/env python3
"""Logs a snapshot of SLURM state to a JSON database. Meant to be run periodically, eg.
hourly from cron over ssh.

The database is a directory with one JSON file per UTC day (eg. 2026-11-04.json), each a
dict mapping keys of the form f"{timestamp}{optional other info}" to dicts holding a
"timestamp" key plus whatever was requested, eg.
    "2026-11-04T17:00:03+00:00": {"timestamp": "2026-11-04T17:00:03+00:00",
                                  "levelfs_rrg_keli": 0.781881}

Writes use UtilsBase's atomic JSON saving, so a day's file is never corrupted. Runs at the
same moment record essentially the same state, so if they race one may replace the other.

Usage:
    python3 LogSLURMState.py --levelfs_rrg_keli
    python3 LogSLURMState.py --levelfs_rrg_keli --database_path /path/to/db
"""
import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess
import sys

import UtilsBase
from UtilsBase import twrite

def get_account_levelfs_record(account, verbose=True):
    """Returns the account-level LevelFS of SLURM account [account]."""
    keys2query_keys = dict(account="Account", user="User", levelfs="LevelFS", rawusage="RawUsage", normusage="NormUsage")

    cmd = f"sshare -A {shlex.quote(account)} -nP --format {','.join(keys2query_keys.values())}"
    if verbose:
        print(f"Running {cmd}")
    try:
        out = subprocess.run(cmd, shell=True,
            capture_output=True, text=True, check=True, timeout=120).stdout
        lines = [line.split("|") for line in out.splitlines() if line.strip()]
        lines = [dict(zip(keys2query_keys.keys(), line)) for line in lines]
        account_line = [line for line in lines if line["user"].strip() == ""][0] # account-level line has empty user
        account_line = {k: v for k,v in account_line.items() if not k == "user"}
        twrite(account_line=account_line)
        account_line = {k: v if k in ["account"] else float(v) for k,v in account_line.items()}
        return account_line
    except subprocess.CalledProcessError as e:
        return dict(account=account, levelfs=None, rawusage=None, normusage=None)
    # except Exception as e:
    #     raise RuntimeError(f"Failed to get account-level LevelFS for {account}: {e}")

def log_record(*, args, record):
    """Atomically adds [record] to the day file for its timestamp. Returns its key."""
    _ = os.makedirs(osp.dirname(args.fpath), exist_ok=True)
    record = record | dict(timestamp=args.timestamp, key_suffix=args.key_suffix)
    key = f"{args.timestamp}" + (f"_{args.key_suffix}" if args.key_suffix else "")
    _ = UtilsBase.atomic_append_lite(data={key: record}, fname=args.fpath)
    return key

def get_args():
    P = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    P.add_argument("--database_path", default="/project/rrg-keli/slurm_tracking",
        help=f"Directory of the database")
    P.add_argument("--fpath", default=None,
        help="Path to file to log records to if given. If not absolute, relative to --database_path")
    P.add_argument("--levelfs_rrg_keli", action="store_true",
        help="Log the account-level LevelFS of rrg-keli_gpu as 'levelfs_rrg_keli'")
    P.add_argument("--key_suffix", default="",
        help="Suffix to add to the key for this record, eg. a hostname or job")
    
    args = P.parse_args()

    timestamp_dt = datetime.datetime.now(datetime.timezone.utc)
    timestamp = timestamp_dt.strftime("%Y-%m-%d-%H-%M-%S")
    timestamp_fp = timestamp_dt.strftime("%Y-%m-%d")
    fpath = args.fpath if args.fpath else f"levelfs_{timestamp_fp}.json"
    fpath = osp.join(args.database_path, fpath) if not osp.isabs(fpath) else fpath
    args = UtilsBase.updated_namespace(args,
        timestamp=timestamp, # Record the timestamp in one place so that the one we record matches the file, even if it actually executes slightly later
        fpath=fpath)
    
    return args


if __name__ == "__main__":
    args = get_args()
    record = dict()

    if args.levelfs_rrg_keli:
        levelfs_rrg_keli_record = get_account_levelfs_record("rrg-keli_gpu")
        record |= dict(levelfs_rrg_keli=levelfs_rrg_keli_record)

    key = log_record(args=args, record=record)
    twrite(f"Logged {key} to {args.fpath}: {record}")
