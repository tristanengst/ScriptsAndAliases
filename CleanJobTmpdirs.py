"""Cleans stale per-job scratch directories on Solar's compute nodes.

On each compute node we expect /localscratch/$USER/job_tmpdirs to contain one
folder per job, named by its SLURM job ID. A folder is stale if (1) its job ID
is no longer recorded by squeue (regardless of state) and (2) it is older than
3 minutes (this grace period avoids deleting a brand-new job's directory before
squeue has registered it). Stale folders are removed.

This script is self-contained so it can be copied to a node and run there with
no access to the rest of this repo. No node is assumed to see any other node's
filesystem, so the controller (run on the Solar login node) ships this very
script to each compute node by base64-encoding its source into the srun command
itself, which decodes it to /localscratch/$USER and runs it in --worker mode.
The worker also reports free space (GB) under /localscratch, /home,
/localscratch/$USER, and /home/$USER.

Usage (on the Solar login node):
    python3 CleanJobTmpdirs.py
    python3 CleanJobTmpdirs.py --nodes cs-venus-05 cs-venus-06
"""
import argparse
import base64
import os
import os.path as osp
import shutil
import subprocess
import time
from types import SimpleNamespace


################################################################################
# Worker: runs on a single compute node.
################################################################################
def squeue_to_job_ids():
    """Returns a (ok, ids) tuple where [ids] is the set of all job IDs squeue
    currently records and [ok] is whether squeue ran successfully. When squeue
    fails we report [ok]=False so the caller deletes nothing.
    """
    try:
        result = subprocess.run(["squeue", "-h", "-o", "%i %A %F"],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"[worker] squeue failed ({e}); will not delete anything")
        return False, set()
    if not result.returncode == 0:
        print(f"[worker] squeue returned {result.returncode}; will not delete anything\n{result.stderr.strip()}")
        return False, set()
    return True, set(result.stdout.split())

def path_to_free_used(fpath):
    """Returns a (free, used, total) tuple of free/used/total space under [fpath]."""
    try:
        usage = shutil.disk_usage(fpath)
        return SimpleNamespace(free=usage.free / 2 ** 30, used=usage.used / 2 ** 30, total=usage.total / 2 ** 30)
    except OSError:
        return SimpleNamespace(free="?", used="?", total="?")

def run_worker(args):
    """Removes stale job folders under /localscratch/$USER/job_tmpdirs and
    prints free-space info for this node.
    """
    user = os.environ["USER"]
    node = os.environ.get("SLURMD_NODENAME", os.uname().nodename)
    tmpdir = f"/localscratch/{user}/job_tmpdirs"

    localscratch_usage = path_to_free_used("/localscratch")
    localscratch_user_usage = path_to_free_used(f"/localscratch/{user}")
    home_usage = path_to_free_used("/home")
    home_user_usage = path_to_free_used(f"/home/{user}")
    disk_usage_str = f"/localscratch: (free={localscratch_usage.free:.1f}GB, used by {user}={localscratch_user_usage.used:.1f}GB) /home: (free={home_usage.free:.1f}GB, used by {user}={home_user_usage.used:.1f}GB)"
    print(f"[worker] node={node} {disk_usage_str}")

    if not osp.isdir(tmpdir):
        print(f"[worker] {tmpdir} does not exist -> nothing to do")
        return

    folders = [osp.join(tmpdir, f) for f in os.listdir(tmpdir) if osp.isdir(osp.join(tmpdir, f))]
    if len(folders) == 0:
        print(f"[worker] {tmpdir} is empty -> nothing to do")
        return

    ok, job_ids = squeue_to_job_ids()
    now = time.time()
    for f in folders:
        if osp.basename(f) in job_ids:
            print(f"[worker] keep {f}: in squeue")
        elif not ok:
            print(f"[worker] keep {f}: squeue unavailable -> not deleting anything")
        elif now - osp.getmtime(f) < args.stale_age:
            age = int(now - osp.getmtime(f))
            print(f"[worker] keep {f}: not in squeue but only {age}s old (< {args.stale_age}s)")
        elif args.dry_run:
            print(f"[worker] would remove {f}: not in squeue and stale (dry_run=True)")
        else:
            try:
                _ = shutil.rmtree(f)
                print(f"[worker] removed {f}: not in squeue and stale")
            except OSError as e:
                print(f"[worker] failed to remove {f}: {e}")

################################################################################
# Controller: runs on the login node, dispatches the worker to each node.
################################################################################
def get_solar_nodes():
    """Returns the list of allocatable Solar compute node names."""
    from MachineInfo import cluster2node2config
    node2config = cluster2node2config["solar"]
    return [n for n, c in node2config.items() if c["can_allocate"]]

def run_controller(args):
    """Sruns the worker on each requested node, relaying its output."""
    user = os.environ["USER"]
    remote = osp.join(f"/localscratch/{user}", osp.basename(__file__))
    nodes = args.nodes if len(args.nodes) > 0 else get_solar_nodes()

    # No shared filesystem is assumed, so embed this script's source as base64
    # in the srun command; the node decodes it to local scratch and runs it.
    with open(osp.abspath(__file__), "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    inner = f"mkdir -p /localscratch/{user} && echo {b64} | base64 -d > {remote} && python3 {remote} --worker --stale_age={args.stale_age} {'--dry-run' if args.dry_run else ''}"

    for node in nodes:
        print(f"\n=== {node} ===")
        cmd = ["srun", f"--nodelist={node}", "--time=5:00",
            "--ntasks-per-node=1", "--cpus-per-task=1", "--mem=2G",
            f"--immediate={args.max_wait}", "bash", "-c", inner]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True,
                timeout=args.max_wait)
        except subprocess.TimeoutExpired:
            print(f"[controller] {node}: timed out")
            continue
        if not result.returncode == 0:
            print(f"[controller] {node}: srun failed (rc={result.returncode}), likely could not allocate within {args.max_wait}s")
            if result.stderr.strip():
                print(result.stderr.strip())
        if result.stdout.strip():
            print(result.stdout.strip())

def get_args():
    P = argparse.ArgumentParser()
    P.add_argument("--worker", action="store_true",
        help="Internal: run the per-node cleanup here instead of dispatching")
    P.add_argument("--nodes", nargs="+", default=[],
        help="Subset of node names to clean (default: all allocatable Solar nodes)")
    P.add_argument("--dry-run", action="store_true",
        help="Do not actually delete anything, just print what would be done")
    P.add_argument("--max_wait", type=int, default=20,
        help="Seconds to wait for an srun allocation on a node before giving up")
    P.add_argument("--stale_age", type=int, default=300,
        help="Seconds a job folder must be old before deletion")
    return P.parse_args()

if __name__ == "__main__":
    args = get_args()
    _ = run_worker(args) if args.worker else run_controller(args)
