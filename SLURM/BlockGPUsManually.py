"""Makes sure each given GPU has a running manual blocker, <machine>_blocker_manual_<IDX> (eg.
A99_blocker_manual_1), one job per GPU, in the block-manual partition: submits it if there's
none, and releases it if it's held. Nothing else releases or holds them: give a GPU back with
`scontrol requeuehold JOBID` (rerun this to block it again), or `scancel -n NAME`. Each job's
comment is its name. Aliased as block_gpus on the APEX cluster (WriteAliases.py). See README.md,
"Blocking GPUs by hand".

Standard library only, so any python3 runs it.
"""
import argparse
import os
import os.path as osp
import re
import socket
import subprocess
import sys

SLURM_DIR = osp.dirname(osp.abspath(__file__))
SLURM_HOME = osp.expanduser("~/.local/slurm")
GRES_CONF = osp.join(SLURM_HOME, "etc", "gres.conf")   # lists every node's GPUs
HOSTNAMES = osp.join(SLURM_DIR, "..", "Private", "MachineHostnames.py")

# What env.sh puts on PATH: this SLURM's commands, ahead of any other SLURM.
os.environ["PATH"] = os.pathsep.join([osp.join(SLURM_HOME, "bin"), osp.join(SLURM_HOME, "sbin"),
    osp.join(SLURM_DIR, "bin"), os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")])
os.environ.pop("SLURM_CONF", None)

def machine2hostname():
    """machine name -> FQDN, from Private/MachineHostnames.py, or {} if it can't be read."""
    scope = dict()
    try:
        with open(HOSTNAMES) as f:
            exec(f.read(), scope)
    except OSError:
        return dict()
    return scope.get("machine2hostname", dict())

def machine_host(name, m2h):
    """[name]'s short hostname (A99 -> cs-apex-99s), or [name] if it isn't a machine name."""
    return m2h[name].split(".")[0] if name in m2h else name

def machine_name(host, m2h):
    """[host]'s machine name (cs-apex-99s -> A99), its first key in m2h, or [host] if none."""
    return next((m for m, h in m2h.items() if h.split(".")[0] == host), host)

def gpus_of(node):
    """The nvidia-smi indices [node] gives SLURM, as strings, in gres.conf's order."""
    idxs = list()
    with open(GRES_CONF) as f:
        for l in f:
            fields = l.split()
            t = re.search(r"Type=g(\d+)", l)
            if fields and fields[0] == f"NodeName={node}" and t and t.group(1) not in idxs:
                idxs.append(t.group(1))
    return idxs

def run(*cmd):
    """Runs [cmd], returning its stdout, or None (after printing why) if it fails."""
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(f"BlockGPUsManually: {' '.join(cmd)}: {p.stderr.strip()}", file=sys.stderr)
        return None
    return p.stdout

def block(node, machine, idx, singleton):
    """Makes sure GPU [idx] of [node] has a running manual blocker. Returns success."""
    name = f"{machine}_blocker_manual_{idx}"
    # If squeue fails, skip: an empty answer would mean "none" and submit a second one.
    out = run("squeue", "-h", "-n", name, "-o", "%i %T %Q")
    if out is None:
        return False
    if not out.strip():
        # Runs in / (exists on every machine); it only sleeps, so there's no output to keep.
        jobid = run("sbatch", "--parsable", "--requeue",
            *(["--dependency=singleton"] if singleton else []),
            "-p", "block-manual", "-w", node, f"--gres=gpu:g{idx}:1", "-c", "1", "-J", name,
            f"--comment={name}", "--chdir=/", "-o", "/dev/null", "-e", "/dev/null",
            "--wrap", "exec sleep infinity")
        if jobid is None:
            return False
        print(f"{name}: submitted ({jobid.strip()})")
        return True
    jobid, state, prio = out.splitlines()[0].split()
    if state == "PENDING" and prio == "0":   # held (--hold or requeuehold)
        # Start now, not after a requeue's ~2 min BeginTime.
        if run("scontrol", "release", jobid) is None \
            or run("scontrol", "update", f"jobid={jobid}", "StartTime=now") is None:
            return False
        print(f"{name}: released ({jobid})")
    else:
        print(f"{name}: already {state} ({jobid})")
    return True

def split_list(xs):
    """Flattens space- and comma-separated arguments."""
    return [x for arg in xs for x in arg.split(",") if x]

if __name__ == "__main__":
    P = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("--nodes", nargs="+", default=[],
        help="Machine names (A99) or hostnames, space or comma separated. Default: this machine only")
    P.add_argument("--gpus", nargs="+", default=[],
        help="nvidia-smi indices, space or comma separated. Default: every GPU each node gives SLURM")
    P.add_argument("--no-singleton", action="store_true",
        help="Don't submit with --dependency=singleton")
    args = P.parse_args()

    m2h = machine2hostname()
    nodes = split_list(args.nodes) or [socket.gethostname().split(".")[0]]
    idxs = split_list(args.gpus)
    ok = True
    for n in nodes:
        node = machine_host(n, m2h)
        all_idxs = gpus_of(node)
        if not all_idxs:
            print(f"BlockGPUsManually: {n} has no GPUs in SLURM", file=sys.stderr)
            ok = False
            continue
        for idx in idxs or all_idxs:
            if idx not in all_idxs:
                print(f"BlockGPUsManually: {n} has no GPU {idx} in SLURM", file=sys.stderr)
                ok = False
                continue
            ok = block(node, machine_name(node, m2h), idx, not args.no_singleton) and ok
    sys.exit(0 if ok else 1)
