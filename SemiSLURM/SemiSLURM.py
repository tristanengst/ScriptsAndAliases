"""SemiSLURM: SLURM-like job running on lab workstations. See Design.md.

Subcommands (run with -h for their arguments):
submit      -- submit a job (sbatch-like); by default also starts a dispatcher for it
dispatch    -- start a dispatcher adopting jobs without a live one
cancel      -- cancel jobs (scancel-like)
queue       -- show jobs (squeue-like; the ssqb alias)
node        -- show or edit a node's override file
tmp         -- print this machine's $TMP
accept      -- [internal] asked over SSH by a dispatcher whether this node takes a job
run         -- [internal] runs one job; started by the accepter

Must run under Python >= 3.10 with only the standard library, since the accepter and
runner use each node's system python3.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import errno
import fcntl
import functools
import hashlib
import getpass
import json
import os
import os.path as osp
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import traceback
import uuid
from zoneinfo import ZoneInfo

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import SSHCommunication
import UtilsBase
from UtilsBase import colorize, twrite

SCRIPT = osp.abspath(__file__)
JOBS_ROOT = os.environ.get("SEMISLURM_JOBS_ROOT", "~/scratch/SemiSLURM/jobs")
REPLY_PREFIX = "SEMISLURM_REPLY "
COMPLETED = ["finished", "crashed", "cancelled"]
BACKOFFS = [10, 20, 30]            # Seconds between asks to a rejecting machine; short, since GPUs free up as often as short jobs end
SWEEP_EVERY = 60                    # Seconds between dispatcher checks of runner liveness
LOG_SYNC_EVERY = 60                 # Seconds between runner appends to the NAS log_file
EARLY_LOG_SYNCS = [5, 10, 20, 30, 40, 50]  # Seconds after start of extra appends, so early output shows quickly
LAUNCH_CONFIRM_WAIT = 60            # Seconds a runner waits for its dispatcher's launch record
UNCERTAIN_LAUNCH_WAIT = 90          # Seconds a job isn't re-offered after a failed ask; > above
JOB_REJECT_WAIT = 600               # Seconds a job isn't re-offered to a machine that rejected it for job-specific reasons
JOB_SPECIFIC = ("unhealthy", "no_env", "excluded")  # Rejection reasons about the job, not the machine's capacity
CANCEL_GRACE = 30                   # Seconds between SIGTERM and SIGKILL of a cancelled job
HEALTH_TTL = dict(ok=1800, bad=600) # Seconds a node's health check result is reused, by outcome
FAIL_FILE = "failed.txt"            # A job may write this to its exp folder to say why it failed
STATE_KEYS = ["job_id", "job_name", "job_state", "reason", "submit_time", "start_time",
    "end_time", "requeues", "exit_code", "node_name", "gpus", "conda_env", "job_dir", "runner",
    "exp_folder", "log_file", "fail_reason"]
LAUNCH_KEYS = ["start_time", "node_name", "gpus", "conda_env", "job_dir", "runner"]

################################################################################
# Paths, time, and processes
################################################################################
def expand(p): return osp.abspath(osp.expanduser(p))
TZ = ZoneInfo("America/Vancouver")  # All recorded times are Pacific, whatever a machine's clock zone
def now_str(): return datetime.now(TZ).strftime("%Y-%m-%dT%H:%M:%S")
def str_to_datetime(s): return datetime.fromisoformat(s).replace(tzinfo=TZ)
def new_uid(): return uuid.uuid4().hex[:12]
def job_to_dir(job_id): return osp.join(expand(JOBS_ROOT), str(job_id))

def to_portable(p):
    """Returns path [p] written from ~ where possible, so it resolves on every machine."""
    p, home = expand(p), expand("~")
    scratch = osp.realpath(osp.join(home, "scratch"))
    if p == home or p.startswith(home + "/"):
        return "~" + p[len(home):]
    elif p == scratch or p.startswith(scratch + "/"):
        return "~/scratch" + p[len(scratch):]
    else:
        return p

def sh_path(p):
    """Returns path [p] quoted for bash, leaving a leading ~ to expand on the target."""
    return "~/" + shlex.quote(p[2:]) if p.startswith("~/") else shlex.quote(p)

def get_tmp():
    """Returns this machine's $TMP, creating it if needed."""
    user = os.environ.get("USER") or getpass.getuser()
    tmp = f"/localscratch/{user}/tmp" if osp.isdir(f"/localscratch/{user}") else f"/tmp/{user}"
    _ = os.makedirs(tmp, exist_ok=True)
    return tmp

def node_dir(*parts):
    """Returns a path in this node's SemiSLURM folder, creating its parent folder."""
    result = osp.join(get_tmp(), "semislurm", *parts)
    _ = os.makedirs(osp.dirname(result), exist_ok=True)
    return result

@functools.cache
def this_machine():
    """Returns the SSH name of this machine (eg. S2)."""
    return (os.environ.get("SEMISLURM_HOST") or SSHCommunication.get_machine_name()
        or socket.gethostname().split(".")[0])

def default_machines():
    """Returns the workstations dispatchers use by default."""
    import MachineInfo
    return [m for m in MachineInfo.machine2info if not m in MachineInfo.machines_cc + ["solar"]]

def run_on(node, cmd, *, timeout=120):
    """Returns a (returncode, stdout) tuple from running bash command [cmd] on [node],
    locally if [node] is this machine. A returncode of 255 means SSH failed.
    """
    argv = (["bash", "-c", cmd] if node == this_machine()
        else ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", node, cmd])
    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
            stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout
    except subprocess.TimeoutExpired:
        return 255, ""

def procs_status(procs):
    """Returns a dict mapping each 'host:pid:uid' record in [procs] to 'alive', 'dead', or
    'unknown' (host unreachable). A process is alive iff the command line of [pid] on
    [host] contains [uid]; this also handles PID reuse. Uses one SSH call per host.
    """
    host2procs, result = defaultdict(list), dict()
    for p in set(procs):
        _ = host2procs[p.split(":")[0]].append(p) if p.count(":") == 2 else result.update({p: "dead"})

    for host, hprocs in host2procs.items():
        pids = ",".join([p.split(":")[1] for p in hprocs])
        _, out = run_on(host, f"ps -o pid=,args= -p {pids}; echo SEMISLURM_PS_OK", timeout=60)
        if not "SEMISLURM_PS_OK" in out:
            result |= {p: "unknown" for p in hprocs}
            continue
        pid2args = {l.split(None, 1)[0]: l.split(None, 1)[-1] for l in out.splitlines() if l.strip()}
        result |= {p: "alive" if p.split(":")[2] in pid2args.get(p.split(":")[1], "") else "dead"
            for p in hprocs}
    return result

def proc_status(p): return procs_status([p])[p]

def kill_proc(p, *, sig="TERM"):
    """Sends [sig] to process record [p] if it is still alive."""
    host, pid, uid = p.split(":")
    return run_on(host, f"ps -o args= -p {pid} | grep -q {shlex.quote(uid)} && kill -{sig} {pid}")

def kill_job_procs_cmd(runner_uid, *, sig="KILL"):
    """Returns a bash command sending [sig] to every process of this user started by
    the job of the runner with [runner_uid], found by its SEMISLURM_RUNNER_UID.
    """
    return (f"for q in $(pgrep -U $(id -u)); do grep -qz '^SEMISLURM_RUNNER_UID={runner_uid}$' "
        f"/proc/$q/environ 2>/dev/null && kill -{sig} $q; done; true")

def kill_job_procs(p, *, sig="KILL"):
    """Sends [sig] to every process started by the job of runner record [p]."""
    return run_on(p.split(":")[0], kill_job_procs_cmd(p.split(":")[2], sig=sig))

################################################################################
# Job files
################################################################################
def retry_stale(fn):
    """Returns [fn] retried on NFS stale file handles (ESTALE), which reading a file that
    another machine just replaced with an atomic rename can raise.
    """
    @functools.wraps(fn)
    def wrapped(*args, **kwargs):
        for idx in range(20):
            try:
                return fn(*args, **kwargs)
            except OSError as e:
                if not e.errno == errno.ESTALE or idx == 19:
                    raise
                time.sleep(0.5)
    return wrapped

@retry_stale
def read_config(jdir): return UtilsBase.load_file_lite(osp.join(jdir, "config.json"))
def write_config(jdir, cfg): UtilsBase.atomic_save_lite(data=cfg, fpath=osp.join(jdir, "config.json"))

@retry_stale
def read_state(jdir):
    """Returns the state.txt of the job at [jdir] as a dict of strings."""
    with open(osp.join(jdir, "state.txt"), "r") as f:
        return dict([kv.split("=", 1) for kv in shlex.split(f.read())])

def write_state(jdir, state):
    """Atomically writes dict [state] to the state.txt of the job at [jdir]."""
    s = " ".join([f"{k}={shlex.quote(str(state[k]))}" for k in STATE_KEYS if k in state])
    _ = UtilsBase.atomic_save_lite(data=s + "\n", fpath=osp.join(jdir, "state.txt"))

def current_dispatcher(jdir):
    """Returns a (k, proc) tuple for the latest adoption of the job at [jdir], where
    [k] is -1 if it was never adopted and [proc] is None if the adopter hasn't yet
    recorded itself.
    """
    ks = [int(d.split("_")[1]) for d in os.listdir(jdir) if d.startswith("dispatcher_")]
    if not ks:
        return -1, None
    proc_file = osp.join(jdir, f"dispatcher_{max(ks)}", "proc.txt")
    return max(ks), (UtilsBase.load_file_lite(proc_file).strip() if osp.exists(proc_file) else None)

def dispatcher_status(jdir, *, k, proc, statuses):
    """Returns the liveness of the dispatcher of [jdir] given a (k, proc) tuple from
    current_dispatcher() and a [statuses] dict from procs_status().
    """
    if k < 0:
        return "dead"
    elif proc is None:  # An adopter that hasn't recorded itself yet
        young = time.time() - osp.getmtime(osp.join(jdir, f"dispatcher_{k}")) < 120
        return "alive" if young else "dead"
    else:
        return statuses[proc]

def adopt_jobs(job_ids, *, me):
    """Returns a dict mapping the jobs in [job_ids] that process [me] adopted to their
    adoption index. A job is adopted only if it has no recorded dispatcher or the
    recorded one is verifiably dead, and by mkdir so simultaneous adoptions can't both
    succeed.
    """
    jid2disp = {jid: current_dispatcher(job_to_dir(jid)) for jid in job_ids}
    statuses = procs_status([p for _, p in jid2disp.values() if not p is None])
    result = dict()
    for jid, (k, proc) in jid2disp.items():
        status = dispatcher_status(job_to_dir(jid), k=k, proc=proc, statuses=statuses)
        if not status == "dead":
            twrite(f"Not adopting job {jid}: its dispatcher {proc} is {status}")
            continue
        try:
            os.mkdir(osp.join(job_to_dir(jid), f"dispatcher_{k+1}"))
        except FileExistsError:
            twrite(f"Not adopting job {jid}: another dispatcher adopted it first")
            continue
        _ = UtilsBase.atomic_save_lite(data=me, fpath=osp.join(job_to_dir(jid), f"dispatcher_{k+1}", "proc.txt"))
        result[jid] = k + 1
    return result

def all_job_ids():
    """Returns the IDs of all submitted jobs, sorted."""
    root = expand(JOBS_ROOT)
    ids = [int(d) for d in os.listdir(root) if d.isdigit()] if osp.isdir(root) else []
    return sorted([jid for jid in ids if osp.exists(osp.join(root, str(jid), "state.txt"))])

def start_dispatcher(job_ids, *, machines, assign=1, background=True):
    """Starts a dispatcher adopting [job_ids], detached if [background] and otherwise
    waiting for it to finish. Returns the path to its log file.
    """
    uid = new_uid()
    cmd = [sys.executable, "-u", SCRIPT, "dispatch", "--adopt", *[str(j) for j in job_ids],
        "--machines", *machines, "--assign", str(assign), "--foreground", "1", "--proc_uid", uid]
    if background:
        log = node_dir("logs", f"dispatcher_{uid}.log")
        with open(log, "a") as f:
            _ = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL,
                stdout=f, stderr=subprocess.STDOUT)
        return log
    else:
        _ = subprocess.run(cmd + ["--loop_every", "5"])
        return None

################################################################################
# Node information
################################################################################
def gpu_name_to_type(name):
    """Returns the MachineInfo.gpu2info alias for nvidia-smi GPU name [name]."""
    import MachineInfo
    name = name.lower().replace(" ", "_").replace("-", "_")
    candidates = [(n, a) for n, a in MachineInfo.gpu_name2alias.items() if n.lower() in name]
    candidates += [(a, a) for a in MachineInfo.gpu2info if a.lower() in name]
    return max(candidates, key=lambda c: len(c[0]))[1] if candidates else name

def query_gpus(override):
    """Returns a list of (idx, gpu_type, busy) tuples for this node's GPUs, using
    nvidia-smi indexing. As in sqb, a GPU is busy if it has errors or a compute process
    with a visible owner; processes whose owner can't be found (eg. stale allocations
    with no live PID here) don't count.
    """
    smi = ["nvidia-smi", "--format=csv,noheader,nounits"]
    gpus = subprocess.run(smi + ["--query-gpu=index,uuid,name,ecc.errors.uncorrected.volatile.total,utilization.gpu"],
        capture_output=True, text=True).stdout
    apps = subprocess.run(smi + ["--query-compute-apps=gpu_uuid,pid"], capture_output=True, text=True).stdout
    uuid_pids = [[f.strip() for f in l.split(",")] for l in apps.splitlines() if l.count(",") == 1]
    pids = [p for _, p in uuid_pids if p.isdigit()]
    owned = subprocess.run(["ps", "-o", "pid=,user=", "-p", ",".join(pids)], capture_output=True, text=True).stdout if pids else ""
    owned_pids = [l.split()[0] for l in owned.splitlines() if len(l.split()) == 2]
    busy_uuids = [u for u, p in uuid_pids if p in owned_pids]
    result = []
    for line in gpus.splitlines():
        fields = [f.strip() for f in line.split(",")]
        if not len(fields) == 5 or not fields[0].isdigit():
            continue
        idx, gpu_uuid, name, ecc, util = fields
        error = not ecc in ["0", "[N/A]", "N/A"] or not util.isdigit()
        gpu_type = override.get("gpu_type") or gpu_name_to_type(name)
        result.append((int(idx), gpu_type, error or gpu_uuid in busy_uuids))
    return result

def conda_envs():
    """Returns a dict mapping the names of conda envs on this machine to their prefixes."""
    user = os.environ.get("USER") or getpass.getuser()
    envs_txt = expand("~/.conda/environments.txt")
    prefixes = [l.strip() for l in open(envs_txt)] if osp.exists(envs_txt) else []
    roots = [expand(r) for r in ["~/miniconda3", "~/anaconda3", "~/miniforge3", f"/localscratch/{user}/miniconda3"]]
    roots += [osp.dirname(osp.dirname(os.environ["CONDA_EXE"]))] if "CONDA_EXE" in os.environ else []
    for r in roots:
        prefixes += [r] + ([osp.join(r, "envs", e) for e in sorted(os.listdir(osp.join(r, "envs")))]
            if osp.isdir(osp.join(r, "envs")) else [])
    is_root = lambda p: osp.isdir(osp.join(p, "condabin"))
    return {("base" if is_root(p) else osp.basename(p)): p for p in reversed(prefixes)
        if p and osp.isdir(osp.join(p, "bin"))}

def read_override():
    """Returns this node's override dict (keys: accept, exclude_gpus, max_gpus, gpu_type)."""
    f = node_dir("override.json")
    return dict(accept=1, exclude_gpus=[], max_gpus=-1, gpu_type="") | (UtilsBase.load_file_lite(f) if osp.exists(f) else dict())

def live_registry():
    """Returns the registry entries of runners on this node that are alive. Entries of
    dead runners are removed along with their $SLURM_JOBDIR.
    """
    result, reg = [], node_dir("registry", "x")
    for f in sorted(os.listdir(osp.dirname(reg))):
        f = osp.join(osp.dirname(reg), f)
        try:
            entry = UtilsBase.load_file_lite(f)
        except (OSError, ValueError):  # A partial write can't exist, but be careful
            continue
        try:
            with open(f"/proc/{entry['pid']}/cmdline", "rb") as cf:
                alive = entry["uid"].encode() in cf.read()
        except OSError:
            alive = False
        if alive:
            result.append(entry)
        else:
            _ = shutil.rmtree(entry["job_dir"], ignore_errors=True) if entry["job_dir"].startswith(get_tmp() + "/") else None
            _ = os.remove(f)
    return result

def runner_job_dir(cfg, *, pid, start_time):
    """Returns the $SLURM_JOBDIR of a runner with [pid] started at [start_time]."""
    exp_name = osp.basename(cfg["exp_folder"].rstrip("/"))
    return osp.join(get_tmp(), f"{exp_name}_{pid}_{start_time.replace(':', '').replace('-', '')}")

################################################################################
# Submit
################################################################################
def tar_code(code_dir, *, out):
    """Atomically writes a tar of the code in [code_dir] to [out]. In a git repository,
    this is the tracked and non-ignored untracked files; otherwise, all files.
    """
    git = subprocess.run(["git", "-C", code_dir, "ls-files", "-co", "--exclude-standard", "-z"],
        capture_output=True, text=True)
    if git.returncode == 0:
        files = [f for f in git.stdout.split("\0") if f and osp.isfile(osp.join(code_dir, f))]
    else:
        files = [osp.relpath(osp.join(d, f), code_dir) for d, _, fs in os.walk(code_dir) for f in fs]
    tmp = f"{out}.tmp_{new_uid()}"
    with tarfile.open(tmp, "w") as t:
        for f in files:
            t.add(osp.join(code_dir, f), arcname=f)
    os.replace(tmp, out)

def submit(args):
    """Submits a job per [args], prints its ID, and returns it."""
    import MachineInfo
    gpus_per_node = {kv.split(":")[0]: sorted([int(c) for c in kv.split(":")[1].split(",")], reverse=True)
        for kv in args.gpus_per_node}
    unknown = [t for t in gpus_per_node if not t in MachineInfo.gpu2info]
    if unknown:
        raise ValueError(f"Unknown GPU types {unknown}; known: {list(MachineInfo.gpu2info)}")

    root = expand(JOBS_ROOT)
    _ = os.makedirs(root, exist_ok=True)
    job_id = max([int(d) for d in os.listdir(root) if d.isdigit()], default=0) + 1
    while True:
        try:
            os.mkdir(osp.join(root, str(job_id)))
            break
        except FileExistsError:
            job_id += 1
    jdir = job_to_dir(job_id)

    exp_folder = to_portable(args.exp_folder or jdir)
    _ = os.makedirs(expand(exp_folder), exist_ok=True)
    _ = shutil.copy(args.script, osp.join(jdir, "job.sh"))
    _ = tar_code(expand(args.code_dir), out=osp.join(expand(exp_folder), "code.tar")) if args.code_dir else None
    cfg = dict(job_name=args.job_name or osp.basename(args.script),
        exp_folder=exp_folder,
        log_file=to_portable(args.log_file or osp.join(expand(exp_folder), f"semislurm_{job_id}.log")),
        priority=args.priority,
        min_disk_gb=args.min_disk_gb,
        nodelist=args.nodelist, exclude=args.exclude,
        gpus_per_node=gpus_per_node,
        conda_env=args.conda_env,
        max_requeues=args.max_requeues,
        success_file=args.success_file,
        require_commands=args.require_commands, require_files=args.require_files, require_imports=args.require_imports,
        state="queued")
    _ = write_config(jdir, cfg)
    _ = write_state(jdir, dict(job_id=job_id, job_name=cfg["job_name"], job_state="queued",
        reason="none", submit_time=now_str(), start_time="-", end_time="-", requeues=0,
        exit_code="-", node_name="-", gpus="-", conda_env="-", job_dir="-", runner="-",
        exp_folder=cfg["exp_folder"], log_file=cfg["log_file"], fail_reason="-"))
    print(f"Submitted batch job {job_id}")

    if args.dispatch:
        log = start_dispatcher([job_id], machines=args.machines)
        twrite(f"Started dispatcher for job {job_id}; log at {log}")
    return job_id

################################################################################
# Dispatcher
################################################################################
def ask(machine, jid, *, me):
    """Returns the reply of [machine]'s accepter for job [jid], or None if it couldn't
    be had (which means the job may or may not have been launched).
    """
    cmd = f"python3 {sh_path(to_portable(SCRIPT))} accept {sh_path(to_portable(job_to_dir(jid)))} --node {machine} --dispatcher {me}"
    _, out = run_on(machine, cmd)
    replies = [l[len(REPLY_PREFIX):] for l in out.splitlines() if l.startswith(REPLY_PREFIX)]
    try:
        return json.loads(replies[-1]) if replies else None
    except json.JSONDecodeError:
        return None

def static_ok(cfg, *, machine, gpu_types):
    """Returns if a job with config [cfg] could run on [machine] given its [gpu_types]."""
    return ((not cfg["nodelist"] or machine in cfg["nodelist"]) and not machine in cfg["exclude"]
        and (gpu_types is None or any([t in cfg["gpus_per_node"] for t in gpu_types])))

def set_reason(jdir, *, reason):
    """Sets [reason] on the queued job at [jdir] if it differs."""
    state = read_state(jdir)
    _ = write_state(jdir, state | dict(reason=reason)) if state["job_state"] == "queued" and not state["reason"] == reason else None

def requeued_state(state):
    """Returns job [state] reset to queued, as a requeue leaves it."""
    return state | {k: "-" for k in LAUNCH_KEYS} | dict(job_state="queued", reason="requeued", end_time="-",
        exit_code="-", fail_reason="-", requeues=int(state["requeues"]) + 1)

def handle_cancel(jid, *, state, jid2kill_time):
    """Acts on a cancelled, or requeued, running job [jid] with [state]. Running jobs'
    runners are sent SIGTERM, and their sessions SIGKILL if that hasn't worked after a
    while. A runner that died without recording its end is recorded as cancelled (or
    queued again) here.
    """
    jdir = job_to_dir(jid)
    if state["job_state"] == "queued" and read_config(jdir)["state"] == "requeue":
        _ = write_config(jdir, read_config(jdir) | dict(state="queued"))
        return
    elif state["job_state"] == "queued":
        _ = write_state(jdir, state | dict(job_state="cancelled", reason="cancelled", end_time=now_str()))
        twrite(f"Job {jid}: cancelled while queued")
        return

    status = proc_status(state["runner"])
    if status == "alive" and not jid in jid2kill_time:
        _ = kill_proc(state["runner"], sig="TERM")
        jid2kill_time[jid] = time.time()
        twrite(f"Job {jid}: sent SIGTERM to runner {state['runner']}")
    elif status == "alive" and time.time() - jid2kill_time[jid] > CANCEL_GRACE + 30:
        _ = kill_proc(state["runner"], sig="KILL")
        _ = kill_job_procs(state["runner"], sig="KILL")
        twrite(f"Job {jid}: sent SIGKILL to runner {state['runner']} and its job's processes")
    elif status == "dead":
        _ = kill_job_procs(state["runner"], sig="KILL")
        state = read_state(jdir)  # The runner may have recorded its end before dying
        if state["job_state"] == "running" and read_config(jdir)["state"] == "requeue":
            _ = write_state(jdir, requeued_state(state))
            _ = write_config(jdir, read_config(jdir) | dict(state="queued"))
        elif state["job_state"] == "running":
            _ = write_state(jdir, state | dict(job_state="cancelled", reason="cancelled", end_time=now_str()))
        twrite(f"Job {jid}: runner {state['runner']} verifiably dead; job is {read_state(jdir)['job_state']}")

def sweep(jid2state, *, jid2cfg):
    """Marks running jobs in [jid2state] whose runners are verifiably dead as crashed,
    requeuing them if they have requeues left.
    """
    running = {jid: s for jid, s in jid2state.items() if s["job_state"] == "running"}
    statuses = procs_status([s["runner"] for s in running.values()])
    for jid, s in running.items():
        if not statuses[s["runner"]] == "dead" or jid2cfg[jid]["state"] in ["cancelled", "requeue"]:
            continue
        _ = kill_job_procs(s["runner"], sig="KILL")  # Orphans of a kill -9'd runner
        s = read_state(job_to_dir(jid))  # The runner may have recorded its end before dying
        if not s["job_state"] == "running":
            continue
        elif int(s["requeues"]) < jid2cfg[jid]["max_requeues"]:
            s |= {k: "-" for k in LAUNCH_KEYS} | dict(job_state="queued", reason="requeued", requeues=int(s["requeues"]) + 1)
            twrite(f"Job {jid}: runner died; requeued ({s['requeues']}/{jid2cfg[jid]['max_requeues']})")
        else:
            s |= dict(job_state="crashed", reason="runner_died", end_time=now_str())
            twrite(f"Job {jid}: runner died; out of requeues, so crashed")
        _ = write_state(job_to_dir(jid), s)

def dispatch(args):
    """Runs a dispatcher over the jobs in [args.adopt] until all are completed."""
    me = f"{this_machine()}:{os.getpid()}:{args.proc_uid}"
    job_ids = all_job_ids() if args.adopt == ["all"] else [int(j) for j in args.adopt]
    jid2k = adopt_jobs(job_ids, me=me)
    twrite(f"Dispatcher {me} adopted jobs {sorted(jid2k)}", machines=args.machines, assign=args.assign)

    machine2backoff = {m: (0, 0.) for m in args.machines}   # (rejections, next ask time)
    machine2gpu_types, jid2retry_after, jid2kill_time, last_sweep, jm2retry_after = dict(), dict(), dict(), 0., dict()
    while jid2k:
        try:
            ##################################################################
            # Read job files; drop jobs adopted by someone else or completed
            ##################################################################
            jid2k = {jid: k for jid, k in jid2k.items()
                if not osp.exists(osp.join(job_to_dir(jid), f"dispatcher_{k+1}"))}
            jid2cfg, jid2state = dict(), dict()
            for jid in jid2k:
                try:
                    jid2cfg[jid], jid2state[jid] = read_config(job_to_dir(jid)), read_state(job_to_dir(jid))
                except (OSError, ValueError) as e:
                    twrite(f"Job {jid}: couldn't read files ({e}); skipping this round")

            ##################################################################
            # Act on config edits, and sweep for dead runners
            ##################################################################
            for jid, cfg in jid2cfg.items():
                if cfg["state"] in ["cancelled", "requeue"] and not jid2state[jid]["job_state"] in COMPLETED:
                    _ = handle_cancel(jid, state=jid2state[jid], jid2kill_time=jid2kill_time)
                elif cfg["state"] == "requeue":  # Killed by a runner that predates requeue, so recorded as ended
                    _ = write_state(job_to_dir(jid), requeued_state(jid2state[jid]))
                    _ = write_config(job_to_dir(jid), cfg | dict(state="queued"))
                    twrite(f"Job {jid}: requeued")
                elif cfg["state"] == "held" and jid2state[jid]["job_state"] == "queued":
                    _ = set_reason(job_to_dir(jid), reason="held")
            if time.time() - last_sweep > SWEEP_EVERY:
                _ = sweep(jid2state, jid2cfg=jid2cfg)
                last_sweep = time.time()

            jid2state = {jid: read_state(job_to_dir(jid)) for jid in jid2state}
            done = [jid for jid, s in jid2state.items() if s["job_state"] in COMPLETED]
            for m in [jid2state[jid].get("node_name") for jid in done]:  # A job ending frees its GPUs,
                if m in machine2backoff:                                   # so ask its machine again now
                    machine2backoff[m] = (0, 0.)
            _ = twrite(f"Jobs {done} completed; no longer tracking them") if done else None
            jid2k = {jid: k for jid, k in jid2k.items() if not jid in done}

            ##################################################################
            # Assign queued jobs to machines
            ##################################################################
            queued = [jid for jid, s in jid2state.items() if s["job_state"] == "queued"
                and jid2cfg[jid]["state"] == "queued" and jid2retry_after.get(jid, 0) <= time.time()]
            queued = sorted(queued, key=lambda jid: (-jid2cfg[jid]["priority"], jid))
            for m in (args.machines if args.assign else []):
                while machine2backoff[m][1] <= time.time():
                    cands = [jid for jid in queued if static_ok(jid2cfg[jid], machine=m, gpu_types=machine2gpu_types.get(m))
                        and jm2retry_after.get((jid, m), 0) <= time.time()]
                    if not cands:
                        break
                    jid = cands[0]
                    reply = ask(m, jid, me=me)
                    machine2gpu_types[m] = reply.get("gpu_types") if reply else machine2gpu_types.get(m)
                    if reply and reply["ok"]:
                        state = read_state(job_to_dir(jid))
                        if state["job_state"] == "queued":  # Else the runner recorded itself
                            launch = dict(node_name=m, gpus=",".join([str(g) for g in reply["gpus"]]),
                                **{k: reply[k] for k in ["start_time", "conda_env", "job_dir", "runner"]})
                            _ = write_state(job_to_dir(jid), state | launch | dict(job_state="running", reason="none"))
                        twrite(f"Job {jid}: launched on {m}", gpus=reply["gpus"], runner=reply["runner"])
                        queued.remove(jid)
                        machine2backoff[m] = (0, 0.)
                    else:
                        reason = reply["reason"] if reply else "unreachable"
                        if not reply or reason.startswith("error"):  # It may have launched
                            jid2retry_after[jid] = time.time() + UNCERTAIN_LAUNCH_WAIT
                        _ = set_reason(job_to_dir(jid), reason=f"{m}:{reason}")
                        twrite(f"Job {jid}: rejected by {m} ({reason})")
                        if reason.startswith(JOB_SPECIFIC):  # Only this job can't run here: try the machine's next candidate
                            jm2retry_after[(jid, m)] = time.time() + JOB_REJECT_WAIT
                            continue
                        n = machine2backoff[m][0]
                        machine2backoff[m] = (n + 1, time.time() + BACKOFFS[min(n, len(BACKOFFS) - 1)])
        except Exception:
            twrite(f"Dispatcher error:\n{traceback.format_exc()}")
        _ = time.sleep(args.loop_every) if jid2k else None
    twrite(f"Dispatcher {me} exiting: all its jobs are completed or adopted elsewhere")

################################################################################
# Accepter
################################################################################
def node_health(conda_env, cfg):
    """Returns '' if this node has what the job with config [cfg] declares it needs in
    [conda_env] (importable modules, with CUDA if torch is one; commands resolvable in an
    interactive shell with the env activated, such as aliases; existing files), and
    otherwise a short reason. Results are cached in node_dir('health') for HEALTH_TTL;
    deleting that folder makes the next ask check again.
    """
    imports, cmds, files = cfg.get("require_imports") or [], cfg.get("require_commands") or [], cfg.get("require_files") or []
    if not (imports or cmds or files):
        return ""
    f = node_dir("health", hashlib.md5(json.dumps([conda_env, imports, cmds, files]).encode()).hexdigest()[:12] + ".json")
    if osp.exists(f):
        c = UtilsBase.load_file_lite(f)
        if time.time() - c["time"] < HEALTH_TTL["bad" if c["reason"] else "ok"]:
            return c["reason"]
    reason = next((f"no_file:{osp.basename(x)}" for x in files if not osp.exists(osp.expanduser(x))), "")
    if not reason and imports:
        code = f"import {', '.join(imports)}" + ("; assert torch.cuda.is_available()" if "torch" in imports else "")
        r = subprocess.run([osp.join(conda_envs()[conda_env], "bin", "python"), "-c", code], capture_output=True, timeout=300)
        reason = "" if r.returncode == 0 else "imports"
    for c in (cmds if not reason else []):
        r = subprocess.run(["bash", "-ic", f"conda activate {shlex.quote(conda_env)} >/dev/null 2>&1; type {shlex.quote(c)}"],
            capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
        if r.returncode:
            reason = f"no_command:{c}"
            break
    _ = UtilsBase.atomic_save_lite(data=dict(time=time.time(), reason=reason), fpath=f)
    return reason

def accept_locked(args):
    """Returns the accepter's reply dict for [args]; called under the node lock."""
    jdir = expand(args.job_dir)
    cfg, state, override = read_config(jdir), read_state(jdir), read_override()
    gpus = query_gpus(override)
    gpu_types = sorted(set([t for _, t, _ in gpus]))
    def no(reason): return dict(ok=False, reason=reason, gpu_types=gpu_types)

    if not cfg["state"] == "queued" or not state["job_state"] == "queued":
        return no("not_queued")
    elif (cfg["nodelist"] and not args.node in cfg["nodelist"]) or args.node in cfg["exclude"]:
        return no("excluded")
    elif not override["accept"]:
        return no("override")
    envs = conda_envs()
    conda_env = next((e for e in cfg["conda_env"] if e in envs), None)
    if conda_env is None:
        return no("no_env")
    elif shutil.disk_usage(get_tmp()).free / 1e9 < cfg["min_disk_gb"]:
        return no("disk")
    elif not os.access(expand(cfg["exp_folder"]), os.W_OK):
        return no("unhealthy:exp_folder")  # eg. the NAS isn't mounted where the job expects
    elif (bad := node_health(conda_env, cfg)):
        return no(f"unhealthy:{bad}")

    registry = live_registry()
    used = [g for r in registry for g in r["gpus"]]
    free = [(idx, t) for idx, t, busy in gpus if not busy and not idx in used and not idx in override["exclude_gpus"]]
    cap = override["max_gpus"] - len(used) if override["max_gpus"] >= 0 else len(gpus)
    for t, counts in cfg["gpus_per_node"].items():
        t_free = [idx for idx, tt in free if tt == t]
        count = next((c for c in sorted(counts, reverse=True) if 0 < c <= min(len(t_free), cap)), None)
        if count is not None:
            grant = t_free[:count]
            break
    else:
        return no("no_gpus")

    uid, start_time = new_uid(), now_str()
    with open(node_dir("logs", f"runner_{uid}.log"), "a") as f:
        p = subprocess.Popen([sys.executable, "-u", SCRIPT, "run", args.job_dir, "--node", args.node,
            "--gpus", *[str(g) for g in grant], "--conda_env", conda_env, "--proc_uid", uid,
            "--start_time", start_time],
            start_new_session=True, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT)
    job_dir = runner_job_dir(cfg, pid=p.pid, start_time=start_time)
    _ = UtilsBase.atomic_save_lite(data=dict(job_id=state["job_id"], pid=p.pid, uid=uid, gpus=grant,
        job_dir=job_dir, dispatcher=args.dispatcher), fpath=node_dir("registry", f"{uid}.json"))
    return dict(ok=True, gpus=grant, runner=f"{args.node}:{p.pid}:{uid}", start_time=start_time,
        conda_env=conda_env, job_dir=job_dir, gpu_types=gpu_types)

def accept(args):
    """Prints the accepter's reply for [args], deciding under a node-local lock."""
    try:
        with open(node_dir("accept.lock"), "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            reply = accept_locked(args)
    except Exception as e:
        reply = dict(ok=False, reason=f"error:{type(e).__name__}", detail=str(e)[:500])
    print(REPLY_PREFIX + json.dumps(reply), flush=True)

################################################################################
# Runner
################################################################################
def run(args):
    """Runs the job at [args.job_dir] and records its state; see Design.md."""
    jdir = expand(args.job_dir)
    cfg = read_config(jdir)
    me = f"{args.node}:{os.getpid()}:{args.proc_uid}"
    slurm_jobdir = runner_job_dir(cfg, pid=os.getpid(), start_time=args.start_time)
    local_log, log_file = osp.join(slurm_jobdir, "semislurm_job.log"), expand(cfg["log_file"])
    launch = dict(start_time=args.start_time, node_name=args.node, gpus=",".join([str(g) for g in args.gpus]),
        conda_env=args.conda_env, job_dir=slurm_jobdir, runner=me)
    flags = dict(cancelled=False, kill_time=None, job=None, recorded=False, log_offset=0)

    def kill_job(sig):
        try:
            _ = os.killpg(flags["job"].pid, sig) if flags["job"] else None
        except ProcessLookupError:
            pass

    def on_term(signum, frame):
        flags["cancelled"], flags["kill_time"] = True, time.time()
        _ = kill_job(signal.SIGTERM)

    def sync_log():
        if not osp.exists(local_log):
            return
        with open(local_log, "rb") as f:
            _ = f.seek(flags["log_offset"])
            data = f.read()
        if data:
            _ = os.makedirs(osp.dirname(log_file), exist_ok=True)
            with open(log_file, "ab") as f:
                _ = f.write(data)
            flags["log_offset"] += len(data)

    def check_recorded(force):
        """Sets flags['recorded'] if the job's state.txt has this runner, waiting for
        the dispatcher's launch record or, after [LAUNCH_CONFIRM_WAIT] or with
        [force], writing it. If the job stopped being queued meanwhile, it is cancelled.
        """
        state = read_state(jdir)
        if state["runner"] == me:
            flags["recorded"] = True
        elif force or time.time() - t0 > LAUNCH_CONFIRM_WAIT:
            if state["job_state"] == "queued" and read_config(jdir)["state"] == "queued":
                _ = write_state(jdir, state | launch | dict(job_state="running", reason="none"))
                flags["recorded"] = True
            elif not flags["cancelled"]:
                twrite(f"Runner {me}: job {state['job_id']} is {state['job_state']} without this runner; stopping")
                _ = on_term(None, None)

    t0, rc, error = time.time(), None, None
    exp = expand(cfg["exp_folder"])
    outcome_files = [osp.join(exp, x) for x in [cfg.get("success_file"), FAIL_FILE] if x]
    _ = signal.signal(signal.SIGTERM, on_term)
    try:
        _ = [os.remove(x) for x in outcome_files if osp.exists(x)]  # Left by an earlier attempt of this job
        _ = os.makedirs(slurm_jobdir)
        code_tar = osp.join(expand(cfg["exp_folder"]), "code.tar")
        if osp.exists(code_tar):
            with tarfile.open(code_tar) as t:
                t.extractall(slurm_jobdir)
        _ = shutil.copy(osp.join(jdir, "job.sh"), osp.join(slurm_jobdir, "semislurm_job.sh"))

        prefix = conda_envs()[args.conda_env]
        gpus_str = ",".join([str(g) for g in args.gpus])
        env = os.environ | dict(PATH=f"{prefix}/bin:{os.environ.get('PATH', '')}", CONDA_PREFIX=prefix,
            CONDA_DEFAULT_ENV=args.conda_env, SLURM_JOBDIR=slurm_jobdir, CUDA_DEVICE_ORDER="PCI_BUS_ID",
            CUDA_VISIBLE_DEVICES=gpus_str, SEMISLURM_GPUS=" ".join([str(g) for g in args.gpus]),
            SEMISLURM_JOB_ID=str(read_state(jdir)["job_id"]), SEMISLURM_NODE=args.node, SEMISLURM_RUNNER_UID=args.proc_uid, TMP=get_tmp())
        with open(local_log, "ab") as f:
            _ = f.write(f"[SemiSLURM {now_str()}] job {env['SEMISLURM_JOB_ID']} on {args.node}, GPUs {gpus_str}, conda env {args.conda_env}, SLURM_JOBDIR={slurm_jobdir}\n".encode())
            f.flush()
            flags["job"] = subprocess.Popen(["bash", osp.join(slurm_jobdir, "semislurm_job.sh")],
                cwd=slurm_jobdir, env=env, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                preexec_fn=os.setpgrp)
        _ = kill_job(signal.SIGTERM) if flags["cancelled"] else None

        last_sync, last_check, early_syncs = time.time(), 0., [t0 + s for s in EARLY_LOG_SYNCS]
        while rc is None:
            time.sleep(1)
            rc = flags["job"].poll()
            if not flags["recorded"] and time.time() - last_check > 5:
                _ = check_recorded(force=False)
                last_check = time.time()
            if flags["cancelled"] and time.time() - flags["kill_time"] > CANCEL_GRACE:
                _ = kill_job(signal.SIGKILL)
            if (early_syncs and time.time() >= early_syncs[0]) or time.time() - last_sync > LOG_SYNC_EVERY:
                early_syncs = [t for t in early_syncs if t > time.time()]
                _ = sync_log()
                last_sync = time.time()
    except Exception as e:
        twrite(f"Runner {me} error:\n{traceback.format_exc()}")
        error = f"runner_error:{type(e).__name__}"
        _ = kill_job(signal.SIGKILL)
    finally:
        end_time = now_str()
        try:
            while not flags["recorded"] and time.time() - t0 < LAUNCH_CONFIRM_WAIT:
                _ = check_recorded(force=False)
                _ = time.sleep(1) if not flags["recorded"] else None
            _ = check_recorded(force=True) if not flags["recorded"] else None
            success = rc == 0 and (not cfg.get("success_file") or osp.exists(osp.join(exp, cfg["success_file"])))
            job_state = "cancelled" if flags["cancelled"] else ("finished" if success else "crashed")
            fail_file = osp.join(exp, FAIL_FILE)
            fail_reason = "-" if job_state != "crashed" else (error or (" ".join(open(fail_file).read().split())[:200] if osp.exists(fail_file)
                else (f"exit_code:{rc}" if rc else f"no_{cfg['success_file']}")))
            if osp.isdir(slurm_jobdir):
                with open(local_log, "ab") as f:
                    _ = f.write(f"[SemiSLURM {now_str()}] job ended: {job_state}, exit code {rc}\n".encode())
                _ = sync_log()
            state = read_state(jdir)
            if state["runner"] == me and flags["cancelled"] and read_config(jdir)["state"] == "requeue":
                _ = write_state(jdir, requeued_state(state))
                _ = write_config(jdir, read_config(jdir) | dict(state="queued"))
                job_state = "requeued"
            elif state["runner"] == me:
                _ = write_state(jdir, state | dict(job_state=job_state, end_time=end_time, fail_reason=fail_reason,
                    exit_code="-" if rc is None else rc, reason="cancelled" if flags["cancelled"] else "none"))
            twrite(f"Runner {me}: job ended", job_state=job_state, exit_code=rc, recorded=state["runner"] == me)
        finally:
            _ = subprocess.run(["bash", "-c", kill_job_procs_cmd(args.proc_uid)])  # Leftover background processes
            _ = shutil.rmtree(slurm_jobdir, ignore_errors=True)
            reg = node_dir("registry", f"{args.proc_uid}.json")
            _ = os.remove(reg) if osp.exists(reg) else None

################################################################################
# Job control, node control, and status display
################################################################################
def cancel(args):
    """Cancels the jobs in [args.jobs]. Their dispatchers act on it; jobs without a live
    dispatcher get a short-lived controller that does and then exits.
    """
    job_ids = all_job_ids() if args.jobs == ["all"] else [int(j) for j in args.jobs]
    to_control = []
    for jid in job_ids:
        jdir = job_to_dir(jid)
        if not osp.exists(osp.join(jdir, "state.txt")):
            twrite(f"Job {jid} not found")
            continue
        elif read_state(jdir)["job_state"] in COMPLETED:
            continue
        _ = write_config(jdir, read_config(jdir) | dict(state="cancelled"))
        to_control.append(jid)

    jid2disp = {jid: current_dispatcher(job_to_dir(jid)) for jid in to_control}
    statuses = procs_status([p for _, p in jid2disp.values() if not p is None])
    jid2status = {jid: dispatcher_status(job_to_dir(jid), k=k, proc=p, statuses=statuses) for jid, (k, p) in jid2disp.items()}
    twrite(f"Cancelling {to_control}; jobs with live dispatchers: {[j for j, s in jid2status.items() if s == 'alive']}")
    _ = [twrite(f"Job {jid}: dispatcher liveness unknown; it is cancelled once one acts") for jid, s in jid2status.items() if s == "unknown"]
    dead = [jid for jid, s in jid2status.items() if s == "dead"]
    _ = sys.stdout.flush()
    _ = start_dispatcher(dead, machines=[], assign=0, background=False) if dead else None

def ensure_dispatchers(job_ids, *, machines):
    """Starts one dispatcher for those of [job_ids] without a live one, so they get run."""
    jid2disp = {jid: current_dispatcher(job_to_dir(jid)) for jid in job_ids}
    statuses = procs_status([p for _, p in jid2disp.values() if not p is None])
    dead = [jid for jid, (k, p) in jid2disp.items() if dispatcher_status(job_to_dir(jid), k=k, proc=p, statuses=statuses) == "dead"]
    if dead:
        log = start_dispatcher(dead, machines=machines or default_machines())
        twrite(f"Started a dispatcher for jobs {dead}; log at {log}")

def hold(args):
    """Holds the queued jobs in [args.jobs], so they aren't started until released."""
    for jid in [int(j) for j in args.jobs]:
        jdir = job_to_dir(jid)
        cfg, state = read_config(jdir), read_state(jdir)
        if state["job_state"] == "queued" and cfg["state"] == "queued":
            _ = write_config(jdir, cfg | dict(state="held"))
        else:
            twrite(f"Job {jid} is {state['job_state']} (config {cfg['state']}); only queued jobs can be held")

def release(args):
    """Releases the held jobs in [args.jobs], making sure they have a dispatcher."""
    released = []
    for jid in [int(j) for j in args.jobs]:
        jdir = job_to_dir(jid)
        cfg = read_config(jdir)
        if cfg["state"] == "held":
            _ = write_config(jdir, cfg | dict(state="queued"))
            _ = set_reason(jdir, reason="none")
            released.append(jid)
        else:
            twrite(f"Job {jid} isn't held (config {cfg['state']})")
    _ = ensure_dispatchers(released, machines=args.machines) if released else None

def requeue(args):
    """Puts the jobs in [args.jobs] back in the queue with the same ID, folder and UID, so
    they resume from their latest checkpoint: running ones are killed first (by their
    dispatcher), and finished, crashed or cancelled ones are reset. With [args.code_dir],
    their code snapshot is refreshed first.
    """
    requeued = []
    for jid in [int(j) for j in args.jobs]:
        jdir = job_to_dir(jid)
        cfg, state = read_config(jdir), read_state(jdir)
        _ = tar_code(expand(args.code_dir), out=osp.join(expand(cfg["exp_folder"]), "code.tar")) if args.code_dir else None
        if state["job_state"] == "running":
            _ = write_config(jdir, cfg | dict(state="requeue"))
        elif state["job_state"] in COMPLETED:
            _ = write_state(jdir, requeued_state(state))
            _ = write_config(jdir, cfg | dict(state="queued"))
        else:
            _ = write_config(jdir, cfg | dict(state="queued"))  # Queued or held: just make sure it can run
        requeued.append(jid)
    twrite(f"Requeued {requeued}")
    _ = ensure_dispatchers(requeued, machines=args.machines) if requeued else None

def node(args):
    """Shows and edits the override file of [args.node]."""
    if not args.node == this_machine():
        _, out = run_on(args.node, f"python3 {sh_path(to_portable(SCRIPT))} node " + shlex.join(sys.argv[2:]))
        print(out, end="")
        return
    override = read_override()
    edits = dict(accept=args.accept, exclude_gpus=args.exclude_gpus, max_gpus=args.max_gpus, gpu_type=args.gpu_type)
    override = (dict() if args.clear else override) | {k: v for k, v in edits.items() if not v is None}
    _ = UtilsBase.atomic_save_lite(data=override, fpath=node_dir("override.json")) if args.clear or any([not v is None for v in edits.values()]) else None
    twrite(f"{this_machine()} override:", **read_override())
    for r in live_registry():
        twrite(f"{this_machine()} runner:", job_id=r["job_id"], pid=r["pid"], gpus=r["gpus"])

def elapsed_str(state):
    """Returns the runtime of a job with [state] as D-HH:MM:SS."""
    if state["start_time"] == "-":
        return "-"
    end = datetime.now(TZ) if state["end_time"] == "-" else str_to_datetime(state["end_time"])
    s = max(0, int((end - str_to_datetime(state["start_time"])).total_seconds()))
    return (f"{s // 86400}-" if s >= 86400 else "") + f"{s % 86400 // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"

def job_rows():
    """Returns (job ID, config, state) tuples for all jobs. Completed jobs' rows are cached
    in node_dir('queue_cache.json') with their state file's mtime, so later calls read
    only active jobs' files from JOBS_ROOT; deleting the cache falls back to reading all.
    """
    cache_f = node_dir("queue_cache.json")
    try:
        cache = json.load(open(cache_f)) if osp.exists(cache_f) else dict()
    except ValueError:
        cache = dict()
    root, rows, new_cache = expand(JOBS_ROOT), [], dict()
    for jid in [int(d) for d in os.listdir(root) if d.isdigit()] if osp.isdir(root) else []:
        jdir = job_to_dir(jid)
        try:
            mtime = os.stat(osp.join(jdir, "state.txt")).st_mtime
            c = cache.get(str(jid))
            if c and c["mtime"] == mtime:
                cfg, state = c["cfg"], c["state"]
            else:
                cfg, state = read_config(jdir), read_state(jdir)
        except (OSError, ValueError):
            continue
        rows.append((jid, cfg, state))
        if state["job_state"] in COMPLETED:
            new_cache[str(jid)] = dict(mtime=mtime, cfg=cfg, state=state)
    try:
        _ = UtilsBase.atomic_save_lite(data=new_cache, fpath=cache_f)
    except OSError:
        pass
    return sorted(rows, key=lambda r: r[0])

def queue(args):
    """Prints the jobs in JOBS_ROOT with derived states, or with [args.summary] a short
    count by state and node plus the failures' reasons.
    """
    rows = [r for r in job_rows() if not args.name or r[1]["job_name"].startswith(args.name)]
    if args.active:
        rows = [r for r in rows if not r[2]["job_state"] in COMPLETED]
    if args.summary:
        counts = defaultdict(int)
        for _, cfg, st in rows:
            counts[(st["job_state"] if cfg["state"] in ["queued", "requeue"] or st["job_state"] in COMPLETED else cfg["state"], st["node_name"])] += 1
        print(" ".join(f"{k[0]}{'@' + k[1] if k[1] != '-' else ''}:{v}" for k, v in sorted(counts.items())))
        fails = [(jid, st) for jid, _, st in rows if st["job_state"] == "crashed"]
        reasons = defaultdict(list)
        _ = [reasons[(st["node_name"], st.get("fail_reason", "-"))].append(jid) for jid, st in fails]
        for (n, r), jids in sorted(reasons.items(), key=lambda x: -len(x[1]))[:args.summary]:
            print(f"crashed@{n}: {r} x{len(jids)} (eg. {jids[-1]})")
        return
    cutoff = time.time() - args.hours * 3600
    rows = [r for r in rows if args.all or not r[2]["job_state"] in COMPLETED
        or r[2]["end_time"] == "-" or str_to_datetime(r[2]["end_time"]).timestamp() > cutoff]

    jid2disp = {jid: current_dispatcher(job_to_dir(jid)) for jid, _, s in rows if s["job_state"] == "queued"}
    statuses = procs_status([p for _, p in jid2disp.values() if not p is None]) if args.check else dict()
    def display_state(jid, state):
        if not state["job_state"] == "queued":
            return state["job_state"]
        k, proc = jid2disp[jid]
        status = dispatcher_status(job_to_dir(jid), k=k, proc=proc, statuses=statuses) if args.check else "alive"
        return dict(alive="pending", dead="dead", unknown="unknown")[status]

    state2color = dict(running="green", pending="blue", dead="red", unknown="orange", finished="no_change",
        crashed="red", cancelled="orange")
    header = ["JOBID", "NAME", "STATE", "REASON", "NODE", "GPUS", "TIME", "PRIO", "REQ", "SUBMIT"]
    table = [[str(jid), cfg["job_name"], display_state(jid, s), s["reason"], s["node_name"], s["gpus"],
        elapsed_str(s), str(cfg["priority"]), s["requeues"], s["submit_time"].replace("T", " ")]
        for jid, cfg, s in rows]
    widths = [max([len(r[idx]) for r in [header] + table]) for idx in range(len(header))]
    print("  ".join([h.ljust(w) for h, w in zip(header, widths)]))
    for r in table:
        cells = [c.ljust(w) for c, w in zip(r, widths)]
        cells[2] = colorize(cells[2], color=state2color.get(r[2], "no_change")) if args.color else cells[2]
        print("  ".join(cells))

################################################################################
# Command line
################################################################################
def get_args():
    """Returns the argparse Namespace for the subcommand in sys.argv."""
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    S = P.add_subparsers(dest="cmd", required=True)

    PS = S.add_parser("submit", help="Submit a job")
    PS.add_argument("script", help="Bash script that is the job body")
    PS.add_argument("--gpus_per_node", nargs="+", required=True,
        help="GPU_TYPE:COUNT1,COUNT2,... per acceptable GPU type, eg. l40s:2,4 3090:2. The largest free allowed count is granted")
    PS.add_argument("--conda_env", nargs="+", required=True,
        help="Acceptable conda envs; the first one present on the node is used")
    PS.add_argument("--job_name", default=None,
        help="Job name; defaults to the script's file name")
    PS.add_argument("--exp_folder", default=None,
        help="Experiment folder (on the NAS); defaults to the job's folder in JOBS_ROOT")
    PS.add_argument("--log_file", default=None,
        help="Log file the job's output is appended to; defaults to EXP_FOLDER/semislurm_JOBID.log")
    PS.add_argument("--code_dir", default=".",
        help="Folder whose code is put in EXP_FOLDER/code.tar and extracted into $SLURM_JOBDIR. '' to keep an existing code.tar")
    PS.add_argument("--priority", type=int, default=0)
    PS.add_argument("--min_disk_gb", type=float, default=10,
        help="Free space needed at $TMP")
    PS.add_argument("--nodelist", nargs="*", default=[])
    PS.add_argument("--exclude", nargs="*", default=[])
    PS.add_argument("--max_requeues", type=int, default=20)
    PS.add_argument("--success_file", default=None,
        help="File (relative to the exp folder) the job writes when it finishes successfully; without it, a zero exit counts as crashed")
    PS.add_argument("--require_imports", nargs="*", default=[],
        help="Modules the conda env must import (torch also needs CUDA); nodes failing this don't take the job")
    PS.add_argument("--require_commands", nargs="*", default=[],
        help="Commands (eg. aliases) an interactive shell with the env activated must resolve")
    PS.add_argument("--require_files", nargs="*", default=[],
        help="Files (eg. ~/.netrc) that must exist on the node")
    PS.add_argument("--dispatch", type=int, default=1, choices=[0, 1],
        help="Start a dispatcher for the job")
    PS.add_argument("--machines", nargs="+", default=None,
        help="Machines the dispatcher may use; defaults to all workstations")

    PD = S.add_parser("dispatch", help="Start a dispatcher for jobs without a live one")
    PD.add_argument("--adopt", nargs="+", required=True,
        help="Job IDs, or 'all' for all jobs")
    PD.add_argument("--machines", nargs="*", default=None,
        help="Machines to use; defaults to all workstations")
    PD.add_argument("--assign", type=int, default=1, choices=[0, 1],
        help="Launch jobs; with 0, only act on config edits and sweep")
    PD.add_argument("--foreground", type=int, default=0, choices=[0, 1])
    PD.add_argument("--loop_every", type=float, default=15,
        help="Seconds between rounds")
    PD.add_argument("--proc_uid", default=None,
        help="Set automatically")

    PC = S.add_parser("cancel", help="Cancel jobs")
    PC.add_argument("jobs", nargs="+",
        help="Job IDs, or 'all' for all jobs")

    PH = S.add_parser("hold", help="Hold queued jobs, so they aren't started until released")
    PH.add_argument("jobs", nargs="+")
    PRL = S.add_parser("release", help="Release held jobs")
    PRL.add_argument("jobs", nargs="+")
    PRL.add_argument("--machines", nargs="*", default=None, help="Machines for a new dispatcher, if one is needed; defaults to all workstations")
    PRQ = S.add_parser("requeue", help="Put jobs back in the queue with the same ID, folder and UID, killing running ones first")
    PRQ.add_argument("jobs", nargs="+")
    PRQ.add_argument("--machines", nargs="*", default=None, help="Machines for a new dispatcher, if one is needed; defaults to all workstations")
    PRQ.add_argument("--code_dir", default=None, help="Refresh the jobs' code snapshot from this folder first")

    PQ = S.add_parser("queue", help="Show jobs")
    PQ.add_argument("--all", type=int, default=0, choices=[0, 1],
        help="Show all completed jobs, not just recent ones")
    PQ.add_argument("--hours", type=float, default=24,
        help="Show jobs completed within this many hours")
    PQ.add_argument("--check", type=int, default=1, choices=[0, 1],
        help="Check dispatcher liveness over SSH (to tell pending from dead)")
    PQ.add_argument("--color", type=int, default=1, choices=[0, 1])
    PQ.add_argument("--active", type=int, default=0, choices=[0, 1], help="Show only jobs not yet completed")
    PQ.add_argument("--name", default=None, help="Show only jobs whose name starts with this")
    PQ.add_argument("--summary", type=int, default=0,
        help="Instead of the table, print one line of counts by state and node, then up to this many lines of crash reasons")

    PN = S.add_parser("node", help="Show or edit a node's override file")
    PN.add_argument("--node", default=None,
        help="Defaults to this machine")
    PN.add_argument("--accept", type=int, default=None, choices=[0, 1],
        help="Whether the node accepts jobs")
    PN.add_argument("--exclude_gpus", type=int, nargs="*", default=None,
        help="GPU indices (nvidia-smi order) never granted")
    PN.add_argument("--max_gpus", type=int, default=None,
        help="Cap on GPUs granted on the node at once; -1 for no cap")
    PN.add_argument("--gpu_type", default=None,
        help="Force the node's GPU type; '' to detect it")
    PN.add_argument("--clear", type=int, default=0, choices=[0, 1],
        help="Reset the override before applying edits")

    _ = S.add_parser("tmp", help="Print this machine's $TMP")

    PA = S.add_parser("accept", help="[internal] Answer whether this node takes a job")
    PA.add_argument("job_dir")
    PA.add_argument("--node", required=True)
    PA.add_argument("--dispatcher", required=True)

    PR = S.add_parser("run", help="[internal] Run a job")
    PR.add_argument("job_dir")
    PR.add_argument("--node", required=True)
    PR.add_argument("--gpus", type=int, nargs="+", required=True)
    PR.add_argument("--conda_env", required=True)
    PR.add_argument("--proc_uid", required=True)
    PR.add_argument("--start_time", required=True)

    args = P.parse_args()
    if args.cmd in ["submit", "dispatch"]:
        args.machines = default_machines() if args.machines is None else args.machines
    if args.cmd == "dispatch":
        args.proc_uid = new_uid() if args.proc_uid is None else args.proc_uid
    if args.cmd == "node":
        args.node = this_machine() if args.node is None else args.node
    return args

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "dispatch" and not args.foreground:
        log = start_dispatcher(args.adopt, machines=args.machines, assign=args.assign)
        twrite(f"Started dispatcher; log at {log}")
    else:
        _ = dict(submit=submit, dispatch=dispatch, cancel=cancel, hold=hold, release=release, requeue=requeue, queue=queue,
            node=node, accept=accept, run=run, tmp=lambda args: print(get_tmp()))[args.cmd](args)
