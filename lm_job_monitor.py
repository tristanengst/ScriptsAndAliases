"""lm_job_monitor: a compact view of jobs and machines, built for language-model agents.

Output is short, plain and stable, and exit codes carry outcomes, so an agent can check on
or wait for runs without long listings or scans of the shared filesystem. Covers SemiSLURM
jobs (through SemiSLURM's local queue cache) and runs started directly on a machine
(GPU processes of this user that SemiSLURM didn't start). Not SLURM, for now.

Commands:
status [--name PREFIX]     One line of SemiSLURM job counts by state and machine, then up to
                           --fails lines of crash reasons.
events [--name PREFIX]     Jobs whose state changed since the last events/wait call with the
                           same --name (the cursor lives in $TMP/lm_job_monitor), one per line.
wait [--name PREFIX]       Prints nothing until --on happens or --timeout passes, then the
                           changes; exit code 0 if no matching job is left unfinished, 1 if
                           any crashed, 3 on another change, 2 on timeout.
nodes [--machines M ...]   One line per machine: whether it accepts jobs, its excluded GPUs,
                           its last health check, free GPUs, and this user's GPU processes
                           (SemiSLURM's and others).

Examples:
lm_job_monitor status --name Oct08-Toy2D-1-7
lm_job_monitor wait --name Oct08-Toy2D-1-7 --on fail done --timeout 3h
lm_job_monitor nodes
"""
import argparse, collections, json, os, os.path as osp, re, subprocess, sys, time
sys.path[:0] = [osp.expanduser("~/.ScriptsAndAliases/SemiSLURM"), osp.expanduser("~/.ScriptsAndAliases")]  # Also when piped over SSH
import SemiSLURM as S

def cursor_path(name):
    """Returns the file holding the events cursor for jobs named [name]*."""
    d = osp.join(S.get_tmp(), "lm_job_monitor")
    _ = os.makedirs(d, exist_ok=True)
    return osp.join(d, f"cursor_{re.sub(r'[^A-Za-z0-9_.-]', '_', name or 'all')}.json")

def job_snapshot(name):
    """Returns a dict mapping job IDs (as strings) of jobs named [name]* to their state
    string: job state (or held), node, and crash reason if any.
    """
    out = dict()
    for jid, cfg, st in S.job_rows():
        if name and not cfg["job_name"].startswith(name):
            continue
        state = cfg["state"] if cfg["state"] == "held" and st["job_state"] == "queued" else st["job_state"]
        reason = f" {st.get('fail_reason', '-')}" if state == "crashed" else ""
        out[str(jid)] = f"{state}@{st['node_name']}{reason}"
    return out

def status(args):
    """Prints job counts by state and machine, then the commonest crash reasons."""
    snap = job_snapshot(args.name)
    counts = collections.Counter(v.split(" ")[0].replace("@-", "") for v in snap.values())
    print(" ".join(f"{k}:{v}" for k, v in sorted(counts.items())) or "no jobs")
    reasons = collections.defaultdict(list)
    _ = [reasons[v.split("@", 1)[1]].append(j) for j, v in snap.items() if v.startswith("crashed")]
    for r, jids in sorted(reasons.items(), key=lambda x: -len(x[1]))[:args.fails]:
        print(f"crashed@{r} x{len(jids)} (eg. job {jids[-1]})")

def changes(name):
    """Returns (new snapshot, list of 'JOB old -> new' lines) since the cursor, and saves it."""
    f = cursor_path(name)
    old = json.load(open(f)) if osp.exists(f) else dict()
    new = job_snapshot(name)
    lines = [f"{j}: {old.get(j, 'new')} -> {v}" for j, v in sorted(new.items(), key=lambda x: int(x[0])) if old.get(j) != v]
    _ = S.UtilsBase.atomic_save_lite(data=new, fpath=f)
    return new, lines

def print_lines(lines, *, limit):
    """Prints up to [limit] of [lines], then how many were left out."""
    _ = [print(l) for l in lines[:limit]]
    _ = print(f"... and {len(lines) - limit} more") if len(lines) > limit else None

def events(args):
    """Prints jobs whose state changed since the last call."""
    _, lines = changes(args.name)
    print_lines(lines, limit=args.limit) if lines else print("no changes")

def wait(args):
    """Waits for [args.on] among jobs named [args.name]*, then prints what changed."""
    end, seen = time.time() + parse_duration(args.timeout), []
    while True:
        snap, lines = changes(args.name)
        seen += lines
        crashed = any("-> crashed" in l for l in lines)
        done = bool(snap) and not any(v.split("@")[0] in ["queued", "running", "held"] for v in snap.values())
        code = 1 if "fail" in args.on and crashed else (0 if "done" in args.on and done else (3 if "change" in args.on and lines else None))
        if code is None and time.time() > end:
            code = 2
        if code is not None:
            print_lines(seen, limit=args.limit) if seen else None
            status(argparse.Namespace(name=args.name, fails=3))
            sys.exit(code)
        time.sleep(args.poll)

def parse_duration(s):
    """Returns seconds for a duration like 90, 90s, 20m or 3h."""
    m = re.fullmatch(r"([\d.]+)([smh]?)", s)
    return float(m.group(1)) * dict(s=1, m=60, h=3600)[m.group(2) or "s"]

def node_report():
    """Prints this machine's report as one JSON line (run on each machine by nodes())."""
    override = S.read_override()
    hdir = osp.join(S.get_tmp(), "semislurm", "health")
    health = [json.load(open(osp.join(hdir, f))) for f in (os.listdir(hdir) if osp.isdir(hdir) else []) if f.endswith(".json")]
    last = max(health, key=lambda h: h["time"]) if health else None
    gpus = S.query_gpus(override)
    used = {g for r in S.live_registry() for g in r["gpus"]}
    user = os.environ.get("USER", "")
    ours = subprocess.run(f"nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null", shell=True,
        capture_output=True, text=True).stdout.split()
    mine = [p for p in ours if osp.exists(f"/proc/{p}") and os.stat(f"/proc/{p}").st_uid == os.getuid()]
    semislurm = [p for p in mine if b"SEMISLURM_JOB_ID=" in open(f"/proc/{p}/environ", "rb").read()]
    print(json.dumps(dict(accept=override["accept"], exclude=override["exclude_gpus"],
        health=(last["reason"] or "ok") if last else "-", free=sum(1 for i, _, busy in gpus if not busy and not i in used and not i in override["exclude_gpus"]),
        total=len(gpus) - len(override["exclude_gpus"]), semislurm=len(semislurm), other=len(mine) - len(semislurm), user=user)))

def nodes(args):
    """Prints one line per machine in [args.machines]."""
    from concurrent.futures import ThreadPoolExecutor
    code = open(osp.abspath(__file__)).read()  # Sent over SSH, so machines needn't have this file
    def ask(m):
        argv = ["bash", "-c", "python3 - node_report"] if m == S.this_machine() else \
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", m, "python3 - node_report"]
        try:
            r = subprocess.run(argv, input=code, capture_output=True, text=True, timeout=60)
            return m, (r.returncode, r.stdout)
        except subprocess.TimeoutExpired:
            return m, (255, "")
    with ThreadPoolExecutor(16) as pool:
        replies = list(pool.map(ask, args.machines))
    for m, (rc, out) in replies:
        try:
            r = json.loads([l for l in out.splitlines() if l.startswith("{")][-1])
            print(f"{m}: {'accepts' if r['accept'] else 'OFF'}{' excl=' + str(r['exclude']) if r['exclude'] else ''} "
                f"health={r['health']} free={r['free']}/{r['total']} mine: semislurm={r['semislurm']} other={r['other']}")
        except (IndexError, ValueError, KeyError):
            print(f"{m}: unreachable" if rc == 255 else f"{m}: no report (rc {rc})")

def get_args():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C = P.add_subparsers(dest="cmd", required=True)
    for name in ["status", "events", "wait"]:
        p = C.add_parser(name)
        p.add_argument("--name", default=None, help="Only jobs whose name starts with this, eg. an experiment suffix")
        p.add_argument("--fails", type=int, default=5, help="Most crash-reason lines to print")
        p.add_argument("--limit", type=int, default=10, help="Most change lines to print")
        if name == "wait":
            p.add_argument("--on", nargs="+", default=["fail", "done"], choices=["fail", "done", "change"])
            p.add_argument("--timeout", default="3h", help="eg. 90s, 20m, 3h")
            p.add_argument("--poll", type=float, default=60, help="Seconds between checks")
    p = C.add_parser("nodes")
    p.add_argument("--machines", nargs="+", default=None, help="Defaults to SemiSLURM's workstations")
    _ = C.add_parser("node_report", help="[internal] This machine's report")
    args = P.parse_args()
    if args.cmd == "nodes" and args.machines is None:
        args.machines = S.default_machines()
    return args

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "node_report":
        node_report()
    else:
        dict(status=status, events=events, wait=wait, nodes=nodes)[args.cmd](args)
