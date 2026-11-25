#!/usr/bin/env python3
"""Coordinates multicluster sibling jobs across clusters (see README.md).

One long-running controller process does the work. Cron keeps it going:
    SLURMMultiClusterController.py --ensure_running --clusters apex solar narval
starts it if it isn't running, and otherwise wakes it (over a Unix socket). Inside it:

- one ClusterWorker thread per cluster. Woken by a wake message or by new verdicts for its cluster, it
  runs --remote_write_read on the cluster, in one SSH call, if it has verdicts to write or
  hasn't run for --min_seconds_between_reads. That writes the verdicts (and renews mc_time on
  the cluster's pending multicluster jobs), then reads the cluster's multicluster jobs. If
  they changed, it sends them to the ComputeWorker.
- one ComputeWorker thread. Woken by a wake message or by a cluster's new jobs, it works out each
  pending job's verdict ('run', 'requeue', 'requeuehold'), and sends each ClusterWorker the
  verdicts that differ from what its jobs' comments say.
- the main thread, which answers the socket: wake messages (which wake every thread, and restart any
  that died), --status and --stop.

Each thread has an inbox (queue.Queue) and a wake event (threading.Event, `wakeup`): a sender
puts a message in the inbox, then sets the wake event; the owner waits on it, clears it, then
empties its inbox. A failure in one thread (eg. a cluster that can't be reached) is logged and
doesn't affect the others.

The controller keeps nothing on disk. Each cluster keeps the multicluster jobs of its last read
in REMOTE_STATE_FILE, to tell which jobs left the queue since then: a job that left and whose
exp_folder has finished.txt finished its experiment.

Clusters are given by name: the one this machine belongs to (its SLURM ClusterName) is run
here, the others over SSH (SSHCommunication) at the ~/.ssh/config Host of the same name,
logging in with a Duo push if needed. Commands run in the user's interactive bash (`bash -ic`),
where SLURM and python are on PATH; output is read from OUTPUT_MARKER on.
"""
import argparse
import getpass
import json
import os
import os.path as osp
import queue
import re
import shlex
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))
import SSHCommunication
import UtilsBase
from UtilsBase import twrite
import MultiClusterComment as MCC

OUTPUT_MARKER = "____slurm_multicluster_response____" # Separates a command's output from what logging in prints
REMOTE_STATE_FILE = "~/.local/state/SLURMMultiCluster/queue_state.json"
STARTED_STATES = ["RUNNING", "CONFIGURING", "COMPLETING"]
MAX_COMMENT_LENGTH = 1024 # Seems to be the default limit on job comments in current SLURM versions
SOCKET_NAME = f"\0SLURMMultiClusterController.{getpass.getuser()}" # Abstract: no file, freed when the process dies

######################################################################################
# Run on each cluster: --remote_write_read
######################################################################################
def run_scontrol(*args):
    """Runs scontrol with [args], returning its error output, or None if it succeeded."""
    result = subprocess.run(["scontrol", *args], capture_output=True, text=True)
    return None if result.returncode == 0 else f"scontrol {' '.join(args)}: {(result.stdout + result.stderr).strip()}"

def read_queue():
    """Returns the IDs of every job of this user in the queue, and a dict mapping the IDs of the
    multicluster jobs among them (except ignored ones) to {'state', 'comment' (as a dict)}.
    """
    squeue = subprocess.run(["squeue", "--me", "-h", "-r", "-o", "%i|%T|%k"], capture_output=True, text=True, check=True)
    queued_jobids, jobid2job = set(), dict()
    for line in squeue.stdout.splitlines():
        if line.count("|") < 2:
            continue
        jobid, state, comment_str = line.split("|", 2) # The comment comes last since it may contain |
        queued_jobids.add(jobid)
        try:
            comment = UtilsBase.job_comment_to_dict(comment_str)
        except ValueError:
            continue
        if MCC.is_multicluster(comment) and not MCC.is_ignored(comment):
            jobid2job[jobid] = dict(state=state, comment=comment)
    return queued_jobids, jobid2job

def write_verdicts(*, jobid2job, verdicts, mc_time):
    """Writes [verdicts] (jobid -> {'mc_cont', 'mc_why'}) into the comments of the pending jobs of
    [jobid2job], merged into each comment as it is now, and renews mc_time to [mc_time] on all
    of them except those already held because their experiment finished. A job given
    'requeuehold' is held first, and gets the verdict only if that worked. Updates the comments
    in [jobid2job], and returns a list of errors.
    """
    errors = []
    for jobid,job in jobid2job.items():
        comment, verdict = job["comment"], verdicts.get(jobid, None)
        if job["state"] != "PENDING" or (verdict is None and comment["mc_cont"] == MCC.REQUEUEHOLD):
            continue
        new_comment = comment | dict(mc_time=mc_time) | (verdict or dict())
        new_comment = {k: v for k,v in new_comment.items() if not (k == "mc_why" and v is None)}
        new_comment_str = UtilsBase.dict_to_job_comment(new_comment)
        if len(new_comment_str) > MAX_COMMENT_LENGTH:
            errors.append(f"job {jobid}: new comment is over {MAX_COMMENT_LENGTH} characters: {new_comment_str}")
            continue
        if new_comment["mc_cont"] == MCC.REQUEUEHOLD and comment["mc_cont"] != MCC.REQUEUEHOLD:
            error = run_scontrol("hold", jobid)
            if error is not None:
                errors.append(error)
                continue
        error = run_scontrol("update", f"JobId={jobid}", f"Comment={new_comment_str}")
        if error is None:
            job["comment"] = new_comment
        else:
            errors.append(error)
    return errors

def find_finished(*, queued_jobids):
    """Returns a list of {'jobid', 'uid', 'exp_folder'} for the multicluster jobs of the last
    read (in REMOTE_STATE_FILE) that are no longer in the queue, and whose exp_folder has
    finished.txt.
    """
    state_file = osp.expanduser(REMOTE_STATE_FILE)
    jobid2previous = UtilsBase.json_to_dict(state_file) if osp.exists(state_file) else dict()
    return [dict(jobid=jobid, uid=previous["uid"], exp_folder=previous["exp_folder"])
        for jobid,previous in jobid2previous.items() if not jobid in queued_jobids
            and osp.exists(osp.join(osp.expanduser(previous["exp_folder"]), "finished.txt"))]

def save_queue_state(*, jobid2job):
    """Saves the multicluster jobs [jobid2job] to REMOTE_STATE_FILE, for the next find_finished()."""
    state_file = osp.expanduser(REMOTE_STATE_FILE)
    os.makedirs(osp.dirname(state_file), exist_ok=True)
    UtilsBase.dict_to_json({jobid: dict(uid=job["comment"]["uid"], exp_folder=job["comment"]["exp_folder"])
        for jobid,job in jobid2job.items()}, state_file)

def remote_write_read(*, verdicts, mc_time, dry_run):
    """Runs on a cluster: writes [verdicts] (see write_verdicts()), then returns a dict with
    'jobs', mapping the IDs of the cluster's multicluster jobs to {'uid', 'exp_folder', 'state',
    'mc_cont', 'mc_why', 'protected'}; 'finished', from find_finished(); and 'errors'. If
    [dry_run], nothing is written or saved.
    """
    queued_jobids, jobid2job = read_queue()
    errors = [] if dry_run else write_verdicts(jobid2job=jobid2job, verdicts=verdicts, mc_time=mc_time)
    finished = find_finished(queued_jobids=queued_jobids)
    if not dry_run:
        save_queue_state(jobid2job=jobid2job)
    jobs = {jobid: dict(uid=job["comment"]["uid"], exp_folder=job["comment"]["exp_folder"], state=job["state"],
        mc_cont=job["comment"]["mc_cont"], mc_why=job["comment"].get("mc_why", None),
        protected=MCC.is_protected(job["comment"])) for jobid,job in jobid2job.items()}
    return dict(jobs=jobs, finished=finished, errors=errors)

######################################################################################
# Talking to clusters
######################################################################################
class ClusterError(Exception):
    """A cluster couldn't be reached, or a command there failed."""

def run_on_cluster(*, host, command, timeout):
    """Returns the output of running shell [command] in an interactive bash on [host] ('local'
    for this machine, else a ~/.ssh/config Host), raising ClusterError if it fails.
    """
    script = f"echo {OUTPUT_MARKER}; {command}"
    try:
        if host == "local":
            result = subprocess.run(["bash", "-ic", script], capture_output=True, text=True,
                timeout=timeout, stdin=subprocess.DEVNULL)
        else:
            result = SSHCommunication.run_command_on_machine(machine=host, command=f"bash -ic {shlex.quote(script)}",
                mfa=True, if_connect_error="HostInfoError", return_process=True, timeout=timeout)
    except (subprocess.TimeoutExpired, SSHCommunication.CommandTimeoutError) as e:
        raise ClusterError(f"{host}: timed out after {timeout} s") from e
    except SSHCommunication.HostInfoError as e:
        raise ClusterError(f"{host}: {e}") from e
    lines = result.stdout.splitlines()
    if not OUTPUT_MARKER in lines: # A whole line: eg. a command echoed back contains the marker too
        raise ClusterError(f"{host}: couldn't run commands (exit {result.returncode}): {(result.stdout + result.stderr).strip()[-500:]}")
    output = "\n".join(lines[lines.index(OUTPUT_MARKER) + 1:])
    if result.returncode != 0:
        raise ClusterError(f"{host}: command exited {result.returncode}: {(output + result.stderr).strip()[-500:]}")
    return output

def write_read(*, host, verdicts, mc_time, args):
    """Runs remote_write_read() on [host], from the copy of this script at the same path relative
    to the home directory there, and returns what it returns.
    """
    script = "~/" + osp.relpath(osp.abspath(__file__), osp.expanduser("~"))
    request = json.dumps(dict(verdicts=verdicts, mc_time=mc_time, dry_run=args.dry_run))
    output = run_on_cluster(host=host, timeout=args.timeout_seconds,
        command=f"python {script} --remote_write_read {shlex.quote(request)}")
    try:
        return json.loads(output.strip().splitlines()[-1])
    except (ValueError, IndexError) as e:
        raise ClusterError(f"{host}: couldn't parse the reply: {output[-500:]}") from e

def get_local_cluster():
    """Returns the name of the SLURM cluster this machine belongs to, or None if it has none."""
    try:
        config = run_on_cluster(host="local", command="scontrol show config", timeout=60)
    except ClusterError:
        return None
    for line in config.splitlines():
        key,_,value = line.partition("=")
        if key.strip() == "ClusterName":
            return value.strip()
    return None

######################################################################################
# Deciding verdicts
######################################################################################
def jobid_ranking_key(job):
    """Ranks by job ID (its leading number, so array tasks and heterogeneous jobs work), then
    cluster name alphabetically. Job IDs of different clusters aren't comparable, but this is
    deterministic, and so far only orders siblings in the same state.
    """
    number = re.match(r"\d+", job["jobid"])
    return (int(number.group()) if number else float("inf"), job["cluster"])

RANKINGS = dict(jobid=jobid_ranking_key) # TODO: rgus_remaining (README.md)

def outranks(a, b, *, ranking):
    """Returns if sibling [a] outranks sibling [b]: a started sibling always outranks one that
    hasn't started; [ranking] (a key of RANKINGS) orders siblings in the same state.
    """
    key = lambda job: (not job["state"] in STARTED_STATES, *RANKINGS[ranking](job))
    return key(a) < key(b)

def compute_verdicts(*, cluster2jobs, cluster2read_time, finished_uid2why, now, args):
    """Returns cluster -> {jobid -> verdict} for the pending jobs whose verdict ({'mc_cont',
    'mc_why'}) differs from what their comment says, given each cluster's latest jobs
    [cluster2jobs], read at [cluster2read_time], and the uids found finished since the last
    computation [finished_uid2why]. Running jobs of a cluster not read for
    --running_info_expiry_seconds don't count.
    """
    jobs = [job | dict(cluster=cluster, jobid=jobid) for cluster,jobid2job in cluster2jobs.items() for jobid,job in jobid2job.items()]
    uid2jobs = dict()
    for job in jobs:
        uid2jobs.setdefault(job["uid"], []).append(job)
    finished_uid2why = finished_uid2why | {job["uid"]: job["mc_why"] for job in jobs if job["mc_cont"] == MCC.REQUEUEHOLD}

    cluster2verdicts = {cluster: dict() for cluster in cluster2jobs}
    for job in jobs:
        if job["state"] != "PENDING" or job["protected"] or job["mc_cont"] == MCC.REQUEUEHOLD:
            continue
        running_siblings = [sibling for sibling in uid2jobs[job["uid"]] if sibling["state"] in STARTED_STATES
            and now - cluster2read_time[sibling["cluster"]] < args.running_info_expiry_seconds]
        if job["uid"] in finished_uid2why:
            verdict = dict(mc_cont=MCC.REQUEUEHOLD, mc_why=finished_uid2why[job["uid"]])
        elif any(outranks(sibling, job, ranking=args.ranking) for sibling in running_siblings):
            verdict = dict(mc_cont=MCC.REQUEUE, mc_why=None)
        else:
            verdict = dict(mc_cont=MCC.RUN, mc_why=None)
        if (verdict["mc_cont"], verdict["mc_why"]) != (job["mc_cont"], job["mc_why"]):
            cluster2verdicts[job["cluster"]][job["jobid"]] = verdict
    return cluster2verdicts

######################################################################################
# The controller process
######################################################################################
class Mailbox:
    """A thread's inbox and wake event. send() puts a message (if any) then sets the wake event;
    the owner's receive() waits for it, clears it, then empties the inbox. Clearing before
    emptying means a message sent meanwhile sets the event again, so it's never missed.
    """
    def __init__(self):
        self.inbox, self.wakeup = queue.Queue(), threading.Event()

    def send(self, message=None):
        if not message is None:
            self.inbox.put(message)
        self.wakeup.set()

    def receive(self):
        self.wakeup.wait()
        self.wakeup.clear()
        messages = []
        while True:
            try:
                messages.append(self.inbox.get_nowait())
            except queue.Empty:
                return messages

class ClusterWorker:
    """The work of one cluster's thread. Messages in its mailbox are verdicts to write."""
    def __init__(self, *, cluster, host, compute_mailbox, args):
        self.name, self.cluster, self.host, self.args = f"write_read {cluster}", cluster, host, args
        self.mailbox, self.compute_mailbox = Mailbox(), compute_mailbox
        self.status = dict(last_error=None)
        self.pending_verdicts = dict() # jobid -> verdict, until written
        self.last_attempt_time = -float("inf")
        self.last_summary = None # What compute was last sent, for telling if anything changed
        self.has_told_compute = False # Compute waits for every cluster's first attempt

    def one_pass(self):
        for verdicts in self.mailbox.receive():
            self.pending_verdicts.update(verdicts)
        if len(self.pending_verdicts) == 0 and time.time() - self.last_attempt_time < self.args.min_seconds_between_reads:
            return
        self.last_attempt_time = time.time()
        try:
            reply = write_read(host=self.host, verdicts=dict(self.pending_verdicts), mc_time=int(time.time()), args=self.args)
        except ClusterError as e:
            twrite(f"[WARNING] {self.cluster}: {e}")
            self.status["last_error"] = f"{time.ctime()}: {e}"
            if not self.has_told_compute:
                self.compute_mailbox.send(dict(cluster=self.cluster, reply=None, read_time=None))
                self.has_told_compute = True
            return
        self.pending_verdicts.clear()
        self.status.update(last_read=time.ctime(), num_jobs=len(reply["jobs"]))
        for error in reply["errors"]:
            twrite(f"[WARNING] {self.cluster}: {error}")
        # New data: a job's state, protection or verdict changed (the reply comes after this call's
        # writes, so compute learns what it wrote), a job left, or a job finished
        summary = {jobid: (job["state"], job["protected"], job["mc_cont"], job["mc_why"]) for jobid,job in reply["jobs"].items()}
        if summary != self.last_summary or len(reply["finished"]) > 0:
            self.last_summary, self.has_told_compute = summary, True
            self.compute_mailbox.send(dict(cluster=self.cluster, reply=reply, read_time=time.time()))

class ComputeWorker:
    """The work of the compute thread. Messages in its mailbox are a cluster's read (its reply
    is None if the cluster couldn't be read); a wake message only wakes it.
    """
    def __init__(self, *, cluster2worker, args):
        self.name, self.cluster2worker, self.args = "compute", cluster2worker, args
        self.mailbox = Mailbox()
        self.status = dict(last_error=None)
        self.cluster2jobs, self.cluster2read_time = dict(), dict()
        self.attempted_clusters = set()
        self.finished_uid2why = dict() # Found since the last computation

    def one_pass(self):
        for message in self.mailbox.receive():
            self.attempted_clusters.add(message["cluster"])
            if message["reply"] is not None:
                for finished in message["reply"]["finished"]:
                    why = f"experiment finished: {finished['exp_folder']}/finished.txt, after {message['cluster']}:{finished['jobid']} left the queue"
                    twrite(f"[INFO] uid {finished['uid']}: {why}")
                    self.finished_uid2why[finished["uid"]] = why
                self.cluster2jobs[message["cluster"]] = message["reply"]["jobs"]
                self.cluster2read_time[message["cluster"]] = message["read_time"]
        if self.attempted_clusters != set(self.cluster2worker):
            return
        cluster2verdicts = compute_verdicts(cluster2jobs=self.cluster2jobs, cluster2read_time=self.cluster2read_time,
            finished_uid2why=self.finished_uid2why, now=time.time(), args=self.args)
        self.finished_uid2why.clear()
        self.status.update(last_computation=time.ctime())
        for cluster,verdicts in cluster2verdicts.items():
            for jobid,verdict in verdicts.items():
                job = self.cluster2jobs[cluster][jobid]
                twrite(f"[INFO] {cluster}:{jobid} (uid {job['uid']}): verdict {job['mc_cont']!r} -> {verdict['mc_cont']!r}")
            if len(verdicts) > 0:
                self.cluster2worker[cluster].mailbox.send(verdicts)

class Controller:
    def __init__(self, *, cluster2host, args):
        self.cluster2host = cluster2host
        self.cluster2worker = dict()
        self.compute = ComputeWorker(cluster2worker=self.cluster2worker, args=args)
        self.cluster2worker |= {cluster: ClusterWorker(cluster=cluster, host=host, compute_mailbox=self.compute.mailbox, args=args)
            for cluster,host in cluster2host.items()}
        self.workers = [self.compute, *self.cluster2worker.values()]
        self.worker2thread = dict()

    def run_worker(self, worker):
        """Runs [worker]'s passes forever, logging any exception rather than letting it end the thread."""
        while True:
            try:
                worker.one_pass()
            except Exception:
                worker.status["last_error"] = f"{time.ctime()}: {traceback.format_exc()}"
                twrite(f"[ERROR] {worker.name}: {traceback.format_exc()}")
                time.sleep(1) # Don't spin on an error repeating at once

    def start_thread(self, worker):
        self.worker2thread[worker] = threading.Thread(target=self.run_worker, args=(worker,), name=worker.name, daemon=True)
        self.worker2thread[worker].start()

    def wake_all(self):
        """Restarts any worker's thread that died, then wakes every worker."""
        for worker in self.workers:
            if not self.worker2thread[worker].is_alive():
                twrite(f"[WARNING] Thread {worker.name} died; restarting it")
                self.start_thread(worker)
            worker.mailbox.send()

    def handle_message(self, message):
        """Returns the reply to socket [message], and whether the controller should exit: for
        'stop', or a wake message with different clusters (cron then starts a new controller).
        """
        if message["type"] == "wake" and set(message["clusters"]) == set(self.cluster2host):
            self.wake_all()
            return dict(ok=True), False
        elif message["type"] in ["wake", "stop"]:
            twrite(f"[INFO] Exiting: {message}")
            return dict(ok=True, exiting=True), True
        elif message["type"] == "status":
            return dict(pid=os.getpid(), clusters=self.cluster2host, threads={worker.name: worker.status
                | dict(alive=self.worker2thread[worker].is_alive()) for worker in self.workers}), False
        return dict(error=f"unknown message {message}"), False

    def run(self):
        for worker in self.workers:
            self.start_thread(worker)
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(SOCKET_NAME)
        server.listen()
        twrite(f"[INFO] Controller running, pid {os.getpid()}, clusters {self.cluster2host}")
        self.wake_all()
        while True:
            connection, _ = server.accept()
            with connection:
                try:
                    _, uid, _ = struct.unpack("3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
                    if uid != os.getuid(): # The abstract namespace is shared by every user on the machine
                        continue
                    reply, should_exit = self.handle_message(json.loads(connection.makefile().readline()))
                    connection.sendall((json.dumps(reply) + "\n").encode())
                    if should_exit:
                        return
                except Exception:
                    twrite(f"[ERROR] socket: {traceback.format_exc()}")

######################################################################################
# Command line
######################################################################################
def send_message(message):
    """Sends [message] to the running controller and returns its reply, raising
    ConnectionRefusedError if no controller is running.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(30)
        s.connect(SOCKET_NAME)
        s.sendall((json.dumps(message) + "\n").encode())
        return json.loads(s.makefile().readline())

def get_args():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = P.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ensure_running", action="store_true",
        help="Start the controller if it isn't running, else wake it (run from cron)")
    mode.add_argument("--run_controller", action="store_true",
        help="Run the controller in this process (--ensure_running starts it this way)")
    mode.add_argument("--status", action="store_true", help="Show the running controller's status")
    mode.add_argument("--stop", action="store_true", help="Stop the running controller")
    mode.add_argument("--remote_write_read", metavar="REQUEST_JSON",
        help="Run on a cluster by the controller: write verdicts, then read the cluster's multicluster jobs")
    P.add_argument("--clusters", nargs="+", default=[],
        help="Names of the clusters to coordinate: this machine's own, and ~/.ssh/config Hosts")
    P.add_argument("--log", default=osp.expanduser("~/.local/state/SLURMMultiCluster/controller.log"),
        help="File a controller started by --ensure_running logs to")
    P.add_argument("--timeout_seconds", type=float, default=300,
        help="Seconds after which a command on a cluster counts as failed (not counting a Duo login)")
    P.add_argument("--min_seconds_between_reads", type=float, default=240,
        help="A cluster with no verdicts to write is read only if its last attempt was at least this long ago")
    P.add_argument("--running_info_expiry_seconds", type=float, default=1200,
        help="A cluster's running jobs stop counting if it hasn't been read for this long")
    P.add_argument("--ranking", choices=list(RANKINGS), default="jobid",
        help="How siblings in the same state are ordered")
    P.add_argument("--dry_run", action="store_true",
        help="Log verdicts, but write nothing on the clusters")
    args = P.parse_args()
    if (args.ensure_running or args.run_controller) and len(args.clusters) == 0:
        P.error("--ensure_running and --run_controller need --clusters")
    return args

if __name__ == "__main__":
    args = get_args()
    if args.remote_write_read is not None:
        request = json.loads(args.remote_write_read)
        print(json.dumps(remote_write_read(verdicts=request["verdicts"], mc_time=request["mc_time"], dry_run=request["dry_run"])))
    elif args.ensure_running:
        try:
            twrite(f"Woke the running controller: {send_message(dict(type='wake', clusters=args.clusters))}")
        except (ConnectionRefusedError, FileNotFoundError):
            # The controller gets this command's own arguments, so every setting reaches it
            controller_argv = [sys.executable, osp.abspath(__file__),
                *["--run_controller" if arg == "--ensure_running" else arg for arg in sys.argv[1:]]]
            os.makedirs(osp.dirname(args.log), exist_ok=True)
            with open(args.log, "a") as log:
                subprocess.Popen(controller_argv, start_new_session=True, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            twrite(f"Started a controller, logging to {args.log}")
    elif args.status or args.stop:
        try:
            print(json.dumps(send_message(dict(type="status" if args.status else "stop")), indent=2))
        except (ConnectionRefusedError, FileNotFoundError):
            print("No controller is running")
    elif args.run_controller:
        sys.stdout.reconfigure(line_buffering=True)
        with UtilsBase.named_lock(name="SLURMMultiClusterController", block=False) as acquired:
            if not acquired:
                twrite("[INFO] Another controller is running; exiting")
                sys.exit(0)
            local_cluster = get_local_cluster()
            Controller(cluster2host={cluster: "local" if cluster == local_cluster else cluster for cluster in args.clusters}, args=args).run()
