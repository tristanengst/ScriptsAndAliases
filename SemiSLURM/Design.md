# SemiSLURM design

SLURM-like job running on the lab workstations/servers, where installing SLURM isn't possible.
Each job is one run on one node (no multinode, preemption, fair-share, or CPU/memory requests).
Agreed in a BetterCondIMLE session on 2026-10-04; implemented in `SemiSLURM.py` on 2026-10-05
(see Implementation notes at the end).

Conventions: lower_snake_case names. Every write to a shared file is a write to a temporary
file followed by an atomic `rename`; anything else is unsafe. Each file has a single writer
where possible. Communication between machines is plain `ssh` (traffic is tiny and rare).

## Roles
1. **Dispatcher.** Loops over its jobs and the machines it may use, asking each machine's
   accepter to take the highest-priority compatible job, until the machine rejects; then moves
   to the next machine. Also acts as the long-running job controller for its jobs (below).
   Any number may run per user, but a job has at most one dispatcher at a time.
2. **Job control.** Short-lived commands (`scancel`/`scontrol`-like) that edit a job's config
   file (cancel, hold/release, priority, requeue). Its dispatcher acts on the edit.
3. **Accepter.** `ssh NODE JobAccepter.py --ask JOB_DIR`, run per request. Under a node-local
   file lock, so decisions on a node are serialized, it checks whether the node can run the
   job, and either starts a runner and answers yes (with the granted GPUs) or answers no with
   a reason. No long-running process.
4. **Runner.** Started by the accepter; executes one job and records its state. Controlled by
   the job's dispatcher, not the accepter.
5. **Node controller.** One-off command that edits a node-local override file (eg. accept
   nothing, exclude some GPUs, cap GPUs). The accepter reads it before each answer.
   Concurrency here doesn't matter.
6. **Status display.** `squeue`-like listing of jobs from their files, with derived states.

## A job
Stored on the NAS in `JOBS_ROOT/<job_id>/` (`JOBS_ROOT` is a shared path, TBD). `job_id` is a
unique integer: the submitter creates `JOBS_ROOT/<n>` with `mkdir` for the next integer,
retrying on `FileExistsError`.

- `job.sh`: the job body, a bash script (SLURM-lite: no `#SBATCH` lines).
- `config.json`: the `#SBATCH`-equivalent settings plus control fields. Written at submit, then
  only by job-control commands (atomically). A rare race between two such edits is accepted.
  ```json
  {
    "job_name": "Oct04-FashionMNISTColor-0-1-3",
    "exp_folder": "...",
    "log_file": "...",
    "priority": 0,
    "min_disk_gb": 20,
    "nodelist": [], "exclude": [],
    "gpus_per_node": {"l40s": [2, 4], "3090": [2]},
    "conda_env": ["py314BcIMLE"],
    "max_requeues": 20,
    "state": "queued"
  }
  ```
  - `gpus_per_node` maps each acceptable GPU type to its allowed GPU counts. The accepter
    grants the largest allowed count that is free. Counts come from a heuristic (eg. VRAM from
    `MachineInfo.gpu2info`) that only lists counts dividing the job's batch size.
  - `conda_env`: the node needs at least one of these envs; the runner uses the first present.
  - `state`: control field, one of `queued`, `held`, `cancelled`.
- `state.txt` (or `.json`): `scontrol show job`-like runtime state. Written by the dispatcher
  until launch, then only by the runner; always atomically.
  ```
  job_id=17 job_name=Oct04-FashionMNISTColor-0-1-3 job_state=running reason=none
  submit_time=... start_time=... end_time=... requeues=0 exit_code=-
  node_name=S1 gpus=0,6,8,9 conda_env=py314BcIMLE job_dir=$TMP/<exp_name>_<pid>_<start>
  dispatcher=S2:<pid>:<uid> runner=S1:<pid>:<uid> exp_folder=... log_file=...
  ```
  Runtime isn't stored; the status display computes it from `start_time`/`end_time`.
- `dispatcher_<k>/`: adoption markers (see Dispatcher).
- At submit, the job's code is copied to `<exp_folder>/code.tar`.

**Job states.** Stored `job_state`: `queued`, `running`, `finished`, `crashed`, `cancelled`.
Displayed states derive from it and dispatcher liveness: `dead` (queued, no live dispatcher),
`pending` (queued, live dispatcher), `running`, `completed` (finished/crashed/cancelled).

## Liveness
Every long-lived process (dispatcher, runner) gets `--proc_uid <random>` on its command line and
records `host:pid:uid`. It is alive iff `ssh host ps -o args= -p pid` shows a command containing
`uid`; this also handles PID reuse.

## Dispatcher
Loop:
1. **Assign.** For each machine not in back-off, offer its highest-priority compatible queued
   job to the accepter; on yes, record the launch in `state.txt` and offer the next; on no,
   record the rejection reason as the jobs' `reason` and back off that machine: next ask after
   at least 30 s, then 1 min, then 3 min, then 3 min onward. Back-off resets on acceptance.
2. **Act on config edits.** Re-read its jobs' `config.json` periodically. `cancelled` running
   jobs: kill the runner over `ssh` (its process group), verify it died; the runner records
   `cancelled`. Priority/hold changes apply immediately.
3. **Sweep.** A job of its own with `job_state=running` whose runner is verifiably dead is
   marked `crashed`; it is requeued (`requeues += 1`) while `requeues < max_requeues`.
It exits once all its jobs are completed.

**One dispatcher per job.** A dispatcher adopts a job only if none is recorded or the recorded
one is verifiably dead, by creating `JOBS_ROOT/<job_id>/dispatcher_<k>` with `mkdir` (`k` =
number of adoptions so far), so simultaneous adoptions can't both succeed.
**Dead jobs.** `dispatch --adopt JOB_IDS` starts a new dispatcher for jobs without a live one.
A runner still going keeps going and records its own end state; control of the job moves to the
new dispatcher. Control actions on a dead job's running runner work the same way: start a
controller, verify the old one is dead, act, verify, exit.

## Accepter checks (in order, under the node-local lock)
1. Node is in `nodelist` (if given) and not in `exclude`.
2. Node override file allows it.
3. A `conda_env` exists on the node.
4. At least `min_disk_gb` free at `$TMP`.
5. The node's GPU type is in `gpus_per_node` and an allowed count of GPUs is free. Free means
   free per `sqb` logic (no process allocated on the GPU) minus GPUs granted to runners on this
   node that are still alive (a node-local registry), so a just-launched job can't be
   double-booked before it allocates.
Then pick specific GPU indices (`nvidia-smi` indexing; `tpython_ddp` translates to CUDA order),
start the runner detached (`setsid`), answer yes with the GPUs. Rejection reasons, eg.
`excluded`, `override`, `no_env`, `disk`, `no_gpus`, become the jobs' `reason` for display.

## Runner
1. Create `$SLURM_JOBDIR = $TMP/<exp_name>_<pid>_<starttime>` (assumed unique) and extract
   `<exp_folder>/code.tar` there. `$SLURM_JOBDIR` is the only SLURM-style variable set; granted
   GPUs etc. reach the job separately.
2. Run `job.sh` in the matched conda env (PATH pinned to it), in its own process group.
3. Log to a local file in `$SLURM_JOBDIR`; append new output to `log_file` (in `exp_folder`, on
   the NAS) every ~60 s and at the end. The NAS is bad at frequent I/O.
4. Record `start_time`, `end_time`, `exit_code`, `job_state` (`running` -> `finished` /
   `crashed` / `cancelled`), and its `host:pid:uid`.
5. On exit (trap), delete `$SLURM_JOBDIR` and remove its GPUs from the node registry. Folders
   of verifiably dead runners (eg. `kill -9`) are swept by the accepter.

## Machine setup
- `$TMP` is set in `.bashrc` by evaluating a ScriptsAndAliases command that gives the right
  value per machine (prefer `/localscratch/$USER/...` where it exists).
- Paths are written from `~`; `~/scratch` must resolve to the same NAS folder on every machine.
  (On 2026-10-04 it didn't: `/NAS` on S2, `/NAS/tme3/scratch` on S1, `/NAS/tme3` on A9.)

## Using it from a project (BetterCondIMLE)
- Generate the run's UID before submitting; `TrainBasic.py --uid UID --exp_name_folder ...`
  makes `get_args()` print the `exp_folder` it would use and exit, so `exp_folder`, `log_file`
  and `code.tar` are known at submit. A requeued job reuses the UID and so resumes from its
  last checkpoint (WandB must allow resuming for such UIDs).
- One job per run, not arrays: a `SlurmSubmit.py`-like generator writes and submits one job per
  configuration, and a `SubmitGrid.py` variant uses it for grids. See
  `~/Development/IMLE-SSL-2/SlurmSubmit.py` for an (overly complicated) example.
- Integrate with `sqb`/`scb`/`scancelb`-style commands so workstations and SLURM clusters look
  alike; read `JobInfo.py`, `Scb.py`, `SubmitJobChain.py`, `Scancelb.py`, `Sqb2.py` first.

## Testing before real use
Many tiny jobs over two or more nodes; two dispatchers adopting the same dead jobs; killing a
dispatcher, a runner (`kill -9`), and an accepter mid-launch; cancelling queued, running, and
dead jobs; back-to-back launches on the same GPUs (double-booking guard); a node override
toggled mid-run; logs reaching `exp_folder` and `$SLURM_JOBDIR` being removed.

## Implementation notes (2026-10-05)
- One file, `SemiSLURM.py`, with subcommands `submit`, `dispatch`, `cancel`, `queue`, `node`, `tmp`, and
  internal `accept`/`run`. Aliases (`WriteAliases.py`): `ssqb`, `ssbatch`, `sdispatch`, and, only where
  `sbatch` doesn't exist, `scancel` and `export TMP=...`. Only `scancel` is offered as job control so far;
  `held` and `priority` in `config.json` are honoured but have no command.
- Stdlib-only and Python 3.10-compatible: accepters and runners use each node's system `python3`.
- `JOBS_ROOT=~/scratch/SemiSLURM/jobs` (per user; `SEMISLURM_JOBS_ROOT` overrides). `$TMP` is
  `/localscratch/$USER/tmp`, else `/tmp/$USER`. Node-local files live in `$TMP/semislurm/`: `accept.lock`,
  `registry/<runner_uid>.json`, `override.json`, and `logs/` (dispatcher and runner logs).
- The dispatcher's `host:pid:uid` lives in `dispatcher_<k>/proc.txt`, not `state.txt`, because a running job's
  `state.txt` belongs to its runner.
- Launch handshake: the dispatcher writes the launch record on yes; the runner waits up to 60 s for it and
  otherwise writes it itself (dispatcher died mid-launch). After a failed ask (SSH error, accepter crash), the
  job isn't re-offered for 90 s, so a runner that did start records itself first.
- Machines are a dispatcher property (`--machines`, default all workstations in `MachineInfo.machine2info`).
  Each machine's GPU types are cached from accepter replies so incompatible jobs aren't offered to it.
- The job sees `SLURM_JOBDIR`, `SEMISLURM_GPUS` (space-separated `nvidia-smi` indices, for
  `tpython_ddp ... --gpus $SEMISLURM_GPUS`), `CUDA_VISIBLE_DEVICES` with `CUDA_DEVICE_ORDER=PCI_BUS_ID`,
  `SEMISLURM_JOB_ID`, `SEMISLURM_NODE`, and `SEMISLURM_RUNNER_UID`. The last tags every job process, so
  processes left by a killed runner (including `setsid`'d ones) are killed on sweep, cancel, and runner exit.
- A job exiting nonzero is `crashed` and not requeued; only jobs whose runner died are requeued.
- Not done yet: the BetterCondIMLE side (UID-first submission, a `SlurmSubmit.py`-like generator).
