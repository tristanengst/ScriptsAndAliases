# SLURM multicluster

Runs equivalent copies ("siblings") of a job on several clusters (apex, solar, Alliance); a
controller on apex lets the first to start run and makes the others wait, and holds them once
the experiment finishes. Tested across apex, solar and narval.

## TL;DR: integrating `SlurmSubmit.py`

Submit one copy per cluster (any resources, account or time limit, same computation), each with
this comment:

```python
sys.path.insert(0, osp.expanduser("~/.ScriptsAndAliases"))
import UtilsBase
comment = UtilsBase.dict_to_job_comment(dict(uid=uid, exp_name=exp_name,
    exp_folder="~/" + osp.relpath(exp_folder, osp.expanduser("~")), mc_cont="run", mc_time="NA"))
subprocess.run(["sbatch", f"--comment={comment}", job_script])
```

and make this the job script's first command:

```bash
python ~/.ScriptsAndAliases/SLURMMultiCluster/SLURMMultiClusterCheck.py || true
```

- `uid`: identical on every copy, and only on copies of this computation.
- `exp_folder`: starts with `~/`, as reached from home on the cluster the copy runs on (eg.
  `~/scratch/...`, not a resolved `/NAS/...`). The job writes `finished.txt` there when, and
  only when, the experiment is complete.
- Encode only with `dict_to_job_comment()` (it escapes `'` in values); at most 1024 characters.
  Pass it as an `sbatch` argument (argument list, no shell). In an `#SBATCH --comment="..."`
  line instead, double every backslash: `sbatch` drops them there.
- The job resumes from its latest checkpoint when run again: a copy can be requeued at any time.

A job missing any of `uid`, `exp_folder`, `mc_cont`, `mc_time` isn't multicluster, and just
runs; the check line does nothing for it, so it can go in every job script.

## How it works

- **The controller** reads every cluster's multicluster jobs and writes a verdict into each
  pending one's comment (`mc_cont`): `'requeue'` if a sibling is running, `'requeuehold'` (plus
  a hold) if the experiment finished, else `'run'`, with the time (`mc_time`).
- **The check**, when a job starts, carries out a fresh stand-down verdict on its own job
  (`scontrol requeue` / `requeuehold`); otherwise, or on any error, the job runs.
- **Bias toward running:** a failure anywhere (controller down, cluster unreachable, stale
  verdict, missing script) lets jobs run. Several copies running at once is acceptable.
- **Nothing leaves a queue unless it's definitely finished:** siblings of a finished experiment
  are held, never cancelled.

## Comment keys

| Key | Written by | Values | Meaning |
|---|---|---|---|
| `uid` | submission | str | Same uid, same computation |
| `exp_folder` | submission | path starting `~/` | `finished.txt` there means the experiment finished |
| `exp_name` | submission | str | For people; not used |
| `mc_cont` | controller (`'run'` at submission) | `'run'`, `'requeue'`, `'requeuehold'` | Verdict for the job if it starts |
| `mc_time` | controller (`'NA'` at submission) | Unix seconds | When the verdict was last confirmed; a verdict older than the check's `--mc_time_stale_after_seconds` (1200) is ignored |
| `mc_why` | controller | str | Why a job got `'requeuehold'` |
| `mc_ignore` | you | 0/1 | 1: the controller acts as if the job didn't exist, and the check lets it run |
| `mc_protected` | you | 0/1 | 1: a sibling (can make others wait, can finish the experiment), but never stood down or held |

Change keys with `scu` (fails on a non-JSON comment before changing anything; with siblings on
one cluster, match jobs by ID or name, not uid):

```bash
scu --comment_update="{'mc_protected': 1}" JOBS...   # old | new
scu --comment_set mc_ignore=1 JOBS...                # one key
scu --comment_del mc_ignore JOBS...
scu --comment=NEW_COMMENT JOBS...                    # replace
```

A job held because its experiment finished is left alone from then on. Released by hand, it
runs once its `mc_time` is stale, or at once with `mc_ignore` set.

## Running the controller

```
*/5 * * * * python ~/.ScriptsAndAliases/SLURMMultiCluster/SLURMMultiClusterController.py --ensure_running --clusters apex solar narval fir
```

`--ensure_running` wakes the running controller or starts one (detached; log in
`~/.local/state/SLURMMultiCluster/controller.log`). Waking it with different `--clusters` makes
it exit, so cron starts one with the new list. `--status` shows each thread's last read and
error; `--stop` stops it; `--dry_run` writes nothing.

The cluster this machine belongs to is run locally; the others over SSH
(`SSHCommunication.run_command_on_machine`, Duo push if needed) at the `~/.ssh/config` Host of
the same name, in `bash -ic`. Commands time out after `--timeout_seconds` (300).

## Internals

| File | |
|---|---|
| `SLURMMultiClusterController.py` | The controller, and `--remote_write_read`, run on each cluster |
| `SLURMMultiClusterCheck.py` | The in-job check (standard library, `UtilsBase`, `MultiClusterComment` only) |
| `MultiClusterComment.py` | Comment rules shared by the two |

Code here imports only `UtilsBase`, `Utils` and `SSHCommunication` from outside the folder.

One process, with threads that talk through an inbox (`queue.Queue`) and a wake event
(`threading.Event`) each; senders put a message then set the event, owners wait, clear, then
empty the inbox. An error in one thread only loses that pass.

- **One thread per cluster.** When woken (by a wake message or new verdicts), and if it has
  verdicts or last ran over `--min_seconds_between_reads` (240) ago, it runs
  `--remote_write_read` there in one SSH call: write the verdicts (merged into each comment as
  it is then; `'requeuehold'` only after the hold works), renew `mc_time` on pending
  multicluster jobs except finished-held ones, read the jobs, and report jobs that left the
  queue since the last read (kept in `~/.local/state/SLURMMultiCluster/queue_state.json` on the
  cluster) whose `exp_folder` has `finished.txt`. Changes go to the compute thread; on failure
  the verdicts are kept for the next try.
- **The compute thread** works out verdicts and sends each cluster thread the ones that differ
  from its jobs' comments. A started sibling outranks a pending one; `--ranking jobid` orders
  siblings in the same state (TODO: `rgus_remaining`). Running jobs of a cluster not read for
  `--running_info_expiry_seconds` (1200) don't count. Finished experiments are recorded only in
  held siblings' comments. It waits for every cluster's first attempt after starting.
- **The main thread** answers an abstract Unix socket (no file; other users ignored). A wake
  message restarts any dead thread, then wakes all of them.

The controller keeps nothing on disk; a restart costs one round of reads.

**Known gaps** (all fail toward running): a job submitted and finished between two reads isn't
seen; a finished experiment whose siblings are all on unreachable clusters isn't recorded.
Holds from apex show as `JobHeldAdmin` there (the controller's user is apex's SLURM admin).

**Open:** `rgus_remaining` ranking (needs a GPU → RGU table); chunking.
