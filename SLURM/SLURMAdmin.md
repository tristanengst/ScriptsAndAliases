# SLURM admin: node state

How to take a machine out of SLURM and put it back, eg. for a GPU error or maintenance. Plain
`scontrol` does it all. You don't need root, because the user running SLURM is its admin
(`SlurmUser`). Run these from any machine with SLURM (`source ~/.ScriptsAndAliases/SLURM/env.sh`
if the commands aren't on PATH).

`scontrol` wants hostnames as `sinfo -N` shows them, not machine names. `machine_host` (from
`env.sh`) converts a machine name:

```bash
NODE=$(machine_host A7)          # machine name -> hostname, from Private/MachineHostnames.py
```

`NodeName=` also takes a list (`a,b`) or a range (`host-[01-03]`).

## The states

| Goal | Command | Running jobs | New jobs |
|---|---|---|---|
| **Drain**: stop new jobs, let running ones finish | `State=DRAIN Reason=...` | keep running | none |
| **Disable now**: drain, then clear what's running | `DRAIN`, then `preempt`/`scancel` | requeued/cancelled | none |
| **Down**: kill everything on it right away | `State=DOWN Reason=...` | killed: batch requeued, others cancelled | none |
| **Resume**: take jobs again | `State=RESUME` | n/a | yes |

`Reason=` is required for `DRAIN` and `DOWN`. SLURM records it along with who set it and when,
and `sinfo -R` shows all three.

## Drain

```bash
scontrol update NodeName=$NODE State=DRAIN Reason="GPU 1 ECC errors"
```

`sinfo` shows the machine as `drng` (draining) while jobs are still running there, then `drain` once
it's empty. There are no time limits, so a machine can stay draining indefinitely. The drain stays
in place until you resume the machine. It survives restarts of `slurmd` and `slurmctld`.

## Disable now

Drain it first, so nothing new starts there, then clear what's still running:

```bash
scontrol update NodeName=$NODE State=DRAIN Reason="GPU error"
squeue -w $NODE                     # what's still there
preempt JOBID...                    # batch jobs: back to the queue, rerun on another machine
scancel JOBID...                    # srun/salloc or --no-requeue jobs (can't be requeued)
scancel -n reserve_MACHINE_IDX      # reservations never end by themselves
```

- **Blockers** (`MACHINE_blocker_IDX`) end and requeue themselves once their GPU is no longer used
  outside SLURM. They can't start on the machine again until it's resumed.
- **Manual blockers** (`MACHINE_blocker_manual_IDX`, from `block_gpus`) only end when you end them:
  `scancel -n NAME`.

## Down

```bash
scontrol update NodeName=$NODE State=DOWN Reason="GPU fell off the bus"
```

This kills every job on the machine at once. Batch jobs are requeued, and `srun`/`salloc` jobs are
cancelled. Use it when the machine is broken and jobs on it are useless anyway.

DOWN doesn't stick. `slurm.conf` sets `ReturnToService=2`, so a DOWN machine becomes usable again
as soon as its `slurmd` re-registers with the controller (eg. after a reboot or a `start_daemons`).
To keep a machine out for a while, use DRAIN.

SLURM also sets machines DOWN itself, with reason `Not responding`, when it loses contact with
their `slurmd`. They come back by themselves once it's running again.

## Resume

```bash
scontrol update NodeName=$NODE State=RESUME
```

This undoes both DRAIN and DOWN and clears the reason. A DOWN machine only takes jobs once its
`slurmd` is responding. If it isn't, start the daemons on that machine (`Setup.md`).

## Checking state

```bash
sinfo -R                            # machines out of service: reason, who, when
sinfo -N -l                         # every machine's state
scontrol show node $NODE            # full detail: State=, Reason=, GPUs (Gres=)
squeue -w $NODE                     # jobs on it
```

Common `sinfo` states:

| State | Meaning |
|---|---|
| `idle` / `mix` / `alloc` | in service: none, some, or all of its resources allocated |
| `drng` | draining: no new jobs, some still running |
| `drain` | drained and empty |
| `down` | down: set by hand, or not responding |
| `*` suffix (eg. `idle*`) | the controller can't reach its `slurmd` |

## Machines that drain themselves

A machine drains itself when a job's `bin/prolog` or `bin/epilog` fails. For example, `/tmp/$USER-home`
may exist but belong to someone else (`Setup.md`, "Run as part of jobs"). `sinfo -R` shows the reason.
Fix the cause, then resume the machine.

## One bad GPU instead of a whole machine

To keep the machine's other GPUs in service, block just the bad one instead of draining:

```bash
block_gpus --nodes A7 --gpus 1      # manual blocker; preempts whatever is on that GPU
```

`README.md`, "Blocking GPUs by hand", covers this. A GPU that `ffg` shows as an error (eg.
uncorrected ECC errors) already counts as in use, so the regular blocker cron keeps it blocked
without any action from you. Use `block_gpus` when the error doesn't show up that way.
