# Workstation SLURM

Stock SLURM (25.11.8) across the lab workstations, replacing SemiSLURM. Commands and job scripts work as on any SLURM cluster (eg. the Alliance clusters), except for what's below. How it's installed, adding machines, and what each file does: `Setup.md`.

## The setup

- **One user's SLURM, without root.** The daemons run as the user who set it up, and only that user's jobs can run: SLURM can't switch users without root.
- **Every machine is a login node and a compute node.** Submit from any machine that has it; jobs may run on any of them (`sinfo -N` lists them). One machine also runs the controller.
- **Getting the better-SLURM commands:** new shells have them where SLURM is installed (`~/.bashrc`, through `WriteAliases.py`, which skips this if another SLURM is already on PATH). Otherwise: `source ~/.ScriptsAndAliases/SLURM/env.sh`.
- **Homes aren't shared,** and are at different paths on different machines; only `/NAS` is shared. In a job, `$HOME` is the home of the machine running it.
- **Users are their user names, not uids.** The same user has different uids on different machines; SLURM here is patched to match users by name (`Setup.md`, "Users and uids"). Use names, never uid numbers: a uid SLURM shows (eg. `scontrol show job`'s `UserId=`) may be another machine's. `squeue -s -u` only works on the controller's machine.
- **GPUs are shared with people who don't use SLURM.** An external, per-node cron job manages a blocker job for each GPU on a system, named `blocking_job_NODENAME_GPU_INDEX` (eg. `blocking_job_cs-apex-99s_1`). These blocking jobs have highest priority. They allocate GPUs in SLURM, but do not actually use them in any way. When the cron job detects a GPU is used outside of SLURM, it releases the corresponding blocker job. The corresponding blocker job ends and requeues itself when it detects its GPU isn't in use. The external cron job is at `~/.ScriptsAndAliases/SLURM/bin/blocker_cron` (run every 30 s; the crontab lines are in `SLURM/crontab`) and the script for the blocking jobs is at `~/.ScriptsAndAliases/SLURM/bin/blocker.sh`. Both use `SLURM/bin/gpu_busy_outside` to tell whether a GPU is in use outside SLURM. A GPU that `ffg` shows as an error (eg. uncorrected ECC errors) counts as in use, so it stays blocked.

A job that starts on a GPU someone grabbed since the last check is requeued before its script runs (`srun`/`salloc` jobs are cancelled instead).

## Writing jobs

- **Where a job runs.** A job runs at the absolute path of the directory it was submitted from (or of `--chdir`), at that same path on whichever machine it lands on, and fails if that path doesn't exist there. Since homes differ per machine, use a path that works everywhere:
  - `/NAS/...`: submit from there, and write output there; or
  - `--chdir=/tmp/$USER-home/some/path`: runs in `$HOME/some/path` on whichever machine the job lands on (`/tmp/$USER-home` links to each machine's home). `#SBATCH` lines aren't shell-expanded, so in a script write the username out: `#SBATCH --chdir=/tmp/tme3-home/proj`.
  
Relative `-o`/`-e` paths are relative to that directory, and written on the machine running the job. An absolute `-o ~/x` means the *submitting* machine's home.
- **`$SLURM_TMPDIR`**. Each job is allocated `$SLURM_TMPDIR` at `~/slurm_tmpdirs/job_$SLURM_JOB_ID` on the machine running it. This is cleaned when the job starts (`SLURM/bin/prolog`), when it ends (`SLURM/bin/epilog`), and there is an external tool at `~/.ScriptsAndAliases/SLURM/bin/sweep_tmpdirs` to find ones that somehow are leftover that runs periodically (cron, every minute).
- **Set up the environment inside the job script** (eg. activate conda there). A job inherits the submitting shell's environment; `HOME` and SLURM's own `PATH` entries are corrected for the machine it runs on, but other paths (conda, `PYTHONPATH`, ...) may point into the submitting machine's home.
- **Don't use `--export=NONE` or `--export=ALL,VAR=value`.** Either makes SLURM load a login environment on the compute node, which needs root: the job is held ("user_env_retrieval_failed_requeued_held"). Set variables in the shell before `sbatch`, or in the script. Plain `--export=ALL` (the default) is fine.
- **GPUs:** ask for `--gpus-per-node=N`, optionally with `--constraint=MODEL` (eg. `2080s`, or `"a|b"`; `sinfo -o '%N %f'` lists them). Don't ask for the `gN` GRES types; those are for blockers. Which GPUs a job gets is only known once it starts, so job scripts never name them: inside the job, its GPUs are CUDA devices 0..N-1 (`cuda:0`, ...), whichever of the machine's GPUs they are. `CUDA_VISIBLE_DEVICES` holds their UUIDs, not numbers, since it's what hides the machine's other GPUs (no cgroups): CUDA and PyTorch accept that, but code that parses it as integers won't. Jobs without GPUs see none.
- **CPUs come with the GPUs.** Each of a machine's GPUs has a fixed set of CPUs, given by the GPU to CPU map: the `cpus: <GPU index> <CPUs>` lines (one per GPU) of the machine's node file, which lives at `~/.ScriptsAndAliases/Private/SLURM/nodes/<hostname>.conf` (`hostname -s`). `SLURM/bin/add_node` writes those lines if they're missing and otherwise keeps them, so they may be edited (`--reset-cpus` rewrites them). `SLURM/bin/task_prolog` forces its use: it pins every process of every GPU job to its GPUs' CPUs from this map, so this is where the `taskset -c` functionality we'd normally use is implemented. Don't ask for CPUs (`-c`) in GPU jobs, and don't use `tpython_ddp` or `taskset`. Use the plain command with GPU ranges starting from zero, eg.
  ```bash
  #SBATCH --gpus-per-node=2
  torchrun --nproc_per_node=2 --nnodes=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:0 script.py --gpus 0 1
  ```
  SLURM still counts a core per task, but only as bookkeeping. CPU-only jobs aren't pinned.
- **Memory** isn't allocated or tracked: `--mem` only picks a machine with that much.
- **Partitions and limits:** one partition, `gpu` (the default), with no time limit.
- **Nothing is enforced** (no cgroups without root): a job can use more CPUs or memory than it asked for, or GPUs it wasn't given (eg. by changing `CUDA_VISIBLE_DEVICES`). Processes that detach (`setsid`, daemons) escape `scancel`.
- **Interactive `srun`/`salloc` only works between machines the network connects:** not from a workstation onto S1–S3 or V21 (it hangs), since the network blocks those for SLURM (`Setup.md`, "Tunnels"). Use `sbatch` for those machines; `srun` inside a batch job works everywhere.
- **Requeued jobs** wait ~2 minutes before they can start again (SLURM's behaviour).
- **No `sacct`** (no accounting database). Instead, on the controller, every job has a record in `~/.local/slurm/var/log/jobs/<JobID>_<JobName>`: what was submitted, its latest state (eg. `COMPLETED`, `FAILED`, `CANCELLED`; `PENDING` once requeued), how many times it ran (`runs`), and its last run's start, end, nodes and exit code. `~/.local/slurm/var/log/jobcomp.txt` has a line per finished run. Eg. `grep -l state=FAILED ~/.local/slurm/var/log/jobs/*`.

## Choosing GPUs

Every machine's GPUs are a single model, and the machine has a **feature** naming it, derived from
nvidia-smi's name (`gpu_names` can rename it). Jobs pick models with `--constraint` and a number
of GPUs with `--gpus-per-node`; `sinfo -N -o '%N %f %G'` lists the machines in SLURM now, with
their features and GPUs. SLURM knows nothing about GPU memory, so memory needs translating into
models and counts (table). The lab's machines, as of 2026-10:

| Machines | GPUs each | Model (feature) | Memory per GPU |
|---|---|---|---|
| S1, S2, S3 | 10 (if none are broken) | L40S (`l40s`) | 45 GiB |
| V21 | 8 | RTX PRO 6000 Blackwell Server Edition (`rtx6000`) | depends on which RTX 6000: check `nvidia-smi` |
| A4, A9 | 2 | RTX 3090 (`3090`) | 24 GiB |
| A99 | 1 (the other GPU stays outside SLURM) | RTX 2080 SUPER (`2080s`) | 8 GiB |

Jobs always run on one machine here (`torchrun --nnodes=1`). The three usual cases:

1. **Any K GPUs on one machine:**
   ```bash
   #SBATCH --nodes=1
   #SBATCH --gpus-per-node=K
   ```
   Runs on any machine that has K GPUs free in SLURM; add `--constraint` to limit the models.
   A K larger than any machine has (eg. 12) is rejected when submitting ("Requested node
   configuration is not available"); K=4 only fits S1–S3 and V21.
2. **Any single GPU:** `#SBATCH --gpus-per-node=1`. For many independent single-GPU runs, a job
   array (`#SBATCH --array=0-99`, one GPU each) spreads them over every free GPU.
3. **At least X GB of GPU memory:** not expressible directly. Turn it into models and counts:
   - *On one GPU* (the model must fit on each GPU, eg. ≥ 24 GiB): list the models that have it,
     `#SBATCH --gpus-per-node=1` and `#SBATCH --constraint="l40s|3090"`.
   - *In total, across one machine's GPUs:* each model needs K = ⌈X / memory per GPU⌉ GPUs, and a
     job has one K, so group the models that share a K. Eg. X = 80 GiB: 2 × L40S, while RTX 3090s
     would need 4 (no machine has that), so `--gpus-per-node=2 --constraint=l40s`. When the
     acceptable models need different Ks (eg. X = 40 GiB: 1 × L40S or 2 × RTX 3090), either take
     the larger K for all of them (`--gpus-per-node=2 --constraint="l40s|3090"`, using one
     more L40S than needed), or pick one model.

When several machines could run a job, SLURM picks the one with the lowest weight in
`node_weights` (unlisted machines: 1; `sinfo -N -o '%N %w'` shows them), so jobs fill those
machines before the others.

A machine's GPU count in SLURM is what its node file says, not what works today: if a GPU
breaks, SLURM still gives it to jobs until that machine reruns `setup_node --gpus` without it
(`Setup.md`).

