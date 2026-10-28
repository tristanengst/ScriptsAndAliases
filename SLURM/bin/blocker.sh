#!/bin/bash
# Body of a blocker job (blocking_job_<node>_<idx>). Holds GPU idx while a non-SLURM process uses it,
# then gives it back by requeueing itself held. blocker_cron releases it when needed again.
idx=$1
dir=$(dirname "$0")
while "$dir/gpu_busy_outside" "$idx"; do sleep 30; done
scontrol requeuehold "$SLURM_JOB_ID"
sleep infinity   # killed by the requeue
