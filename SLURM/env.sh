# Source to use the workstation SLURM: puts sbatch/squeue/... on PATH (they find their config,
# ~/.local/slurm/etc/, compiled in). Also sourced by the scripts in bin/.
#
# Homes aren't shared and live in different places per machine. Scripts that SLURM runs
# (TaskProlog, Epilog) get no HOME, or the submitting machine's, so they source this with
# --node-home: HOME then comes from this machine's node file, written by bin/add_node from $HOME.
SLURM_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SLURM_PRIVATE_DIR=$(cd "$SLURM_DIR/../Private" 2>/dev/null && pwd)/SLURM   # hostnames etc.
SLURM_NODE_FILE=$SLURM_PRIVATE_DIR/nodes/$(hostname -s).conf
if [ "${1:-}" = --node-home ]; then
    HOME=$(sed -n 's/^home=//p' "$SLURM_NODE_FILE" 2>/dev/null)
    [ -n "$HOME" ] || { echo "env.sh: no home= in $SLURM_NODE_FILE" >&2; return 1; }
fi
export HOME
export PATH="${PATH:-/usr/local/bin:/usr/bin:/bin}"
SLURM_HOME=$HOME/.local/slurm
case ":$PATH:" in
    *":$SLURM_HOME/bin:"*) ;;
    *) export PATH="$SLURM_HOME/bin:$SLURM_HOME/sbin:$SLURM_DIR/bin:$PATH" ;;
esac
# Each job's SLURM_TMPDIR is $SLURM_TMPDIR_ROOT/job_<id>, on the machine running it. Created by
# bin/prolog, deleted by bin/epilog when the job ends, and by bin/sweep_tmpdirs if left behind.
SLURM_TMPDIR_ROOT=$HOME/slurm_tmpdirs
# The same path on every machine, linking to that machine's home (made by bin/prolog): jobs
# use --chdir=$SLURM_HOME_LINK/some/path to run in $HOME/some/path wherever they land.
export SLURM_HOME_LINK=/tmp/$(id -un)-home
