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
# tunnel_port NODE FILE: the port NODE's tunnel uses on the controller (Setup.md, "Tunnels"), or
# nothing if NODE's node file FILE has no tunnel. tunnel_port=P in FILE sets it by hand; tunnel=yes
# derives it from the name, the same on every machine: 46900 + (cksum of the name mod 1000).
tunnel_port() {
    local p
    p=$(sed -n 's/^tunnel_port= *\([0-9]*\).*/\1/p' "$2" 2>/dev/null)
    if [ -z "$p" ] && grep -qx 'tunnel=yes' "$2" 2>/dev/null; then
        p=$((46900 + $(printf %s "$1" | cksum | cut -d' ' -f1) % 1000))
    fi
    echo "$p"
}
# node_addr NODE: the address to reach NODE (a short hostname, as in slurm.conf) by, or nothing to
# use the name itself (Setup.md, "Addresses"). addr=IP in NODE's node file, for machines without a
# DNS record, everywhere; otherwise, only where NODE doesn't resolve here (eg. a search domain
# other than the lab's), its full name from ../Private/MachineHostnames.py if that resolves.
node_addr() {
    local a
    a=$(sed -n 's/^addr= *\([^ #]*\).*/\1/p' "$SLURM_PRIVATE_DIR/nodes/$1.conf" 2>/dev/null)
    if [ -z "$a" ] && ! getent ahostsv4 "$1" > /dev/null; then
        a=$(sed -n "s/^ *[A-Za-z0-9_]* *= *\"\($1\.[^\"]*\)\".*/\1/p" "$SLURM_PRIVATE_DIR/../MachineHostnames.py" 2>/dev/null | head -1)
        [ -n "$a" ] && getent ahostsv4 "$a" > /dev/null || a=
    fi
    echo "$a"
}
# machine_name HOST: HOST's machine name (eg. cs-apex-99s -> A99), its first key in
# ../Private/MachineHostnames.py, or HOST if it has none. machine_host NAME: the reverse (eg. A99 ->
# cs-apex-99s), or NAME if it isn't a machine name.
machine_name() {
    local m; m=$(sed -n "s/^ *\([A-Za-z0-9_]*\) *= *\"$1\..*/\1/p" "$SLURM_PRIVATE_DIR/../MachineHostnames.py" 2>/dev/null | head -1)
    echo "${m:-$1}"
}
machine_host() {
    local h; h=$(sed -n "s/^ *$1 *= *\"\([^.\"]*\).*/\1/p" "$SLURM_PRIVATE_DIR/../MachineHostnames.py" 2>/dev/null | head -1)
    echo "${h:-$1}"
}
# The same path on every machine, linking to that machine's home (made by bin/prolog): jobs
# use --chdir=$SLURM_HOME_LINK/some/path to run in $HOME/some/path wherever they land.
export SLURM_HOME_LINK=/tmp/$(id -un)-home
