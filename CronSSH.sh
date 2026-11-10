#!/usr/bin/env bash
# Runs a command on HOST over ssh non-interactively (eg. from cron), in the remote user's
# interactive ~/.bashrc shell. Reuses/creates a persistent ControlMaster connection; when none
# exists and the host demands Duo MFA (eg. Alliance clusters), answers the prompt with option 1
# so a Duo push is sent to your phone. Approve it and later runs reuse the connection.
#
# Usage: CronSSH.sh [--refresh] HOST COMMAND...   (COMMAND must not contain single quotes)
#   --refresh: first close any existing master connection (waiting for in-flight runs), forcing
#              a fresh login and so a Duo push, eg. daily at a time you're around to approve it.
#   eg.  CronSSH.sh nibi python '~/.ScriptsAndAliases/LogSLURMState.py' --levelfs_rrg_keli
set -u
REFRESH=0
if [[ ${1:-} == --refresh ]]; then REFRESH=1; shift; fi
HOST=$1; shift
CMD="$*"
HERE=$(cd "$(dirname "$0")" && pwd)

LOCK="$HOME/.cache/CronSSH_$HOST.lock"
mkdir -p "$(dirname "$LOCK")"

echo "=== $(date -Is) $HOST: $CMD"
if (( REFRESH )); then
    echo "Refreshing: closing existing master connection to $HOST"
    flock "$LOCK" ssh -O exit "$HOST" 2>&1 || true
fi
export SSH_ASKPASS="$HERE/SSHDuoAskpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=${DISPLAY:-:0}
# -n: don't wait if a previous run is still waiting on a push. -o: don't let the backgrounded
# master inherit the lock.
exec flock -n -o "$LOCK" \
    ssh -o ControlMaster=auto -o ControlPersist=yes -o ConnectTimeout=20 \
        -o PreferredAuthentications=publickey,keyboard-interactive \
        "$HOST" "bash -ic '$CMD'" < /dev/null
