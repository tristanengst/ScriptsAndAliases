#!/usr/bin/env bash
# Opens a persistent ControlMaster connection to HOST, so later `ssh HOST ...` commands reuse it
# with no MFA. Does nothing if one is already open. When the host demands Duo MFA (eg. Alliance
# clusters), SSHDuoAskpass.sh answers the prompt with option 1, sending a Duo push to your phone.
# Works without a terminal (eg. from an AI agent's shell). For cron, use CronSSH.sh.
#
# Usage: SSHMFA.sh HOST
set -u
HOST=$1
HERE=$(cd "$(dirname "$0")" && pwd)
ssh -O check "$HOST" 2>/dev/null && exit 0
SSH_ASKPASS="$HERE/SSHDuoAskpass.sh" SSH_ASKPASS_REQUIRE=force DISPLAY=${DISPLAY:-:0} \
    ssh -o ControlMaster=yes -o ControlPersist=yes -o ConnectTimeout=20 \
        -o PreferredAuthentications=publickey,keyboard-interactive -fN "$HOST" < /dev/null
