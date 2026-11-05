#!/usr/bin/env bash
# SSH_ASKPASS helper for CronSSH.sh: answers Duo's "Passcode or option" prompt with 1 (push).
# Refuses any other prompt (eg. a password), so ssh fails instead of guessing.
echo "$(date -Is) askpass prompt: $1" >&2
case "$1" in
    *"Passcode or option"*) echo 1 ;;
    *) exit 1 ;;
esac
