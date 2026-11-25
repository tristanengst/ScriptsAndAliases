#!/usr/bin/env python3
"""File for allowing inter-machine communication via SSH. run_command_on_machine() is the way
to run a command on another machine.

A key challege is that figuring out what other machines are actually called is
surprisingly nontrivial given that we can't control how it's done incredibly weirdly
sometimes.

Hostnames of ComputeCanada clusters are public and stored here. Those of other machines
(workstations, Solar) live in the private submodule at Private/MachineHostnames.py, which
isn't in the public repo; without it, those machines are reached by their SSH names (eg.
~/.ssh/config aliases) instead.

Hosts requiring Duo MFA (eg. Alliance clusters) are reached by logging in once to open a
persistent ControlMaster connection (using the ControlPath that ~/.ssh/config gives every
host), which later ssh commands reuse with no MFA. When run_command_on_machine(mfa=True) finds
no usable connection, it logs in with mfa_login(), which runs ssh in a pseudo-terminal that it
reads, answering Duo's prompt with option 1, sending a push to your phone. This works without a
terminal (eg. from cron or an AI agent's shell).

Unapproved Duo pushes can get the account locked, so each login sends at most one push, and
after MFA_MAX_FAILURES consecutive pushes that don't end in a login, mfa_login() stops sending
them, failing instead, and emails ALERT_EMAIL from Private/UserConfigPrivate.py at most once
per MFA_ALERT_INTERVAL. Running `SSHCommunication.py --mfa_login HOST`, whose user is
presumably around to approve the push, tries anyway; a successful login re-enables automatic
ones. Duo locks a whole account, and every host we reach with MFA (the Alliance clusters)
shares one, so the count covers all hosts: one success resets it. (If hosts with different Duo
accounts are added, key the count by account.) MFA logins take turns, so only one push is
pending at a time, and their state is kept in Private/SSHCommunication_mfa.json, which only the
login holding the turn uses:
    failures:   consecutive Duo pushes that didn't end in a login
    emailed:    epoch seconds of the last alert email (0 if none since the last success)
    last_error: the terminal output of the last failed login

Paramiko would be cleaner, but violates the no non-standard-library desiderata.

Usage:
    SSHCommunication.py                     # Show machine names and this machine's name
    SSHCommunication.py --mfa_login HOST    # Open a persistent connection to HOST, sending a Duo push if needed
    SSHCommunication.py --machine HOST --command COMMAND [--timeout SECONDS]
        # Run COMMAND on HOST via run_command_on_machine(mfa=True), eg. from cron, passing
        # through its output and exit status (255 if HOST can't be reached, 124 on a timeout).
        # COMMAND is one argument, run by the remote login shell, eg.
        #   --machine nibi --command "bash -ic 'python ~/.ScriptsAndAliases/LogSLURMState.py --levelfs_rrg_keli'"
"""
import argparse
import functools
import os
import os.path as osp
import pwd
import re
import math
import shlex
import signal
import socket
import subprocess
import sys
import time

import UtilsBase
from UtilsBase import twrite

######################################################################################
# Machine names to hostnames. Public ones are here, and private ones are merged in from
# the private submodule if it's checked out.
######################################################################################
machine2hostname = dict(
    narval="narval.alliancecan.ca",
    trillium="trillium-gpu.alliancecan.ca",
    fir="fir.alliancecan.ca",
    rorqual="rorqual.alliancecan.ca",
    nibi="nibi.alliancecan.ca",
    vulcan="vulcan.alliancecan.ca",
    killarney="killarney.alliancecan.ca",
    killa="killarney.alliancecan.ca",
    tamia="tamia.alliancecan.ca",)
try:
    import Private.MachineHostnames
    machine2hostname |= Private.MachineHostnames.machine2hostname
    has_private_hostnames = True
except ImportError:
    has_private_hostnames = False

######################################################################################
######################################################################################
######################################################################################

class HostInfoError(Exception):
    """Custom exception for HostInfo-related errors."""
    pass

class CommandTimeoutError(Exception):
    """A command run by run_command_on_machine() took longer than its timeout."""

@functools.cache
def get_hostname():
    result = socket.getfqdn()
    if not "local" in result and "." in result:
        return result

    ##################################################################################
    # Annoying alternative methods to get the hostname in case the code above fails,
    # probably because the system isn't configured quite right in the first place. If
    # any fail, the original result is returned and we hope for the best.
    ##################################################################################
    # Method 1: read IP address from terminal and pass to gethostbyaddr()
    elif not "local" in result and not "." in result:
        try:
            ip_address = subprocess.getoutput(r"ip route get 1.1.1.1 | grep -oP 'src \K[0-9.]+'")
            socket_hostname = socket.gethostbyaddr(ip_address)[0]
        except (OSError, UnicodeError): # eg. no `ip`, so [ip_address] is an error message
            return os.uname().nodename
        if not "local" in socket_hostname and "." in socket_hostname:
            return socket_hostname

    twrite(f"[WARNING] get_hostname() got unexpected result: {result}")
    return result

@functools.cache
def hostname_to_machine(h):
    """Returns the machine name corresponding to hostname [h]. If several machine names
    share [h] (eg. an alias), the first one in [machine2hostname] is returned.
    """
    for m,v in machine2hostname.items():
        if h == v:
            return m
    twrite(f"[WARNING] hostname_to_machine() couldn't find a machine for hostname={h}{'' if has_private_hostnames else ' (Private submodule not checked out?)'}")
    return None

@functools.cache
def get_machine_name(): return hostname_to_machine(get_hostname())

CHECK_CONNECTION_CACHE_SECONDS = 60
_check_connection_cache = dict() # (machine, ssh_args) -> (time checked, result)

def check_connection(machine, *, ssh_args=()):
    """Returns if [machine] can be SSHed to without a prompt, passing [ssh_args] to SSH. Results
    are reused for CHECK_CONNECTION_CACHE_SECONDS, which keeps scripts making many calls fast
    while letting long-running ones notice a connection that has since gone (or come back).
    """
    key = (machine, tuple(ssh_args))
    checked, result = _check_connection_cache.get(key, (-math.inf, None))
    if time.monotonic() - checked < CHECK_CONNECTION_CACHE_SECONDS:
        return result
    cmd = f"ssh -o ConnectTimeout=1 -o BatchMode=yes {' '.join(ssh_args)} {machine} true"
    try:
        result = subprocess.run(cmd, shell=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=5).returncode == 0
    except subprocess.TimeoutExpired:
        result = False
    _check_connection_cache[key] = (time.monotonic(), result)
    return result

@functools.cache
def to_ssh_name(machine):
    """Returns the best SSHable name of [machine]. This is ideally an alias defined
    by ~/.ssh/config, but if not, it will be the FQDN of the machine if it's known,
    or just [machine] if it's not.
    """
    hostname = machine2hostname.get(machine, machine) # FQDN
    
    ##################################################################################
    # Ideally, we can map the hostname to something in ~/.ssh/config, which may better
    # specify how to SSH to it. Otherwise, we can just use the hostname, since it
    # should be an FQDN. It may even be that [machine] is an alias in ~/.ssh/config,
    # in which case we can just use it directly.
    ##################################################################################
    ssh_conf_alias2hostname = get_ssh_conf_alias2hostname()
    if machine in ssh_conf_alias2hostname:
        return machine
    else:
        aliases = [a for a,h in ssh_conf_alias2hostname.items() if h == hostname]
        if aliases:
            # Handle the case of multiple aliases for the same hostname by returning the shortest one
            aliases = sorted(aliases, key=lambda x: len(x))
            return aliases[0]
        else:
            return hostname

def connection_is_open(*, target):
    """Returns if a ControlMaster connection to SSH name [target] is open."""
    return subprocess.run(["ssh", "-O", "check", target], stdin=subprocess.DEVNULL,
        capture_output=True).returncode == 0

def close_connection(*, target):
    """Closes any ControlMaster connection to SSH name [target]."""
    _ = subprocess.run(["ssh", "-O", "exit", target], stdin=subprocess.DEVNULL, capture_output=True)

# Prompts, matched against the end of ssh's unanswered output: a prompt is output ending in a
# colon with no newline after it, ie. ssh waiting for input, so banner text doesn't match.
PROMPT = re.compile(r":\s*$")
DUO_PROMPT = re.compile(r"Passcode or option[^\n]*:\s*$")

MFA_MAX_FAILURES = 3
MFA_ALERT_INTERVAL = 24 * 60 * 60

def login_answering_duo(*, cmd, timeout):
    """Runs ssh command [cmd] in a pseudo-terminal that this reads, so ssh prompts there as if
    a person were logged in. The one prompt answered is Duo's "Passcode or option", once, with
    1, sending a push; any other prompt (eg. a password, or Duo's again after a denied or
    expired push) stops ssh, as does a login taking more than [timeout] seconds. So a login
    sends at most one push. Returns ssh's exit status (nonzero if stopped), whether a Duo push
    was sent, and the terminal output plus why the login failed, if it did.
    """
    import pty
    import select
    import signal

    pid, fd = pty.fork()
    if pid == 0: # Child: become ssh, with the pseudo-terminal as its terminal
        try:
            os.execvp(cmd[0], cmd)
        finally:
            os._exit(127)

    transcript, unanswered, pushed, failure = "", "", False, None
    deadline = time.monotonic() + timeout
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                failure = f"gave up after {timeout}s"
                break
            if not select.select([fd], [], [], remaining)[0]:
                continue
            try:
                data = os.read(fd, 4096)
            except OSError: # EIO: every process has closed the terminal, ie. ssh is done
                data = b""
            if not data:
                break
            text = data.decode(errors="replace")
            transcript, unanswered = transcript + text, unanswered + text
            if DUO_PROMPT.search(unanswered) and not pushed:
                _ = os.write(fd, b"1\n")
                unanswered, pushed = "", True
            elif PROMPT.search(unanswered):
                failure = "stopped at a prompt it doesn't answer"
                break
    finally:
        if failure is not None:
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        status = os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1])
        os.close(fd)

    transcript = transcript.replace("\r\n", "\n")
    if failure is not None:
        transcript += f"\nlogin_answering_duo(): {failure}\n"
        status = status if status > 0 else 1
    return status, pushed, transcript

MFA_STATE_FILE = osp.join(osp.dirname(osp.abspath(__file__)), "Private", "SSHCommunication_mfa.json")

def get_mfa_state(*, name, default):
    """Returns the MFA login state [name], or [default] if it's unset."""
    return UtilsBase.persisted_state_get(name, default=default, persisted_state_file=MFA_STATE_FILE)

def update_mfa_state(**kwargs):
    """Sets the MFA login state to [kwargs]."""
    UtilsBase.persisted_state_update(persisted_state_file=MFA_STATE_FILE, **kwargs)

def maybe_email_mfa_alert(*, target, failures):
    """Emails an alert that MFA logins to [target] have stopped, unless one was sent within
    MFA_ALERT_INTERVAL. A failed send isn't recorded, so the next attempt tries again.
    """
    if time.time() - get_mfa_state(name="emailed", default=0) < MFA_ALERT_INTERVAL:
        return
    body = "\n".join([
        f"SSHCommunication on {socket.getfqdn()} has stopped automatic MFA logins (last to {target}) after",
        f"{failures} consecutive Duo pushes that didn't end in a login, to avoid getting the account",
        "locked. Programs that need to reach it will fail until a login succeeds.",
        "",
        "To retry (sends a Duo push; a successful login re-enables automatic logins):",
        f"    {osp.abspath(__file__)} --mfa_login {target}",
        "",
        "Last failed login:",
        get_mfa_state(name="last_error", default="(none recorded)"),
        "",
        f"You'll get at most one of these emails every {MFA_ALERT_INTERVAL // 3600} hours."])
    try:
        import Private.UserConfigPrivate as C
        unset = [k for k,v in C.EMAIL_SENDER.items() if v is None]
        if unset:
            raise ValueError(f"set {', '.join(unset)} in EMAIL_SENDER in Private/UserConfigPrivate.py")
        UtilsBase.send_email(subject="SSHCommunication: automatic MFA logins stopped",
            body=body, to_email=C.ALERT_EMAIL, **C.EMAIL_SENDER)
    except Exception as e:
        twrite(f"[WARNING] Failed to email MFA alert: {type(e).__name__}: {e}")
        return
    update_mfa_state(emailed=int(time.time()))
    twrite("Emailed alert that automatic MFA logins stopped")

def mfa_login(*, target, force=False, timeout=120):
    """Opens a persistent ControlMaster connection to SSH name [target] if none is open, logging
    in with login_answering_duo(), so a Duo push is sent if the host demands MFA. MFA logins
    take turns, so a caller that waits on another's login to [target] reuses its connection.
    After MFA_MAX_FAILURES consecutive Duo pushes that didn't end in a login, refuses to send
    more unless [force] is True (see the module docstring). Returns a CompletedProcess whose
    returncode is 0 on success (including if a connection was already open) and whose stderr
    explains any failure.
    """
    def result(returncode, stderr=""):
        return subprocess.CompletedProcess(args=["mfa_login", target], returncode=returncode, stdout="", stderr=stderr)

    with UtilsBase.named_lock(name="SSHCommunication_mfa_login"):
        if connection_is_open(target=target):
            return result(0)
        control_path = subprocess.run(["ssh", "-G", target], stdin=subprocess.DEVNULL,
            capture_output=True, text=True).stdout
        control_path = [l.split(maxsplit=1)[1] for l in control_path.splitlines() if l.startswith("controlpath ")]
        if not control_path or control_path[0] == "none":
            raise HostInfoError(f"Can't keep a connection to {target} open: ~/.ssh/config gives it no ControlPath")

        failures = get_mfa_state(name="failures", default=0)
        if failures >= MFA_MAX_FAILURES and not force:
            maybe_email_mfa_alert(target=target, failures=failures)
            return result(1, f"Not logging in to {target}: automatic MFA logins stopped after {failures} consecutive Duo pushes that didn't end in a login. Run `{osp.abspath(__file__)} --mfa_login {target}` to retry.")

        # -f backgrounds ssh after logging in, so the ssh we wait on exits then; the backgrounded
        # master detaches from the terminal and points its stdio at /dev/null.
        status, pushed, transcript = login_answering_duo(timeout=timeout,
            cmd=["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=yes", "-o", "ConnectTimeout=20",
                "-o", "PreferredAuthentications=publickey,keyboard-interactive", "-fN", target])
        if status == 0:
            if failures > 0:
                update_mfa_state(failures=0, emailed=0)
        elif pushed: # Only failures that cost a Duo push risk getting the account locked
            failures += 1
            update_mfa_state(failures=failures, last_error=f"ssh {target}:\n{transcript}")
            transcript += f"\n{failures} consecutive Duo pushes didn't end in a login; automatic logins stop at {MFA_MAX_FAILURES}.\n"
            if failures >= MFA_MAX_FAILURES:
                maybe_email_mfa_alert(target=target, failures=failures)
        return result(status, transcript)

def run_command_on_machine(*, machine, command, ssh_args=[], if_connect_error="error", mfa=False,
    return_process=False, timeout=None):
    """Runs shell [command] on [machine] and returns its output. It's run locally if [machine]
    is this machine, and otherwise over ssh, by the remote user's login shell.

    Args:
    machine             -- machine name in [machine2hostname], or SSHable thing to run
                            command on
    command             -- command to run on machine
    ssh_args            -- additional arguments passed to SSH (probably none needed)
    if_connect_error    -- Something returned on connection error, or "HostInfoError"
                            to raise one
    mfa                 -- if [machine] can't be reached without a login prompt, log in,
                            sending a Duo push if needed (see mfa_login()), rather than
                            treating it as a connection error
    return_process      -- return a subprocess.CompletedProcess with separate stdout and
                            stderr, and returncode (255 if ssh itself failed), rather than
                            the combined output with any trailing newline removed
    timeout             -- seconds after which [command] is stopped, raising
                            CommandTimeoutError; None waits forever. A login (see [mfa]) has
                            its own timeout, before this one starts. On another machine the
                            command is also run under `timeout` there, so it doesn't keep
                            running after ssh is stopped; its exit status 124 (`timeout`'s,
                            so also a command that exits 124 itself) raises CommandTimeoutError
    """
    hostname = machine2hostname.get(machine, machine)
    if hostname == get_hostname():
        cmd, shell = command, True
    else:
        target = to_ssh_name(machine)
        if not check_connection(target, ssh_args=tuple(ssh_args)):
            login = mfa_login(target=target) if mfa else None
            if login is not None and login.returncode == 0:
                pass
            elif if_connect_error == "HostInfoError":
                why = f":\n{login.stderr.strip()}" if login is not None and login.stderr.strip() else ""
                raise HostInfoError(f"Can't SSH to machine={machine} (hostname={hostname}) via {target}{why}")
            else:
                return if_connect_error() if callable(if_connect_error) else if_connect_error
        # [ssh_args] are shell-split, since callers pass eg. "-o ControlMaster=auto" as one
        # element. They come first, so they override ours. BatchMode stops ssh from ever
        # waiting on a prompt if the connection checked above has gone.
        # With a timeout, the remote user's shell ($SHELL, expanded there) runs [command] under
        # `timeout`, so it's stopped there too: ending ssh doesn't end a command without a tty.
        if timeout is not None:
            command = f"timeout -k 10 {math.ceil(timeout)} \"$SHELL\" -c {shlex.quote(command)}"
        cmd, shell = ["ssh", *shlex.split(" ".join(ssh_args)), "-o", "BatchMode=yes", target, command], False

    # cwd="/" fixes an issue at one point; not sure why. The command gets its own process group
    # (session), so on a timeout everything it started is killed, not just its first process.
    # A remote command is given 30 s more than its own timeout to end by itself.
    process = subprocess.Popen(cmd, shell=shell, cwd="/", stdin=subprocess.DEVNULL, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE if return_process else subprocess.STDOUT,
        start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=None if timeout is None else timeout + (0 if shell else 30))
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise CommandTimeoutError(f"Command on machine={machine} took over {timeout} s: {command}")
    if timeout is not None and not shell and process.returncode == 124: # `timeout` stopped it there
        raise CommandTimeoutError(f"Command on machine={machine} took over {timeout} s: {command}")
    result = subprocess.CompletedProcess(args=cmd, returncode=process.returncode, stdout=stdout, stderr=stderr)
    if return_process:
        return result
    return result.stdout[:-1] if result.stdout.endswith("\n") else result.stdout

@functools.cache
def get_ssh_conf_alias2hostname():
    """Returns a dict mapping aliases (Host) to hostnames (HostName) by parsing the
    ~/.ssh/config file on the current machine.
    """
    # ssh finds ~ from the user database, not $HOME, so do the same to read the same file
    ssh_config = osp.join(pwd.getpwuid(os.getuid()).pw_dir, ".ssh", "config")
    if not osp.exists(ssh_config):
        return dict()

    with open(ssh_config, "r") as f:
        lines = f.readlines()
        lines = [l.strip().split("#", 1)[0] for l in lines] # Remove comments
    
    ssh_machine2hostname = dict()
    cur_hosts = None
    for l in lines:
        if l.startswith("Host "):
            cur_hosts = l.strip().split()[1:]
            cur_hosts = [h for h in cur_hosts if not h.startswith("*")]
        elif l.startswith("HostName ") and not cur_hosts is None:
            for h in cur_hosts:
                ssh_machine2hostname[h] = l.split()[1]
            cur_hosts = None
        else:
            continue
        
    return ssh_machine2hostname

if __name__ == "__main__":
    P = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__.split("\n\n", 1)[1])
    P.add_argument("--mfa_login", metavar="HOST",
        help="Open a persistent connection to HOST (eg. a name from ~/.ssh/config), sending a Duo push if needed, even if automatic MFA logins to it have stopped after repeated failures")
    P.add_argument("--machine", metavar="HOST",
        help="Machine to run --command on")
    P.add_argument("--command",
        help="Command to run on --machine, as one argument")
    P.add_argument("--timeout", type=float, default=None,
        help="Seconds after which --command is stopped (default: none)")
    args = P.parse_args()
    if (args.machine is None) != (args.command is None):
        P.error("--machine and --command go together")
    if args.command is not None:
        sys.stdout.reconfigure(line_buffering=True)
        try:
            result = run_command_on_machine(machine=args.machine, command=args.command, mfa=True,
                if_connect_error="HostInfoError", return_process=True, timeout=args.timeout)
        except HostInfoError as e:
            sys.stderr.write(f"{e}\n")
            sys.exit(255)
        except CommandTimeoutError as e:
            sys.stderr.write(f"{e}\n")
            sys.exit(124)
        sys.stdout.write(result.stdout)
        sys.stderr.write(result.stderr)
        sys.exit(result.returncode)
    if args.mfa_login:
        login = mfa_login(target=to_ssh_name(args.mfa_login), force=True)
        sys.stderr.write(login.stderr)
        sys.exit(login.returncode)
    twrite(machine2hostname=machine2hostname)
    twrite(current_machine=get_machine_name())
