"""Copies a Claude Code session between machines so it can be resumed on either one.

Usage:
    claudesend SESSION OTHER_MACHINE    # send SESSION from this machine to OTHER_MACHINE
    claudesend OTHER_MACHINE SESSION    # fetch SESSION from OTHER_MACHINE (not implemented)
    claudesend SESSION nas              # put SESSION in --handoff_dir for any machine to fetch
    claudesend nas SESSION              # fetch SESSION from --handoff_dir

SESSION is a session ID or a unique prefix of one. The project directory is assumed
to have the same path relative to $HOME on both machines, eg. ~/Development/Foo.
After sending, run `claude --resume SESSION` from that directory on OTHER_MACHINE.
"""
import argparse
from datetime import datetime
import glob
import json
import os
import os.path as osp
import re
import socket
import subprocess
import sys
import tempfile

from UtilsBase import twrite

claude_dir = osp.expanduser("~/.claude")

def path_to_project_dir_name(path):
    """Returns the name Claude Code uses for the folder of transcripts for [path]."""
    return re.sub(r"[^a-zA-Z0-9]", "-", path)

def session_to_jsonl(session):
    """Returns the path to the local transcript of [session], an ID or unique prefix,
    or None if there is no local session matching [session].
    """
    if not re.fullmatch(r"[0-9a-f-]+", session):
        return None
    matches = glob.glob(f"{claude_dir}/projects/*/{session}*.jsonl")
    if len(matches) > 1:
        sys.exit(f"Session prefix {session} is ambiguous: {', '.join(sorted(matches))}")
    return matches[0] if len(matches) == 1 else None

def jsonl_to_cwd(jsonl):
    """Returns the working directory recorded in transcript [jsonl]."""
    with open(jsonl, "r") as f:
        for line in f:
            if "cwd" in (entry := json.loads(line)):
                return entry["cwd"]
    sys.exit(f"Could not find a working directory recorded in {jsonl}")

def jsonl_to_rel_cwd(jsonl):
    """Returns the working directory of transcript [jsonl] relative to $HOME."""
    cwd, home = jsonl_to_cwd(jsonl), osp.expanduser("~")
    if (rel_cwd := osp.relpath(cwd, home)).startswith(".."):
        sys.exit(f"Project directory {cwd} is not under $HOME={home}")
    return rel_cwd

def dir_to_symlinks(path):
    """Returns the paths relative to [path] of all symlinks under [path]."""
    return [osp.relpath(osp.join(root, n), path) for root, dirs, files in os.walk(path)
        for n in dirs + files if osp.islink(osp.join(root, n))]

def rsync_all(transfers, *, host=None, flags=()):
    """Rsyncs each (src, dst) pair in [transfers] whose [src] exists, with each [dst] on
    [host] if given and extra rsync [flags]. --update never clobbers a copy at [dst]
    that is newer, eg. if the session was continued there.
    """
    prefix = "" if host is None else f"{host}:"
    for src, dst in [(s, d) for s, d in transfers if osp.exists(s)]:
        cmd = ["rsync", "-a", "--update", "--itemize-changes", *flags, src, f"{prefix}{dst}/"]
        if not subprocess.run(cmd).returncode == 0:
            sys.exit(f"Failed: {' '.join(cmd)}")

def ssh_run(host, cmd):
    """Returns the stdout of running [cmd] on [host]."""
    result = subprocess.run(["ssh", host, cmd], capture_output=True, text=True)
    if not result.returncode == 0:
        sys.exit(f"Command '{cmd}' failed on {host}: {result.stderr.strip()}")
    return result.stdout.strip()

def send_session(jsonl, *, host):
    """Copies the session with transcript [jsonl] to [host], placing it in the folder
    for the project at the same path relative to $HOME on [host].
    """
    session_id = osp.basename(jsonl).removesuffix(".jsonl")
    cwd, home, rel_cwd = jsonl_to_cwd(jsonl), osp.expanduser("~"), jsonl_to_rel_cwd(jsonl)
    remote_home = ssh_run(host, "echo $HOME")
    remote_cwd = osp.join(remote_home, rel_cwd)
    remote_claude_dir = f"{remote_home}/.claude"
    remote_project_dir = f"{remote_claude_dir}/projects/{path_to_project_dir_name(remote_cwd)}"
    _ = ssh_run(host, f"mkdir -p '{remote_project_dir}' '{remote_claude_dir}/file-history'")

    # The transcript, its sidecar folder of large tool outputs, and checkpoint history
    rsync_all([(jsonl, remote_project_dir),
        (f"{osp.dirname(jsonl)}/{session_id}", remote_project_dir),
        (f"{claude_dir}/file-history/{session_id}", f"{remote_claude_dir}/file-history")], host=host)

    twrite(f"Sent session {session_id} to {host}:{remote_project_dir}")
    twrite(f"To resume: cd ~/{rel_cwd} && claude --resume {session_id}")
    if not remote_home == home:
        twrite(f"[NOTE] $HOME on {host} is {remote_home}, not {home}. The transcript still records {cwd} as its working directory.")

def send_session_to_handoff(jsonl, *, handoff_dir):
    """Copies the session with transcript [jsonl] to a folder in [handoff_dir], along
    with a meta.json recording its project directory relative to $HOME. Symlinks in the
    session's sidecar folder (eg. scratchpad -> $TMP/...) are copied as the data they
    point to, and recorded in meta.json so the fetching machine can recreate them.
    """
    session_id = osp.basename(jsonl).removesuffix(".jsonl")
    dst, sidecar = f"{handoff_dir}/claude_session_{session_id}", f"{osp.dirname(jsonl)}/{session_id}"
    link2target = {l: osp.realpath(f"{sidecar}/{l}") for l in dir_to_symlinks(sidecar)}
    dangling = [l for l, t in link2target.items() if not osp.exists(t)]
    for l in dangling:
        twrite(f"[WARNING] Not sending {sidecar}/{l}: its target {link2target[l]} no longer exists")

    os.makedirs(f"{dst}/file-history", exist_ok=True)
    rsync_all([(jsonl, dst), (f"{claude_dir}/file-history/{session_id}", f"{dst}/file-history")])
    rsync_all([(sidecar, dst)], flags=["--copy-links", *[f"--exclude=/{session_id}/{l}" for l in dangling]])
    meta = dict(session_id=session_id, rel_cwd=jsonl_to_rel_cwd(jsonl), cwd=jsonl_to_cwd(jsonl),
        home=osp.expanduser("~"), host=socket.gethostname(), time=datetime.now().isoformat(timespec="seconds"),
        symlinks={l: t for l, t in link2target.items() if not l in dangling})
    with open(f"{dst}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    twrite(f"Sent session {session_id} to {dst}")
    twrite(f"To fetch on another machine: claudesend nas {session_id[:8]}")

def fetch_session_from_handoff(session, *, handoff_dir):
    """Copies the session matching [session], an ID or unique prefix, from [handoff_dir]
    into the folder for the project at the same path relative to $HOME on this machine.
    Data that was behind a symlink on the sending machine is placed under this machine's
    $TMP/claude-UID/ENCODED_CWD/SESSION_ID/ and symlinked to from the sidecar folder.
    """
    matches = glob.glob(f"{handoff_dir}/claude_session_{session}*")
    if not len(matches) == 1:
        sys.exit(f"Expected one session matching {session} in {handoff_dir}, found {len(matches)}: {', '.join(sorted(matches))}")
    with open(f"{(src := matches[0])}/meta.json", "r") as f:
        meta = json.load(f)

    session_id, home, links = meta["session_id"], osp.expanduser("~"), meta.get("symlinks", {})
    cwd_name = path_to_project_dir_name(osp.join(home, meta["rel_cwd"]))
    project_dir, tmp_dir = f"{claude_dir}/projects/{cwd_name}", f"{tempfile.gettempdir()}/claude-{os.getuid()}"
    link2data = {f"{project_dir}/{session_id}/{l}": f"{tmp_dir}/{cwd_name}/{session_id}/{l}" for l in links}
    if (clashes := [l for l in link2data if osp.exists(l) and not osp.islink(l)]):
        sys.exit(f"Not replacing existing non-symlinks with symlinks: {', '.join(clashes)}")

    os.makedirs(f"{project_dir}/{session_id}", exist_ok=True)
    os.makedirs(f"{claude_dir}/file-history", exist_ok=True)
    os.makedirs(tmp_dir, mode=0o700, exist_ok=True)
    rsync_all([(f"{src}/{session_id}.jsonl", project_dir),
        (f"{src}/file-history/{session_id}", f"{claude_dir}/file-history")])
    rsync_all([(f"{src}/{session_id}", project_dir)], flags=[f"--exclude=/{session_id}/{l}" for l in links])
    for (link, data), l in zip(link2data.items(), links):
        os.makedirs(osp.dirname(data), exist_ok=True)
        rsync_all([(f"{src}/{session_id}/{l}", osp.dirname(data))])
        os.makedirs(osp.dirname(link), exist_ok=True)
        if osp.islink(link): os.remove(link)
        os.symlink(data, link)

    twrite(f"Fetched session {session_id} (sent from {meta['host']} at {meta['time']}) to {project_dir}")
    twrite(f"To resume: cd ~/{meta['rel_cwd']} && claude --resume {session_id}")
    if not meta["home"] == home:
        twrite(f"[NOTE] $HOME on {meta['host']} was {meta['home']}, not {home}. The transcript still records {meta['cwd']} as its working directory.")

def get_args():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    P.add_argument("first", help="SESSION when sending, OTHER_MACHINE when fetching")
    P.add_argument("second", help="OTHER_MACHINE when sending, SESSION when fetching")
    P.add_argument("--handoff_dir", default="~/scratch/$USER",
        help="Directory on a shared filesystem used when OTHER_MACHINE is 'nas'")
    args = P.parse_args()
    args.handoff_dir = osp.expanduser(osp.expandvars(args.handoff_dir))
    return args

if __name__ == "__main__":
    args = get_args()
    if args.first == "nas":
        fetch_session_from_handoff(args.second, handoff_dir=args.handoff_dir)
    elif args.second == "nas":
        if (jsonl := session_to_jsonl(args.first)) is None:
            sys.exit(f"No local session matches {args.first}")
        send_session_to_handoff(jsonl, handoff_dir=args.handoff_dir)
    elif not (jsonl := session_to_jsonl(args.first)) is None:
        send_session(jsonl, host=args.second)
    elif re.fullmatch(r"[0-9a-f-]+", args.second):
        raise NotImplementedError()
    else:
        sys.exit(f"No local session matches {args.first}, and {args.second} does not look like a session ID")
