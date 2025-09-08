"""Intelligent way of sending files between clusters when they are guarunteed to be on
the same path with respect to ~/scratch on both.

Syntax is:

python Rsyncb.py [optional rsync flags] file_or_folder1_to_send_substring ... file_or_folderN_to_send_substring cluster
"""
import argparse
import glob
import os
import os.path as osp
import subprocess
import sys
from collections import defaultdict

import UtilsBase
from UtilsBase import twrite

known_clusters = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A99", "emily",
    "S1", "S2", "S3", "solar", "trillium", "cedar", "narval", "rorqual", "fir", "nibi"]

def get_current_shell():
    """Returns the current shell."""
    shell = os.getenv("SHELL", "bash")
    return osp.basename(shell)

def file_substr_to_glob(f, *, args):
    """Returns the list of files that match the substring [f], but in a way where existing globs would be treated nicely by bash."""
    if osp.exists(f):
        return [f]
    else:
        fw = "*" + UtilsBase.strip_left(UtilsBase.strip_right(f, "*"), "*") + "*"
        
        globs = []
        all_matched_files = set()
        for d in args.search_dirs:
            if not osp.exists(d):
                continue

            test_file = osp.join(d, fw)
            matched_files = glob.glob(test_file)
            if matched_files:
                globs.append(test_file)
                all_matched_files |= set(matched_files)
                
                if args.one_match_per_substr:
                    break
        
        if len(globs) == 0:
            raise ValueError(f"No files found matching substring {f} in search_dirs={args.search_dirs}")
        elif not "*" in f and len(all_matched_files) > 1:
            globs = "\n\t".join(sorted(globs))
            _ = twrite(f"[ERROR] file={f} matches multiple possibilities but does not contain *:\nglobs=\n{sorted(globs)}\nall_matched_files=\n{sorted(all_matched_files)}")
        else:
            return globs

def file_to_nonambiguous_path(f):
    """Returns the non-ambiguous path to file [f]."""
    abs_f = osp.abspath(osp.realpath(osp.expanduser(f)))
    
    abs_prefix2non_ambiguous_prefix = {
        osp.abspath(osp.expanduser("~/scratch")): "~/scratch",
        osp.abspath(osp.expanduser("~")): "~",
        "/NAS": "~/scratch",
    }

    non_ambiguous_f = [osp.join(v, abs_f[len(k)+1:]) for k,v in abs_prefix2non_ambiguous_prefix.items() if abs_f.startswith(k)]
    non_ambiguous_f = list(set(non_ambiguous_f))
    if len(non_ambiguous_f) == 0:
        return f
    elif len(non_ambiguous_f) > 1:
        _ = twrite(f"[ERROR] file={f} has multiple non-ambiguous paths: {sorted(non_ambiguous_f)}")
        return f
    else:
        return non_ambiguous_f[0]

if __name__ == "__main__":
    P = argparse.ArgumentParser(add_help=False)
    P.add_argument("--help", action="help", help="Show this help message and exit")

    # Flags for rsync that work exactly as in rsync
    P.set_defaults(r=True, v=False, h=True, a=False, info="progress2")
    P.add_argument("-r", action="store_true", dest="r",)
    P.add_argument("-no-r", action="store_false", dest="r")
    P.add_argument("-v", action="store_true", dest="v",)
    P.add_argument("-no-v", action="store_false", dest="v")
    P.add_argument("-h", action="store_true", dest="h",)
    P.add_argument("-no-h", action="store_false", dest="h")
    P.add_argument("-a", action="store_true", dest="a",)
    P.add_argument("-no-a", action="store_false", dest="a")
    P.add_argument("--info", type=str, dest="info", default="progress2")
    P.add_argument("--no-info", action="store_const", const=None, dest="info")

    # Flags for rsync whose behavior is different from rsync
    P.add_argument("--exclude", type=str, nargs="+", default=[],
        help="Patterns to exclude, e.g., '*.tmp' or 'checkpoints'")
    P.add_argument("--include", type=str, nargs="+", default=[],
        help="Patterns to include, e.g., '*.pt' or 'checkpoints'")
    
    # Note that in zsh, files with * in them would not be interpreted sensibly
    # (ie. bash-style.) so we will have to expand them manually.
    P.add_argument("files", type=str, nargs="+",
        help="Substrings of files or folders to send")
    P.add_argument("--clusters", type=str, nargs="+", default=[],
        help="Clusters to send to")
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("--verbose", action="store_true", help="Print out extra information")

    P.add_argument("--search_dirs", type=str, nargs="+", default=[
        osp.expanduser("~/scratch/IMLE-SSL"),
        osp.expanduser("~/scratch/IMLE-SSL/models_imle"),
        osp.expanduser("~/scratch/IMLE-SSL/models_mae"),
        osp.expanduser("~/scratch/IMLE-SSL/finetunes")],
        help="Directories to search for files matching the substrings")
    P.add_argument("--one_match_per_substr", action="store_true",
        help="When it is ambigous which files to send for a particular substring, only send the one coming from the first search_dir that matches (with the current working directory taking precedence, followed by anything in --extra_search_dirs)")
    P.add_argument("--extra_search_dirs", type=str, nargs="+", default=[],
        help="Extra directories to search for files matching the substrings")
    
    # If parsing fails, most likely cause is that an element of [files] starts with a
    # dash. In this case, assume that only the first element of the command line
    # arguments should have flags. 
    try:
        args = P.parse_args()
    except:
        fixed_argv = []
        for a in sys.argv[1:]:
            if a.startswith("--"):
                fixed_argv.append(a)
            elif a.startswith("-") and not (set(a[1:]) - set("rvha")):
                fixed_argv.append(a)
            else:
                fixed_argv.append(UtilsBase.strip_left(a, "-"))
        args = P.parse_args(fixed_argv)

    # If --clusters wasn't specified, then either the first or last element of
    # --files is the cluser. If there's a colon or @ symbol in either element, then
    # that's the one. Otherwise, find the one that's a member of [known_clusters]
    if not args.clusters:
        send_to_cluster = args.files[-1]
        send_from_cluster = args.files[0]
        if not ((":" in send_to_cluster) or ("@" in send_to_cluster) or (send_to_cluster.split("@")[-1].split(":")[0] in known_clusters)):
            send_to_cluster = False
            args.files = args.files[1:]
        if not ((":" in send_from_cluster) or ("@" in send_from_cluster) or (send_from_cluster.split("@")[-1].split(":")[0] in known_clusters)):
            send_from_cluster = False
            args.files = args.files[:-1]
            
        
        if not send_to_cluster and not send_from_cluster:
            raise ValueError(f"Could not deduce cluster from command line arguments, got not clusters: {args.files}")
        elif send_to_cluster and send_from_cluster:
            raise ValueError(f"Could not deduce cluster from command line arguments, got multiple clusters: {args.files}")
        else:
            args.clusters = [send_to_cluster if send_to_cluster else send_from_cluster]
    
    twrite(f"[INFO] Sending to clusters: {args.clusters}", quiet=not args.verbose)
    twrite(f"[INFO] files={args.files}", quiet=not args.verbose)

    # Concatenate search directories and append the current working directory
    args.search_dirs = [os.getcwd()] + args.extra_search_dirs + args.search_dirs
    args.files = list(set(args.files))
        
    rsync_str = "rsync "
    rsync_str += "-" if any([args.r, args.v, args.h, args.a]) else ""
    rsync_str += "r" if args.r else ""
    rsync_str += "v" if args.v else ""
    rsync_str += "h" if args.h else ""
    rsync_str += "a" if args.a else ""
    rsync_str += " ".join([f"--exclude='{e}'" for e in args.exclude]) + " " if args.exclude else ""
    rsync_str += " ".join([f"--include='{i}'" for i in args.include]) + " " if args.include else ""
    rsync_str += f" --info={args.info} " if args.info else ""

    # These globs represent the files that will actually be sent with rsync
    sources = UtilsBase.flatten([file_substr_to_glob(f, args=args) for f in args.files])
    _ = twrite(f"[INFO] Files/globs to send: {sources}", quiet=not args.v)

    # These files represent where the files will actually end up on the destination
    dests = [file_to_nonambiguous_path(s) for s in sources]
    _ = twrite(f"[INFO] Non-ambiguous paths to send: {dests}", quiet=not args.v)

    # Essentially, this is the mapping from destination directories to the files that will
    # be sent to each. Possibly we could use fewer rsync commands by grouping by not the
    # most specific destination directory, but this isn't the usual case.
    dest2files = defaultdict(list)
    for g,d in zip(sources, dests):
        dest2files[f"{osp.dirname(d)}/"].append(g)

    # If there are multiple clusters, we want to open a connection to each immediately.
    # This ensures that any MFA authentication happens presently, rather than at some
    # indeterminate time in the future when, say, one might be asleep.
    # This requires your ssh config to have ControlMaster enabled well

    # TODO: not implemented yet
    # if len(args.clusters) > 1:
    #     _ = twrite(f"[INFO] Multiple clusters={args.clusters} -> open connections now")
    #     for c in args.clusters:
    #         cmd = f"ssh -t {c} bash 'Connected to {c}'"
    #         result = subprocess.run(cmd, shell=True, check=True)

    if send_to_cluster:
        commands = [f"{rsync_str} {' '.join(file_glob)} {cluster}:{dest}" for cluster in args.clusters for dest,file_glob in dest2files.items()]
    elif send_from_cluster:
        commands = [f"{rsync_str} {cluster}:{dest}/{f} {dest}" for cluster in args.clusters for dest,file_glob in dest2files.items() for f in file_glob]
    else:
        raise ValueError()


    twrite(f"[INFO] Commands to run:\n" + "\n\t".join(commands))

    for c in UtilsBase.tqdm(commands):
        _ = twrite(f"[INFO] {'Would run' if args.dry_run else 'Running'}\t{c}")
        if not args.dry_run:
            result = subprocess.run(f"bash -c '{c}'", shell=True, check=True)







    