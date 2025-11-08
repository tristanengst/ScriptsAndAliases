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

import FileFinding
import Utils
import UtilsBase
from UtilsBase import twrite, tqdm

known_clusters = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A99", "emily",
    "S1", "S2", "S3",
    "solar", "solar1",
    "trillium", "cedar", "narval", "rorqual", "fir", "nibi", "vulcan"]

def get_args(args=None):
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
    P.add_argument("files", type=str, nargs="*",
        help="Substrings of files or folders to send")
    P.add_argument("--clusters", type=str, nargs="+", default=[],
        help="Clusters to send to")
    P.add_argument("--dry_run", action="store_true")
    P.add_argument("--verbose", action="store_true", help="Print out extra information")

    P.add_argument("--search_dirs", type=str, nargs="+", default=FileFinding.exp_search_dirs,
        help="Directories to search for files matching the substrings")
    P.add_argument("--one_match_per_substr", action="store_true",
        help="When it is ambigous which files to send for a particular substring, only send the one coming from the first search_dir that matches (with the current working directory taking precedence, followed by anything in --extra_search_dirs)")
    P.add_argument("--extra_search_dirs", type=str, nargs="+", default=[],
        help="Extra directories to search for files matching the substrings")
    
    P.add_argument("--output_as_meta", action="store_true")
    P.add_argument("--terminal_size", type=int, default=None,
        help="If provided, use this as the terminal size instead of querying")
    P.add_argument("--argparse_input_file", type=str, default=None,
        help="If provided, read command line arguments from this file")

    
    # If parsing fails, most likely cause is that an element of [files] starts with a
    # dash. In this case, assume that only the first element of the command line
    # arguments should have flags. 
    try:
        args = P.parse_args(args if args else None)
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

    # If there is an --argparse_input_file, then we should read the the file it points
    # to for as the command-line input, and then remove it. This is useful as it can
    # resolve awkward quoting issues.
    if args.argparse_input_file:
        sys_args_file = osp.expanduser(args.argparse_input_file)
        sys_args = UtilsBase.load_file_lite(sys_args_file).split()
        args = get_args(args=sys_args)
        os.remove(sys_args_file)
    
    
    return args

if __name__ == "__main__":
    args = get_args()
    

    ##################################################################################
    ##################################################################################
    ##################################################################################

    # Build the initial part of the rsync string
    rsync_str = "rsync "
    rsync_str += "-" if any([args.r, args.v, args.h, args.a]) else ""
    rsync_str += "r" if args.r else ""
    rsync_str += "v" if args.v else ""
    rsync_str += "h" if args.h else ""
    rsync_str += "a" if args.a else ""
    rsync_str += " ".join([f"--exclude='{e}'" for e in args.exclude]) + " " if args.exclude else ""
    rsync_str += " ".join([f"--include='{i}'" for i in args.include]) + " " if args.include else ""
    rsync_str += f" --info={args.info} " if args.info else ""
    


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

    

    if not args.output_as_meta:
        sending_getting_str = "Sending to" if send_to_cluster else "Getting from"
        twrite(f"[INFO] {sending_getting_str} clusters: {args.clusters}")
        twrite(f"[INFO] files={args.files}")
    
    # If we are sending from the cluster in question, then we can simply find the
    # paths and either (a) output the rsync commands that'd allow another cluster to
    # get the files, or (b) run the rsync commands ourselves to send the files.
    if send_to_cluster:

        # Concatenate search directories and append the current working directory
        # args.search_dirs = [os.getcwd()] + args.extra_search_dirs + args.search_dirs
        args.search_dirs = args.search_dirs + args.extra_search_dirs
        args.files = list(set(args.files))
        
        # These globs represent the files that will actually be sent with rsync
        sources = [FileFinding.file_substr_to_glob(f, search_dirs=args.search_dirs) for f in args.files]
        sources = UtilsBase.flatten(sources)
        _ = twrite(f"[INFO] Files/globs to send: {sources}", quiet=not args.verbose)

        # These files represent where the files will actually end up on the destination
        dests = [FileFinding.file_to_nonambiguous_path(s) for s in sources]
        _ = twrite(f"[INFO] Non-ambiguous paths to send: {dests}", quiet=not args.verbose)

        # Essentially, this is the mapping from destination directories to the files that will
        # be sent to each. Possibly we could use fewer rsync commands by grouping by not the
        # most specific destination directory, but this isn't the usual case.
        dest2files = defaultdict(list)
        for g,d in zip(sources, dests):
            dest2files[f"{osp.dirname(d)}/"].append(g)
        dest2files = {dest: set(files) for dest,files in dest2files.items()}


        # If [output_as_meta] is set, then another cluster is calling essentially
        # asking for the rsync commands that will put the files on the current host on
        # the right place on it. Otherwise, we want to run the commands ourselves to
        # send the files to the destination cluster.
        if args.output_as_meta:
            dest2all_files = {d: UtilsBase.flatten([glob.glob(osp.join(d, f)) for f in fs]) for d,fs in dest2files.items()}
            dest2all_files = {d: set([UtilsBase.path_from_home(f) for f in fs]) for d,fs in dest2all_files.items()}
            dest2files_desc = {d: UtilsBase.list_to_pretty_str(files, one_per_line=True) for d,files in dest2all_files.items()}
            dest2files_desc = "\n".join([f"{UtilsBase.path_from_home(dest)} <- [\n\t{files_desc.strip()}\n]" for dest,files_desc in dest2files_desc.items()])
            _ = UtilsBase.write_meta(dest2files_desc=dest2files_desc)
            
            commands = [f"{rsync_str} {cluster}:{UtilsBase.path_from_home(f)} {UtilsBase.path_from_home(dest)}" for cluster in args.clusters for dest,file_glob in dest2files.items() for f in file_glob]
            
            _ = UtilsBase.write_meta(commands=commands)
            sys.exit(0)
        else:
            commands = [f"{rsync_str} {' '.join(file_glob)} {cluster}:{dest}" for cluster in args.clusters for dest,file_glob in dest2files.items()]

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


        twrite(f"[INFO] Commands to run:\n" + "\n\t".join(commands))

        for c in UtilsBase.tqdm(commands):
            _ = twrite(f"[INFO] {'Would run' if args.dry_run else 'Running'}\t{c}")
            if not args.dry_run:
                result = subprocess.run(f"bash -c '{c}'", shell=True, check=True)

    else:
        import MachineInfo
        import shlex
        import uuid

        def run_cmd(ssh_name, command):
            """Runs [command] on machine [ssh_name], either locally or via ssh. There
            can be a lot of quoting issues, so it's easier to send the command as a
            file that gets read and removed.
            """
            cwd = os.getcwd()
            os.chdir("/") # Not sure why this fixes an issue. Need to change back to the normal directory after running the command
            try:
                rsync_tmp_file = f"rsyncb_cmd_{str(uuid.uuid4()).replace('-', '')[:8]}.txt"
                dir_rsync_tmp_file = osp.join(osp.dirname(__file__), rsync_tmp_file)
                
                UtilsBase.atomic_save_lite(data=command, fname=dir_rsync_tmp_file)
                cmd0 = f"rsync -rv {dir_rsync_tmp_file} {ssh_name}:{rsync_tmp_file}"
                twrite(f"[INFO] Send command file to {ssh_name} via:\n{cmd0}")
                result0 = subprocess.getoutput(cmd0)

                cmd1 = f"python ~/.ScriptsAndAliases/Rsyncb.py --argparse_input_file ~/{rsync_tmp_file}"
                cmd1 = f"ssh -t {ssh_name} \" bash -lic {shlex.quote(cmd1)} \""
                twrite(f"[INFO] Running command to get meta info of files: \n{cmd1}")
                result = subprocess.getoutput(cmd1)
                os.remove(dir_rsync_tmp_file)
                os.chdir(cwd)
            except subprocess.CalledProcessError as e:
                os.chdir(cwd)
                raise e
            return result
                
        cluster2send_command = {c: f"{' '.join(args.files)} {c} --output_as_meta --terminal_size {os.get_terminal_size().columns}" for c in args.clusters}
        cluster2output = {c: run_cmd(c, s) for c,s in cluster2send_command.items()}

        try:
            cluster2output = {c: UtilsBase.load_meta(o) for c,o in cluster2output.items()}
        except:
            _ = twrite(f"[ERROR] Could not parse output from Rsyncb.py on remote cluster, got:\n{cluster2output}")
            sys.exit(1)

        for cluster,output in UtilsBase.tqdm(cluster2output.items()):
            dest2files_desc = output["dest2files_desc"]
            commands = [f"{c}/" if not c.endswith("/") else c for c in output["commands"]]

            _ = twrite(f"[INFO] {cluster} -> {MachineInfo.get_current_machine()}:\n{dest2files_desc}")
            commands_str = "\n\t".join(commands)
            _ = twrite(f"[INFO] {'Would run' if args.dry_run else 'Running'}\n\t{commands_str}")
            
            if not args.dry_run:
                for c in UtilsBase.tqdm(commands):
                    result = subprocess.run(f"bash -c '{c}'", shell=True, check=True)








    







    