import argparse
import os
import os.path as osp
import shlex
import subprocess
import time

import Utils
import UtilsBase
from UtilsBase import twrite, tqdm

def default_tar_name(fname):
    """Returns a default tar name."""
    realpath = osp.realpath(fname)
    if realpath.startswith("/NAS"):
        cluster_type = "NAS"
    else:
        cluster_type = Utils.get_cluster_type()
    
    date = time.strftime("%Y_%m_%d")
    return f"{fname}_{cluster_type}_{date}.tar"

def tar_imle_ssl_dir(args):
    # dirs_to_tar = ["models_mae", "models_imle", "models_stop", "models_dino", "probes", "finetunes"]
    dirs_to_tar = ["models_imle"]
    scratch_dir = osp.expanduser("/scratch/tme3/IMLE-SSL") if osp.exists("/scratch/tme3/IMLE-SSL") else osp.expanduser("~/scratch/IMLE-SSL")

    if Utils.get_cluster_type() == "nibi":
        out_dir = osp.expanduser("/project/rrg-keli/tme3/IMLE-SSL-storage")
    else:
        out_dir = scratch_dir
    
    dirs_to_tar = [d for d in dirs_to_tar if osp.exists(osp.join(scratch_dir, d))]
    for d in tqdm(dirs_to_tar):
        d = osp.join(scratch_dir, d)
        out = default_tar_name(d)
        _ = tar_folder(argparse.Namespace(**vars(args) | dict(dir=d, out=out)))

def is_newer_than(f, days, ignore_errors=False):
    """Returns if file [f] is newer than [days] days."""
    return osp.getmtime(f) > time.time() + days * 86400

def tar_folder(args):
    def allow_file(f, args):
        """Returns if file [f] should be included in the tar file."""
        if args.ignore_hidden and f.startswith("."):
            return False
        elif args.ignore_no_pt and osp.isdir(f) and not any([f_.endswith(".pt") for f_ in os.listdir(f)]):
            return False
        elif any([ex in f for ex in args.exclude]):
            return False
        else:
            return True

    if args.out is None:
        args.out = default_tar_name(args.dir)
        _ = twrite(f"TarFiles.py: No output file given on folder={args.dir}. Defaulting to out={args.out}")
    
    _ = twrite(f"TarFiles.py:  {args.dir} -> {args.out}")

    files_in_folder = [f for f in os.listdir(args.dir)]

    # By changing the working directory to inside of the folder we want to tar, we
    # can extract the files and folders in it without their full path.
    current_dir = os.getcwd()
    _ = os.chdir(osp.expanduser(args.dir))

    possible_files_to_tar = [f for f in tqdm(files_in_folder,
        desc=f"Finding files new enough",
        dynamic_ncols=True) if is_newer_than(f, args.last_k_days)]
    files_to_tar = {f for f in tqdm(possible_files_to_tar,
        desc=f"Finding non-ignored files",
        dynamic_ncols=True) if allow_file(f, args)}
    files_to_skip = {f for f in possible_files_to_tar if not f in files_to_tar}

    _ = twrite(f"TarFiles.py: files_in_folder={len(files_in_folder)}, newer than {args.last_k_days} days={len(possible_files_to_tar)}, non-ignored={len(files_to_tar)}, ignored={len(files_to_skip)}")

    if args.list_files:
        _ = tqdm.write(f"Files that would be tarred in {args.dir}:\n" + "\n".join(files_to_tar))


    if len(files_to_tar) > 0:
        # Includes all files underneath each folder
        total_files_to_tar = sum(
            len(files_in_folder_to_tar)
            for folder in tqdm(files_to_tar, desc=f"[INFO] Counting files under {args.dir}", dynamic_ncols=True, leave=True)
            for _, _, files_in_folder_to_tar in os.walk(folder)
        )
        twrite(f"[INFO] Counting files under {args.dir} -> total_files_to_tar={total_files_to_tar}")

        file_list = "\n".join(files_to_tar)
        cmd = f"tar -cvf {shlex.quote(args.out)} --files-from=-"
        proc = subprocess.Popen(shlex.split(cmd), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        proc.stdin.write(file_list)
        proc.stdin.close()
        try:
            with tqdm(total=total_files_to_tar, desc=f"[INFO] Tarring (top-level={len(files_to_tar)} total={total_files_to_tar}) files under {args.dir} -> {args.out}", dynamic_ncols=True) as pbar:
                for _ in proc.stdout:
                    pbar.update(1)
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            proc.wait()
            raise KeyboardInterrupt
    else:
        _ = tqdm.write(f"No files to tar in {args.dir}. Skipping.")
    _ = os.chdir(current_dir)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--dir", default=None,
        help="Folder to tar files in")
    P.add_argument("--last_k_days", type=int,
        help="Only tar files after this date", default=-60)
    P.add_argument("--out", default=None,
        help="Name of tar file to create")
    P.add_argument("--ignore_hidden", default=1, type=int, choices=[0, 1],
        help="Ignore hidden files")
    P.add_argument("--ignore_no_pt", default=1, type=int, choices=[0, 1],
        help="Ignore folders that do not contain a .pt file")
    P.add_argument("--imle_ssl_scratch_tar", choices=[0, 1], type=int, default=0,
        help="Generate tarfiles for IMLE-SSL")
    P.add_argument("--exclude", nargs="+", default=[],
        help="Way of specifying files to exclude from the tar.")
    P.add_argument("--list_files", action="store_true",
        help="Just list the files that would be tarred, without actually tarring them.")
    args = P.parse_args()

    if args.last_k_days > 0:
        tqdm.write(f"Got last_k_days={args.last_k_days} -> interpret as negative value")
        args.last_k_days = -1 * args.last_k_days

    if args.imle_ssl_scratch_tar:
        _ = tar_imle_ssl_dir(args)
    else:
        _ = tar_folder(args)
