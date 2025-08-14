import argparse
import os
import os.path as osp
import time

# TQDM isn't always installed but is really nice given that this can take ours to run
try:
    from tqdm import tqdm
except ImportError:
    class tqdm(object):
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
        def __iter__(self):
            return iter(self.iterable)
        def __next__(self):
            return next(self.iterable)
        def write(self, *args):
            print(*args, **self.kwargs)

import Utils

def default_tar_name(fname):
    """Returns a default tar name."""
    if Utils.is_workstation():
        import MachineInfo
        cluster_type = MachineInfo.hostname_to_machine(os.uname()[1]).lower()
    else:
        cluster_type = Utils.get_cluster_type()
    
    date = time.strftime("%Y_%m_%d")
    return f"{fname}_{cluster_type}_{date}.tar"

def tar_imle_ssl_dir(args):
    dirs_to_tar = ["models_mae", "models_imle", "models_stop", "models_dino", "probes", "finetunes"]
    scratch_dir = osp.expanduser("/scratch/tme3/IMLE-SSL") if osp.exists("/scratch/tme3/IMLE-SSL") else osp.expanduser("~/scratch/IMLE-SSL")
    
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
        if args.ignore_no_pt and osp.isdir(f) and not any([f_.endswith(".pt") for f_ in os.listdir(f)]):
            return False
        return True
        
    if args.out is None:
        args.out = default_tar_name(args.dir)
        _ = tqdm.write(f"TarFiles.py: No output file given on folder={args.dir}. Defaulting to out={args.out}")
    
    _ = tqdm.write(f"TarFiles.py:  {args.dir} -> {args.out}")

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

    _ = tqdm.write(f"TarFiles.py: files_in_folder={len(files_in_folder)}, newer than {args.last_k_days} days={len(possible_files_to_tar)}, non-ignored={len(files_to_tar)}, ignored={len(files_to_skip)}")

    if len(files_to_tar) > 0:
        for f in tqdm(files_to_tar,
            desc=f"Tarring files in {args.dir}",
            dynamic_ncols=True):

            _ = os.system(f"tar -rf {args.out} {f}")
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
    args = P.parse_args()

    if args.last_k_days > 0:
        tqdm.write(f"Got last_k_days={args.last_k_days} -> interpret as negative value")
        args.last_k_days = -1 * args.last_k_days

    if args.imle_ssl_scratch_tar:
        _ = tar_imle_ssl_dir(args)
    else:
        _ = tar_folder(args)
