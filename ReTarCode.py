"""Re-tars the code on each file found."""

import argparse
import glob
import os
import os.path as osp

def find_code_tarfile(*, fname, args):

    found_files = []
    for s in args.search_dirs:
        s = osp.expanduser(s) if s.startswith("~") else s
        # In this case, [fname] is probably a UID
        if len(fname) == 8:
            fname_ = f"-{fname.strip('*').strip('-')}-"
        else:
            fname_ = fname.strip("*")
        files = glob.glob(f"{s}/*{fname_}*")
        found_files += files
    
    if len(found_files) == 0:
        tqdm.write(f"No files found for fname={fname}. Skipping")
        return None
    elif len(found_files) == 1:
        if osp.exists(f"{found_files[0]}/{args.code_tarfile}"):
            return f"{found_files[0]}/{args.code_tarfile}"
    else:
        tqdm.write(f"No files found for fname={fname}: {found_files}")
        return None 


P = argparse.ArgumentParser()
P.add_argument("--file_substrs", nargs="+",
    help="Substrings uniquely specifying each experiment to update the code of")
P.add_argument("--code_tarfile", default="code.tar",
    help="Name of tarfile to update")
P.add_argument("--search_dirs", default=["~/scratch/IMLE-SSL/models_mae",
    "~/scratch/IMLE-SSL/models_imle", "~/scratch/IMLE-SSL/models_stop",
    "~/scratch/IMLE-SSL/models_dino", "~/scratch/IMLE-SSL/probes",
    "~/scratch/IMLE-SSL/finetunes"],
    help="Directories that could contain each of --file_substrs")
P.add_argument("--things_to_tar", default=["*.py", "original_code", "original_code_stop", "original_code_dino", "MiscCode", "*.txt"], nargs="+",
    help="Things to tar")
P.add_argument("--dry_run", choices=[0, 1], default=0, type=int,
    help="Dry run")
args = P.parse_args()

tarfiles = [find_code_tarfile(fname=f, args=args) for f in args.file_substrs]

for t in tarfiles:
    for th in args.things_to_tar:
        if args.dry_run:
            print(f"Would run: tar -rf {t} {th}")
        else:
            os.system(f"tar -rf {t} {th}")