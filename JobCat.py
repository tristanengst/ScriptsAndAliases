"""Allows viewing job results and/or a SLURM submission script without necessarily
being in the right folder.
"""

import argparse
import os
import os.path as osp
import subprocess
import UtilsBase

result_search_dirs = ["pretrain_results", "finetune_results"]
slurm_search_dirs = ["slurm"]

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-r", "--result", action="store_true",
        help="Print job results")
    P.add_argument("-s", "--slurm", action="store_true",
        help="Print job submission script")
    P.add_argument("--substr", required=True,
        help="Substring that identifies job")
    P.add_argument("--search_dirs",
        default=[osp.expanduser("~/Development/IMLE-SSL-2")],
        help="Directories to search in")
    args = P.parse_args()

    args.search_dirs = [s for s in args.search_dirs if osp.exists(s)]
    if not len(args.search_dirs):
        raise FileNotFoundError(f"No directories to search, do they all exist?")
    
    result_files, slurm_files = [], []
    for s1 in args.search_dirs:
        s2s = []
        s2s += result_search_dirs if args.result else []
        s2s += slurm_search_dirs if args.slurm else []
        s1s2s = [osp.join(s1,s2) for s2 in s2s if osp.exists(osp.join(s1,s2))]

        for s1s2 in s1s2s:
            for f in os.listdir(s1s2):
                if args.substr in f and f.endswith(".sh"):
                    slurm_files.append(osp.join(s1s2, f))
                elif args.substr in f and f.endswith(".txt"):
                    result_files.append(osp.join(s1s2, f))
                else:
                    pass

    # If the lists of possible results are unique, then print the result. Otherwise,
    # apply a heuristic: --job must come in the second half of the file's base name.
    # If this does not give unique results, raise an error.
    if args.result and len(result_files) == 0:
        print(f"No job result found for {args.substr}")
    if args.slurm and len(slurm_files) == 0:
        print(f"No submission script found for {args.substr}")
    if args.result and len(result_files) == 1:
        subprocess.run(f"cat {result_files[0]}", shell=True)
    if args.result and len(result_files) > 1:
        result_file2halfname = {f: osp.basename(f)[len(osp.basename(f)) // 2:] for f in result_files}
        result_file2halfname = {o: h for o,h in result_file2halfname.items() if args.substr in h}
        if len(result_file2halfname).values() == 1:
            subprocess.run(f"cat {list(result_file2halfname.keys())[0]}", shell=True)
        else:
            result_files_str = "\n".join(result_files)
            raise ValuError(f"Got multiple possible result files:\n{result_files_str}")
    
    if args.slurm and len(slurm_files) == 1:
        subprocess.run(f"cat {slurm_files[0]}", shell=True)
    if args.slurm and len(slurm_files) > 1:
        slurm_file2halfname = {f: osp.basename(f)[len(osp.basename(f)) // 2:] for f in slurm_files}
        slurm_file2halfname = {s: h for s,h in slurm_file2halfname.items() if args.substr in h}
        if len(slurm_file2halfname).values() == 1:
            subprocess.run(f"cat {list(slurm_file2halfname.keys())[0]}", shell=True)
        else:
            slurm_files_str = "\n".join(slurm_files)
            raise ValuError(f"Got multiple possible SLURM files:\n{slurm_files_str}")



        
        