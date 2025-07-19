"""Allows viewing job results and/or a SLURM submission script without necessarily
being in the right folder.
"""

import argparse
import os
import os.path as osp
import subprocess
import sys
import UtilsBase

result_search_dirs = ["pretrain_results", "finetune_results"]
slurm_search_dirs = ["slurm"]

def handle_multiple_files(*, files):
    """If multiple files are found, raise an error."""
    files = [f.replace(f"/home/{os.environ['USER']}", "~") for f in files]
    files_str = "\n".join([f"({idx+1}): {f}" for idx,f in enumerate(files)])
    
    user_select = "None" 
    while not user_select.isdigit() or int(user_select) > len(files):
        user_select = input(f"Select file by number:\n{files_str}\n")

    subprocess.run(f"cat {files[int(user_select) - 1]}", shell=True)

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-r", "--result", action="store_true",
        help="Print job results")
    P.add_argument("-s", "--slurm", action="store_true",
        help="Print job submission script")
    P.add_argument("--substr", required=True,
        help="Substring that identifies job")
    P.add_argument("--search_dirs",
        default=[osp.expanduser("~/Development/IMLE-SSL-2"),
                 osp.expanduser("~/Development/IMLE-SSL-Dev")],
        help="Directories to search in")

    try:
        args = P.parse_args()
    except:
        argv = sys.argv[1:]
        for idx,a in enumerate(argv):
            if a == "--substr":
                argv[idx+1] = UtilsBase.strip_left(argv[idx+1], "-")
                break
        args = P.parse_args(argv)


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
        result_file2halfname = {f: osp.basename(f)[len(osp.basename(f)) // 3:] for f in result_files}
        result_file2halfname = {o: h for o,h in result_file2halfname.items() if args.substr in h}
        if len(result_file2halfname.values()) == 1:
            subprocess.run(f"cat {list(result_file2halfname.keys())[0]}", shell=True)
        else:
            _ = handle_multiple_files(files=list(result_file2halfname.keys()))
    
    if args.slurm and len(slurm_files) == 1:
        subprocess.run(f"cat {slurm_files[0]}", shell=True)
    if args.slurm and len(slurm_files) > 1:
        slurm_file2halfname = {f: osp.basename(f)[len(osp.basename(f)) // 3:] for f in slurm_files}
        slurm_file2halfname = {s: h for s,h in slurm_file2halfname.items() if args.substr in h}
        if len(slurm_file2halfname.values()) == 1:
            subprocess.run(f"cat {list(slurm_file2halfname.keys())[0]}", shell=True)
        else:
            _ = handle_multiple_files(files=list(result_file2halfname.keys()))


    print("") # So that the new terminal prompt is on a new line.




        
        