import argparse

import FileFinding
import UtilsBase

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("-s", "--cmd_start_substr", type=str, default="TrainSSL2.py")
    P.add_argument("--substrs", nargs="+",)
    args = P.parse_args()

    args.substrs = [UtilsBase.strip_right(UtilsBase.strip_left(s, "*"), "*") for s in args.substrs]
    slurm_scripts = [FileFinding.str_to_file(s, file_type="slurm", resolve="half_then_user") for s in args.substrs]

    for substr,s in zip(args.substrs, slurm_scripts):
        script = UtilsBase.load_file_lite(s)

        cmd_lines = [l.strip() for l in script.split("\n") if l.strip().startswith(args.cmd_start_substr) and args.cmd_start_substr in l]
        
        file_info = f"Found: {substr} -> {s}\n"
        print(file_info)

        if len(cmd_lines) == 0: 
            print(f"[WARNING] No command lines found in {s} starting with {args.cmd_start_substr}")
        elif len(cmd_lines) > 1:
            print(f"====================== COMMANDS IN {s} ======================")
            for cl in cmd_lines:
                print(cl)
            print("=============================================================")
        else:
            print(f"{cmd_lines[0]}")