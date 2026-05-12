import argparse
import os
import subprocess
import sys
import time
import torch

from UtilsBase import twrite

if __name__ == "__main__":
    P  = argparse.ArgumentParser()
    P.add_argument("-m", "--message", default=f"Reserved by {os.environ['USER']}; please contact for use",
        help="Message to associate with the reservation")
    P.add_argument("-r", "--release_after", type=float, default=24,
        help="Release after this many hours")
    P.add_argument("--gpus", type=int, nargs="+", required=True,
        help="GPU IDs to reserve")
    P.add_argument("--worker", action="store_true",
        help="If set, run as a worker process to hold the reservation rather than creating the reservation")
    args = P.parse_args()

    if args.worker:
        start_time = time.time()
        gpu_dict = {gpu_idx: torch.randn(100).to(torch.device(f"cuda:{int(gpu_idx)}")) for gpu_idx in args.gpus}
        while (time.time() - start_time) < args.release_after * 3600:
            time.sleep(15)
        del gpu_dict
        twrite(f"[INFO] Released GPU reservation for message: {args.message}")
        # sys.exit(0)
    else:
        gpu_str = " ".join([str(int(g)) for g in args.gpus])
        cmd = f"\"exec -a '{args.message}' python ~/.Scripts/ReserveGPUs.py --gpus {gpu_str} --worker --release_after {args.release_after}\""
        _ = twrite(f"[INFO] Creating GPU reservation with command: {cmd}")
        _ = subprocess.run(cmd, shell=True)