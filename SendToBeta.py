import argparse
from collections import defaultdict
import glob
import os
import os.path as osp
import subprocess
import time
import uuid

import UtilsBase
from UtilsBase import twrite

def fname_from_common_path(fname):
    """Returns a path to [fname] that will work from any home directory."""
    abspath = osp.abspath(osp.realpath(osp.expanduser(fname)))

    # For now, assume that ~/scratch/IMLE-SSL/...... will always work.
    imle_ssl_index = abspath.find("IMLE-SSL")
    if imle_ssl_index == -1:
        return fname
    else:
        return f"scratch/{abspath[imle_ssl_index:]}"

def send_file_to_clusters_via_intermediate(*, fname, clusters, intermediate="A4"):
    """Sends file [fname] to the specified clusters [clusters] via the intermediate cluster [intermediate]."""
    if not osp.exists(fname):
        _ = twrite(f"File {fname} does not exist, cannot send to clusters {clusters}")
        return

    fname_common = fname_from_common_path(fname)
    # First send the file to a tempfile on the intermediate cluster
    fname_common = fname_from_common_path(fname)
    fname_base, ext = osp.splitext(fname_common)
    tmp_file = f"__tempfile__{str(uuid.uuid4()).replace('-', '')[:8]}_{osp.basename(fname_base)}.tmp"
    cmd = f"rsync --info=progress2 {fname} {intermediate}:{tmp_file}"
    result = subprocess.run(cmd, shell=True, check=True)
    _ = twrite(f"Sent file {fname} to intermediate={intermediate} as {tmp_file}")

    # Then have the intermediate cluster send it to the final clusters
    mv_cmds = " ; ".join([f"rsync --info=progress2 {tmp_file} {c}:{tmp_file} ; ssh {c} \"mv {tmp_file} {fname_common}\" " for c in clusters])
    rm_cmd = f"rm {tmp_file}"
    cmd = f"ssh {intermediate} ' {mv_cmds} ; {rm_cmd} '"
    result = subprocess.run(cmd, shell=True, check=True)
    _ = twrite(f"Intermediate cluster={intermediate} moved: {tmp_file} to clusters=({', '.join(args.clusters)}) as {fname_common}")
    
def watch_and_send(args):
    """Watches a file and sends it to specified clusters."""
    cluser2send_files = defaultdict(set)
    start_time = time.time()
    while True:
        files = glob.glob(args.watch)
        if len(files) == 0:
            _ = twrite(f"No files found matching {args.watch}")
        else:
            _ = twrite(f"Found files={files} matching {args.watch} -> sending to clusters=({', '.join(args.clusters)}) if not already sent")
            for fname in files:
                clusters = [c for c in args.clusters if not fname in cluser2send_files[c]]
                if len(clusters) == 0:
                    _ = twrite(f"fname={fname} already sent to all clusters=({', '.join(args.clusters)}), skipping")
                else:
                    _ = send_file_to_clusters_via_intermediate(fname=fname, clusters=clusters, intermediate=args.intermediate)
                    cluser2send_files = {c: cluser2send_files[c] | {fname} for c in clusters}

        if UtilsBase.time_since_time(start_time) > UtilsBase.time_str_to_time(args.max_time):
            _ = twrite(f"Stopping watching files after {UtilsBase.time_since_time(start_time)} seconds, max time={args.max_time}")
            break
        else:
            time.sleep(args.check_iter)



if __name__ == "__main__":
    P = argparse.ArgumentParser(description="Send files to clusters or watch for changes.")
    P.add_argument("--watch", default=None,
        help="Glob pattern to to watch for changes")
    P.add_argument("--clusters", nargs="+", choices=["beluga", "cedar", "narval", "rorqual", "nibi", "A4", "S1", "S2", "S3", "solar"],
        help="Clusters to send to")
    P.add_argument("--intermediate", default="A4", choices=["A4", "S1", "S2", "S3", "solar"],
        help="Intermediate cluster to send files to before sending to the final destination")
    P.add_argument("--check_iter", type=int, default=120,
        help="Number seconds to wait between checks for new files")
    P.add_argument("--max_time", default="168:00:00",
        help="Maximum time the file can be watched for changes. Specify as HH:MM:SS")
    args = P.parse_args()

    _ = watch_and_send(args)
        





