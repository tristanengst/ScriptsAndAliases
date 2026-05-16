import argparse
from collections import defaultdict
from datetime import datetime
import json
import os
import os.path as osp

import MachineInfo
import SSHCommunication
import UtilsBase
from UtilsBase import twrite, tqdm

# All other clusters need to have their directories accessed via ls
diskusage_report_clusters = ["vulcan", "narval", "fir"]

# No rrg-keli output on these clusters
skip_rrg_on_clusters = ["narval", "fir"]

def per_user_diskusage_report_to_info(diskusage_report):
    """Returns a dictionary mapping user -> project -> a dict with file_count,
    file_storage_GiB keys.
    """
    def file_size_unit_to_GiB(*, size_num, size_unit):
        """Returns the size the files taking up [size_num] number of [size_unit] units
        in GiB.
        """
        size_num = float(size_num)
        unit2GiB = dict(B=1/(1024**3), KiB=1/(1024**2), MiB=1/1024, GiB=1, TiB=1024, PiB=1024**2)
        if not size_unit in unit2GiB:
            raise ValueError(f"Unknown size unit: {size_unit}")
        return size_num * unit2GiB[size_unit]


    user2project2usage = defaultdict(lambda: defaultdict(dict))
    lines = diskusage_report.strip().split("\n")
    lines = [l.strip() for l in lines]
    
    cur_project = None
    for line in lines:
        if line.startswith("Breakdown for project"):
            cur_project = UtilsBase.strip_left(line, "Breakdown for project").split()[0]
        elif line.startswith("Total"):
            cur_project = None
        elif cur_project is None or any(line.startswith(prefix) for prefix in ["User", "---"]):
            continue
        else:
            user, file_count, file_size_num, file_size_unit = line.split()[:4]
            user2project2usage[user][cur_project] = dict(file_count=int(file_count),
                file_storage_GiB=file_size_unit_to_GiB(size_num=file_size_num, size_unit=file_size_unit))
    
    user2project2usage = {user: dict(project2usage) for user, project2usage in user2project2usage.items()}
    return user2project2usage

def ls_diskusage_report_to_info(*, ls_output, ls_path):
    """Returns a dictionary mapping user -> project -> a dict with file_count,
    file_storage_GiB keys set to 'exists'."""
    project = osp.basename(osp.abspath(ls_path))
    user2project2usage = defaultdict(dict)

    for user in ls_output.strip().split():
        user2project2usage[user] = dict()
        user2project2usage[user][project] = dict(file_count="?", file_storage_GiB="?")
    user2project2usage = {user: dict(project2usage) for user, project2usage in user2project2usage.items()}
    return user2project2usage

def user_data_to_str(cluster2project2usage):
    strs = []
    for cluster,project2usage in cluster2project2usage.items():
        for project,usage in project2usage.items():
            key = f"{cluster}:{project}"
            file_count, file_storage_GiB = usage["file_count"], usage["file_storage_GiB"]
            if file_count == "?":
                strs.append(f"{key}=exists")
            else:
                strs.append(f"{key}=({file_count} files, {file_storage_GiB}GiB)")
    return "\t".join(strs)

def user_data_to_totals(cluster2project2usage):
    total_file_count = 0
    total_file_storage_GiB = 0.0
    for cluster,project2usage in cluster2project2usage.items():
        for project,usage in project2usage.items():
            file_count, file_storage_GiB = usage["file_count"], usage["file_storage_GiB"]
            total_file_count += file_count if UtilsBase.is_numeric(file_count) else 1
            total_file_storage_GiB += file_storage_GiB if UtilsBase.is_numeric(file_storage_GiB) else 1e-3
    return dict(total_file_count=total_file_count, total_file_storage_GiB=total_file_storage_GiB)

def get_ls_data_from_cluster(*, cluster):
    possible_ls_dirs = ["/project/aip-keli", "/project/def-keli", "/project/rrg-keli"]

    search_cmd_template = """for dir in LS_DIR_PATH/*/; do
    user=$(basename "$dir")
    if id "$user" &>/dev/null; then
        echo "$user"
    fi
done"""

    ls_path2output = dict()
    for p in possible_ls_dirs:
        search_cmd = search_cmd_template.replace("LS_DIR_PATH", p)
        output = SSHCommunication.run_command_on_machine(machine=cluster, command=search_cmd)
        if output.startswith("ls: cannot access"):
            continue
        else:
            ls_path2output[p] = output.strip()
    return ls_path2output

def get_diskusage_report_from_cluster(*, cluster):
    output = SSHCommunication.run_command_on_machine(machine=cluster, command="bash -lc \"diskusage_report --per_user --all_users --project\" 2>&1", ssh_args=["-tt"])
    return output.strip()
        
def get_default_record_fname():
    date_time_as_str = datetime.now().strftime("%Y-%m-%d-%H:%M")
    return osp.join(osp.abspath(osp.dirname(__file__)), f"diskusage_report_{date_time_as_str}.json")
    
if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("--record", type=str, default=None, nargs="?", const=get_default_record_fname(),
        help="Record to this file")
    P.add_argument("--clusters", "-c", choices=MachineInfo.cluster2node2config.keys(),
        nargs="+", default=["vulcan", "killarney", "nibi", "narval", "fir", "rorqual", "trillium"],
        help="Clusters to source data from")
    # P.add_argument("--diskusage_report", "-d", required=True,
    #     help="--diskusage_report output or output of running LS on relevant dirs")
    # P.add_argument("--ls_path", "-l", default=None,
    #     help="If --diskusage_report is ls output, the path where it's run")
    P.add_argument("--quiet", "-q", action="store_true", default=False,
        help="If set, suppress long output at end messages")
    P.add_argument("--sort", "-s", choices=["user", "size", "quota"], default="user",
        help="How to sort users in output")
    args = P.parse_args()

    user2cluster2project2usage = dict()

    for cluster in tqdm(args.clusters):
        twrite(f"[INFO] Gathering data from cluster: {cluster}")

        if cluster in diskusage_report_clusters:
            diskusage_report = get_diskusage_report_from_cluster(cluster=cluster)
            user2project2usage = per_user_diskusage_report_to_info(diskusage_report)
        else:
            user2project2usage = defaultdict(dict)
            ls_path2output = get_ls_data_from_cluster(cluster=cluster)
            for ls_path,ls_output in ls_path2output.items():
                user2project2usage_ = ls_diskusage_report_to_info(ls_output=ls_output, ls_path=ls_path)
                for user,project2usage in user2project2usage_.items():
                    user2project2usage[user] |= project2usage
            user2project2usage = dict(user2project2usage)

        for user,project2usage in user2project2usage.items():
            if not user in user2cluster2project2usage:
                user2cluster2project2usage[user] = dict()
            if not cluster in user2cluster2project2usage[user]:
                user2cluster2project2usage[user][cluster] = dict()
            for project,usage in project2usage.items():
                user2cluster2project2usage[user][cluster][project] = usage

    if args.sort == "user":
        user2cluster2project2usage = dict(sorted(user2cluster2project2usage.items(), key=lambda x: x[0]))
    elif args.sort == "size":
        user2totals = {user: user_data_to_totals(cluster2project2usage) for user,cluster2project2usage in user2cluster2project2usage.items()}
        user2cluster2project2usage = dict(sorted(user2cluster2project2usage.items(),
            key=lambda x: user2totals[x[0]]["total_file_storage_GiB"], reverse=True))
    elif args.sort == "quota":
        user2totals = {user: user_data_to_totals(cluster2project2usage) for user,cluster2project2usage in user2cluster2project2usage.items()}
        user2cluster2project2usage = dict(sorted(user2cluster2project2usage.items(),
            key=lambda x: user2totals[x[0]]["total_file_count"], reverse=True))

    if not args.quiet:
        for user,cluster2project2usage in user2cluster2project2usage.items():
            s = user_data_to_str(cluster2project2usage)
            print(f"{user}:\t{s}")

    if args.record is None:
        twrite(f"[INFO] --record not given -> not saving data to file")
    else:
        UtilsBase.atomic_save_lite(data=user2cluster2project2usage, fpath=args.record)
        twrite(f"[INFO] Wrote aggregated diskusage report to {args.record}")







    