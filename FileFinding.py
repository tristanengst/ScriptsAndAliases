import argparse
import copy
import glob
import json
import inspect
import os
import os.path as osp
import sys

import Utils
from Utils import get_slurm_status
import UtilsBase
from UtilsBase import twrite
import UserConfig

exp_search_dirs = UserConfig.checkpoints_search_dirs
slurm_script_search_dirs = UserConfig.slurm_script_search_dirs
job_result_search_dirs = UserConfig.job_result_search_dirs
job_error_search_dirs = UserConfig.job_error_search_dirs

# Custom code for me; you might delete this
if Utils.get_cluster_type() == "cedar":
    exp_search_dirs += [osp.expanduser("~/Development/IMLE-SSL-Cedar/pretrain_results"),
        osp.expanduser("~/Development/IMLE-SSL-Cedar/finetune_results")]
    slurm_script_search_dirs += [osp.expanduser("~/Development/IMLE-SSL-Cedar/slurm")]

file_search_dirs = slurm_script_search_dirs + job_result_search_dirs + job_error_search_dirs + exp_search_dirs

class MultipleMatchesError(Exception):
    def __init__(self, message, *, matches):
        super().__init__(message)
        self.matches = matches

resolve_choices = ["pos", "user", "half", "half_then_user", "latest", "all"]
def maybe_resolve_multiple_matches(*, matches, s, resolve="pos", verbose=False, if_not_found="error"):
    """Returns a single match from [matches] according to the strategy [resolve] if
    possible.

    Args:
    matches -- list of matches to resolve among
    s       -- string that was matched on, used for some resolution strategies
    resolve -- strategy to use to resolve among multiple matches. One of:
                pos     --  choose where the match ends nearest to the end
                user    -- the user is prompted to choose
                half    -- choose the one where the match is in the second half of the
                        basename; if multiple, an error is raised
                half_then_user  -- the one where the match is in the second half of
                                    the basename is chosen; if multiple,
                                    the user is prompted to choose
                latest  -- the one with the most recent modification time is chosen
                all     -- all matches are returned as a list
    """
    if len(matches) == 0 and if_not_found == "error":
        raise FileNotFoundError(f"[ERROR] maybe_resolve_multiple_matches(): no matches for {s}")
    elif len(matches) == 0 and if_not_found == "none":
        return None
    elif len(matches) == 0:
        return if_not_found() if callable(if_not_found) else if_not_found
    elif len(matches) == 1:
        return matches[0]

    # Resolve by finding the match where the match ends nearest to the end of the
    # string. This is a simple heuristic that's highly exploitable; put the UID of the
    # current thing towards the end.
    elif resolve == "pos":
        matches = sorted(matches, key=lambda m: len(s) - m.rfind(s))
        return matches[0]

    # Resolve by asking the user
    elif resolve == "user":
        print(f"[INFO] Found multiple matches for {s}:")
        return UtilsBase.query_among_list(prompt=f"Multiple matches found for {s}, please choose:", options=matches)

    # Valid matches are those where the match ends in the second half of the basename.
    # This tends to be the most unique part of the name.
    elif resolve == "half":
        matches2match_idxs = {m: (osp.basename(m).rfind(s), osp.basename(m).rfind(s) + len(s)) for m in matches if s in m}
        new_matches = [m for m, (start_idx, end_idx) in matches2match_idxs.items() if start_idx >= len(osp.basename(m)) // 2]
        if len(new_matches) == 0:
            raise ValueError(f"[ERROR] zero matches for {s} with resolve='{resolve}', but there were multiple original matches:\n\t{UtilsBase.list_to_pretty_str(matches)}")
        elif len(new_matches) > 1:
            raise MultipleMatchesError(f"[ERROR] multiple matches for {s} with resolve='{resolve}':\n\t{UtilsBase.list_to_pretty_str(new_matches)}")
        else:
            return new_matches[0]

    # Try first using resolve='half', and if this fails, fall back to the user.
    elif resolve == "half_then_user":
        try:
            half_matches = maybe_resolve_multiple_matches(matches=matches, s=s, resolve="half", verbose=verbose)
            half_match = half_matches[0] if isinstance(half_matches, list) else half_matches
            twrite(f"[INFO] Resolved with resolve='half' for {s} -> {half_match}", verbose=verbose)
            return half_match
        except Exception as e:
            return maybe_resolve_multiple_matches(matches=matches, s=s, resolve="user", verbose=verbose)
    
    # Return the most-recently modified match. Need to check all files in the folder,
    # but assume we don't need to do so recursively.
    elif resolve == "latest":
        matches2mtime = {m: max([osp.getmtime(osp.join(m, f)) for f in os.listdir(m)]) for m in matches}
        matches = sorted(matches, key=lambda m: matches2mtime[m])
        return matches[-1]
    elif resolve == "all":
        return matches
    else:
        raise ValueError(f"[ERROR] Unknown resolve method {resolve}")

def file_substr_to_glob(f, *, search_dirs=exp_search_dirs + file_search_dirs, first_match=False, resolve="half_then_user", verbose=False):
    """Returns the list of files that match the substring [f], but in a way where
    existing globs would be treated nicely by bash.

    Args:
    f           -- substring to match. Can contain * at beginning and/or end
    search_dirs -- directories to search in if [f] is not an absolute path
    first_match -- return on the first match found
    resolve     -- how to resolve multiple matches. One of:
                    pos -- the one where the match ends nearest to the end of the string is chosen
                    user -- the user is prompted to choose
                    half_then_user -- the one where the match is in the second half of the basename is chosen; if multiple, the user is prompted to choose
                    latest -- the one with the most recent modification time is chosen
                    all -- all matches are returned as a list
    """
    if osp.exists(f):
        return [f]
    else:
        fw = "*" + UtilsBase.strip_left(UtilsBase.strip_right(f, "*"), "*") + "*"

        f = f.strip()
        # Semantics for globbing: if [f] starts XOR ends with a glob, then no more
        # globs are added. Otherwise, a glob is added to both sides. A glob in the
        # middle of [f] has no effect on those added to the ends.
        fw = f if f.startswith("*") or f.endswith("*") else f"*{f}*"
        
        globs = []
        all_matched_files = set()
        for d in search_dirs:
            if not osp.exists(d):
                continue

            test_file = osp.join(d, fw)
            matched_files = glob.glob(test_file)
            if matched_files:
                globs.append(test_file)
                all_matched_files |= set(matched_files)
                if first_match:
                    break
        
        if len(globs) == 0:
            raise ValueError(f"No files found matching substring {f} in search_dirs={search_dirs}")
        elif not "*" in f and len(all_matched_files) > 1 and resolve is None:
            globs = "\t" + "\n\t".join(sorted(globs))
            all_matched_files = "\t" + "\n\t".join(sorted(all_matched_files))
            _ = twrite(f"[ERROR] file={f} matches multiple possibilities but does not contain *:\nglobs=\n{globs}\nall_matched_files=\n{all_matched_files}")
        elif not "*" in f and len(all_matched_files) > 1 and resolve in ["pos", "user", "half", "half_then_user", "latest"]:
            globs = "\t" + "\n\t".join(sorted(globs))
            all_matched_files_str = "\t" + "\n\t".join(sorted(all_matched_files))
            _ = twrite(f"[WARNING] file={f} matches multiple possibilities but does not contain *:\nglobs=\n{globs}\nall_matched_files=\n{all_matched_files_str}\n-------------- [INFO] -> resolve using resolve='{resolve}'... --------------")
            result = maybe_resolve_multiple_matches(matches=list(all_matched_files), s=f, resolve=resolve, verbose=False)
            twrite(f"[INFO] Got {result}\n")
            return result
        else:
            return globs

def file_to_nonambiguous_path(f):
    """Returns the non-ambiguous path to file [f] by prefering symlinks from the home
    directory.
    """
    # Where the file actually is, and the absolute path to it respecting the symlinks
    # it was passed with. 
    real_f = osp.realpath(osp.expanduser(f))
    abs_f = osp.abspath(osp.expanduser(f))

    abs_prefix2non_ambiguous_prefix = {osp.abspath(osp.expanduser(p)): p for p in ["~/scratch", "~"]}
    abs_prefix2non_ambiguous_prefix |= {"/NAS": "~/scratch"}
    
    non_ambiguous_f = [osp.join(v, abs_f[len(k)+1:]) for k,v in abs_prefix2non_ambiguous_prefix.items() if abs_f.startswith(k)]
    non_ambiguous_f = list(set(non_ambiguous_f))
    if len(non_ambiguous_f) == 0:
        twrite(f"[WARNING] Could not find non-ambiguous path for {f}. real_f={real_f} abs_f={abs_f} abs_prefix2non_ambiguous_prefix={abs_prefix2non_ambiguous_prefix} -> return original path", verbose=True)
        return f
    elif len(non_ambiguous_f) > 1:
        _ = twrite(f"[ERROR] file={f} has multiple non-ambiguous paths: {sorted(non_ambiguous_f)}")
        return f
    else:
        return non_ambiguous_f[0]

def exp_folder_to_uid(exp_folder, verbose=False):
    """Tries to find the UID for an experiment folder."""
    if not osp.exists(exp_folder) or not osp.isdir(exp_folder):
        _ = twrite(f"[WARNING] exp_folder={exp_folder} does not exist or is not a folder", verbose=verbose)
        return None
    
    if osp.exists(osp.join(exp_folder, "config.json")):
        content = UtilsBase.load_file_lite(osp.join(exp_folder, "config.json"))
        if "uid" in content:
            return content["uid"]
    
    _ = twrite(f"[WARNING] Could not find UID for exp_folder={exp_folder}", verbose=verbose)
    return None


def str_to_result_file(s, search_dirs=job_result_search_dirs, verbose=False, resolve="pos", if_not_found="error"):
    """Returns the result file corresponding to string [s].

    Args:
    s           -- string to match. Does not need pre-globbing
    search_dirs -- directories to search in if [s] is not an absolute path
    verbose     -- whether to print verbose output
    resolve     -- method for resolving multiple matches
    if_not_found    -- behavior when no matches are found ('error', 'none', or a default value)
    """
    return str_to_file(s, search_dirs=search_dirs, file_type="result", verbose=verbose, resolve=resolve, if_not_found=if_not_found)

def str_to_slurm_script(s, search_dirs=slurm_script_search_dirs, verbose=False, resolve="pos", if_not_found="error"):
    """Returns the SLURM script corresponding to string [s].

    Args:
    s           -- string to match. Does not need pre-globbing
    search_dirs -- directories to search in if [s] is not an absolute path
    verbose     -- whether to print verbose output
    resolve     -- method for resolving multiple matches
    if_not_found    -- behavior when no matches are found ('error', 'none', or a default value)
    """
    return str_to_file(s, search_dirs=search_dirs, file_type="slurm", verbose=verbose, resolve=resolve, if_not_found=if_not_found)

def str_to_exp_folder(s, search_dirs=exp_search_dirs, resolve="pos", verbose=False, if_not_found="error"):
    """Returns the experiment folder corresponding to string [s].

    Args:
    s               -- string to match. Does not need pre-globbing
    search_dirs     -- directories to search in if [s] is not an absolute path
    resolve         -- method for resolving multiple matches
    verbose         -- whether to print verbose output
    matches         -- list of existing matches to consider
    if_not_found    -- behavior when no matches are found ('error', 'none', or a default value)
    """
    return str_to_file(s, search_dirs=search_dirs, file_type="exp", verbose=verbose, resolve=resolve, if_not_found=if_not_found)

def str_to_file(s, search_dirs=[], file_type="slurm", verbose=False, matches=None, resolve="pos", if_not_found="error"):
    """Returns the file(s) corresponding to string [s].
    
    Args:
    s           -- string to match. Does not need pre-globbing
    search_dirs -- directories to search in if [s] does not exist directly
    """
    s = s.strip()
    if osp.exists(s) and (osp.isfile(s) or (file_type == "exp" and osp.isdir(s))):
        return s
    
    matches = matches if matches else str_to_all_files(s, search_dirs=search_dirs, verbose=verbose, file_type=file_type)
    return maybe_resolve_multiple_matches(matches=matches, s=s, resolve=resolve, verbose=verbose, if_not_found=if_not_found)

def str_to_all_files(s, search_dirs=[], file_type="result", verbose=False):
    """Returns all files that match the string [s]."""
    s = s.strip()
    # Semantics for globbing: if the string starts XOR ends with a glob, then not more
    # globs are added. Otherwise, a glob is added to both sides. A glob in the middle
    # of [s] has no effect on those added to the ends.
    s_glob = s if s.startswith("*") or s.endswith("*") else f"*{s}*"
    
    if file_type == "result":
        search_dirs = job_result_search_dirs + search_dirs
        result_is_file = True
    elif file_type == "slurm":
        search_dirs = slurm_script_search_dirs + search_dirs
        result_is_file = True
    elif file_type == "error":
        search_dirs = job_error_search_dirs + search_dirs
        result_is_file = True
    elif file_type == "exp":
        search_dirs = exp_search_dirs + search_dirs
        result_is_file = False
    else:
        search_dirs = search_dirs
        result_is_file = False

    search_dirs = set([d for d in search_dirs if osp.exists(d) and osp.isdir(d)])
    return [m for d in search_dirs for m in glob.glob(osp.join(d, s_glob)) if (osp.isfile(m) if result_is_file else True)]

def get_args(args=None):
    P = argparse.ArgumentParser()
    P.add_argument("--fn", choices=["str_to_all_files",
        "str_to_file",
        "str_to_exp_folder",
        "str_to_slurm_script",
        "str_to_result_file",],
        required=True, help="Function to run")
    
    P.add_argument("-s", "--value", required=True, help="Value to search for")
    P.add_argument("--key", "Key to search under", default="name", help="Key to search under")
    P.add_argument("--resolve", choices=["pos", "user", "half", "half_then_user", "latest", "all"],
        default="pos", help="How to resolve multiple matches")
    P.add_argument("--verbose", action="store_true", help="Whether to print verbose messages")
    
    P.add_argument(f"--search_dirs", nargs="+", default=exp_search_dirs,
        help="Directories to search for files in if the value is not an absolute path")
    P.add_argument(f"--file_search_dirs", nargs="+", default=file_search_dirs,
        help="Directories to search for files in if the value is not an absolute path")
    P.add_argument(f"--exp_search_dirs", nargs="+", default=exp_search_dirs,
        help="Directories to search for experiment folders in if the value is not an absolute path")
    
    def parse_json_kwargs(s):
        try:
            return json.loads(s)
        except Exception as e:
            raise ValueError(f"[ERROR] Could not parse JSON string {s}: {e}")
    P.add_argument("--json_kwargs", default=dict(), type=parse_json_kwargs,
        help="Additional keyword arguments to pass to the function, in JSON format")
    
    P.add_argument("--output_as_meta", default=None,
        help="If set, key under the meta string to output the result under")
    args = P.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.fn == "str_to_file":
        result = str_to_file(args.value, search_dirs=args.file_search_dirs, verbose=args.verbose, resolve=args.resolve, **args.json_kwargs)
    elif args.fn == "str_to_all_files":
        result = str_to_all_files(args.value, search_dirs=args.file_search_dirs, verbose=args.verbose, resolve=args.resolve, **args.json_kwargs)
    elif args.fn == "str_to_exp_folder":
        result = str_to_exp_folder(args.value, search_dirs=args.exp_search_dirs, verbose=args.verbose, resolve=args.resolve, **args.json_kwargs)
    elif args.fn == "str_to_slurm_script":
        result = str_to_slurm_script(args.value, search_dirs=args.slurm_script_search_dirs, verbose=args.verbose, resolve=args.resolve, **args.json_kwargs)
    elif args.fn == "str_to_result_file":
        result = str_to_result_file(args.value, search_dirs=args.job_result_search_dirs, verbose=args.verbose, resolve=args.resolve, **args.json_kwargs)
    else:
        raise ValueError(f"[ERROR] Unknown function {args.fn}")

    _ = UtilsBase.write_meta({args.output_as_meta: result}) if args.output_as_meta else print(result)

    





