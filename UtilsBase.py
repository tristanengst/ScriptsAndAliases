"""Utility functions I think should by in Python's standard library."""
import argparse
import copy
from collections import defaultdict
from datetime import datetime
import functools
import json
import math
import os
import os.path as osp
import uuid
import time

try:
    from tqdm import tqdm

except ImportError:
    class tqdm_lite:
        """Stand-in for tqdm if it is not installed."""
        def __init__(self, iterable, **kwargs):
            self.iterable = iterable
        def __iter__(self): return iter(self.iterable)
        def write(s): print(s)
    tqdm = tqdm_lite


####### File information #############################################################
def is_tarfile(f):
    """Returns if [f] is a .tar file"""
    return f.endswith(".tar") or f.endswith(".tar.gz") or f.endswith(".tgz")

######################################################################################
######################################################################################
######################################################################################

####### I/O Functions ################################################################
def write_meta(meta_key=None, **kwargs):
    """Prints [s] in a way that indicates it's meta information. The intended use case
    is essentially as a way to return information to a function called via bash.
    """
    s = json.dumps(kwargs | {"__meta_key__": meta_key})
    print(f"__WRITE_META_SEP____START_META__{json.dumps(kwargs)}__END_META__")
    

def load_meta(s, as_dict=True, meta_key="__first_key__"):
    """Loads meta information from [s] that was printed by write_meta().
    
    There are multiple possible meta informations, each separated by a string
    '__WRITE_META_SEP__'. Within each is a dictionary containing meta information as
    JSON. If [as_dict] is set, return the result as a dictionary mapping meta names to
    their corresponding meta information.
    """
    s = s.strip()
    metas = s.split("__WRITE_META_SEP__")
    metas = metas[1:] if len(metas) > 1 else metas

    results = []
    
    # If we ever wrote a meta-string, then the first element of splitting by
    # '__WRITE_META_SEP___' would not include a meta string.
    for m in metas:
        if "__START_META__" in m and "__END_META__" in m:
            start = m.find("__START_META__") + len("__START_META__")
            end = m.find("__END_META__")
            meta_str = m[start:end]
            try:
                results.append(json.loads(meta_str))
            except json.JSONDecodeError as e:
                twrite(f"[ERROR] load_meta() could not parse meta string:\n{meta_str}")
                raise e
        else:
            twrite(f"[ERROR] load_meta() could not parse metas:\n{m}")
            raise ValueError("[ERROR] load_meta() could not parse metas")
            
    index_keys = ["__meta_key__"]
    if as_dict and meta_key == "__first_key__":
        result = dict()
        for r in results:
            if len(r) == 1:
                result |= r
            elif len(r) == 2:
                first_key = (list(r.keys())[0])
                result[first_key] = r[first_key]
            else:
                result |= {k: v for k,v in r.items() if not k in index_keys}
        return result
    else:
        return [{k: v for k,v in r.items() if not k in index_keys} for r in results]


def twrite(*args, time=True, verbose=1, quiet=False, offset=False, **kwargs):
    """Lite version of twrite(). Doesn't support multiple processes."""
    if quiet or verbose < 1:
        return

    def pretty_time(offset=False):
        offset = " " * 6 if offset else ""
        return f"[{datetime.now().isoformat(sep=' ', timespec='seconds')}]{offset}"
    
    def pretty_time_space(): return pretty_time(offset=True)

    def separated_str(*strs): return " ".join([s for s in strs if not s == ""])

    meta_str = f"{pretty_time(offset=offset)}" if time else (" " * 6 if offset else "")
    kwargs_str = " ".join([f"{k}={v}" for k,v in kwargs.items()])
    args_str = " ".join([str(a) for a in args])
    s = separated_str(meta_str, args_str, kwargs_str)
    tqdm.write(s)

def load_file_lite(fname, json_kwargs=dict(), **kwargs):
    """Loads a file [fname] with [kwargs]. The kind of load function is inferred from
    the file extension. This version does not support .pt files.
    """
    with open(fname, "r") as f:
        if fname.endswith(".pt"):
            raise NotImplementedError("Loading .pt files is not supported")
        elif fname.endswith(".json"):
            return json.load(f, **json_kwargs)
        elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
            return f.read()
        else:
            raise NotImplementedError(f"Unknown file extension for {fname}")

def atomic_save_lite(*, data, fname, **kwargs):
    """Atomically saves [data] to [fname] with [kwargs]. The kind of save function is
    inferred from the file extension. This version does not support .pt files.
    """
    _ = os.makedirs(osp.dirname(fname), exist_ok=True) if osp.dirname(fname) else None
    fname_base, ext = osp.splitext(fname)
    tmp_file = f"__tempfile__{str(uuid.uuid4()).replace('-', '')}_{osp.basename(fname_base)}.tmp"
    tmp_file = osp.join(osp.dirname(fname), tmp_file)

    if fname.endswith(".pt"):
        raise NotImplementedError("Saving .pt files is not supported")
    elif fname.endswith(".json"):
        kwargs["indent"] = 4 if not "indent" in kwargs else kwargs["indent"]
        with open(tmp_file, "w+") as f:
            json.dump(data, f, **kwargs)
    elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
        with open(tmp_file, "w+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")
    os.rename(tmp_file, fname)

def atomic_append_lite(*, data, fname, **kwargs):
    fname = osp.expanduser(fname)
    if fname.endswith(".pt"):
        raise NotImplementedError("Appending to .pt files is not supported")
    elif fname.endswith(".json"):
        return atomic_save_lite(data=load_file_lite(f) | d, fname=f, indent=4, sort_keys=True)
    elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
        with open(fname, "a+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")


def dict_to_json(d, f): return atomic_save_lite(data=d, fname=f, indent=4, sort_keys=True)
def json_to_dict(f): return load_file_lite(f)
def dict_append_json(d, f): atomic_append_lite(data=d, fname=f)

def path_from_home(f):
    """Returns a path to [f] that will work from any home directory."""
    abspath = osp.abspath(osp.expanduser(f))
    home = osp.abspath(osp.expanduser("~"))
    return f"~/{abspath[len(home)+1:]}"



    if abspath.startswith(home):
        return f"~/{abspath[len(home)+1:]}"
    else:
        return abspath




    
######################################################################################
######################################################################################
######################################################################################


###### String Processing Functions ###################################################
def strip_right(s, remove):
    """Returns [s] with the substring [remove] removed from the right side of it if
    [s] ends with [remove], and [s] otherwise. Contrast with rstrip, which removes any
    character in [remove], which is not what I'd expect.
    """
    return s[:-len(remove)] if s.endswith(remove) else s

def strip_left(s, remove):
    """Returns [s] with the substring [remove] removed from the left side of it if
    [s] starts with [remove], and [s] otherwise. Contrast with lstrip, which removes any
    character in [remove], which is not what I'd expect.
    """
    return s[len(remove):] if s.startswith(remove) else s

def remove_nonnumeric(s):
    """Returns [s] with all non-numeric characters removed."""
    return "".join([c for c in s if c.isnumeric()])

def digits_after(s, substr):
    """Returns the longest substring of [s] that directly follows [substr] or [substr]
    and an equals sign that permits numeric interpretation. Returns None if no such
    substring exist or raises an error if [substr] isn't in [s].
    """
    idx = s.find(substr)
    if idx == -1:
        raise ValueError(f"Substring '{substr}' not found in string '{s}'")
    if idx + len(substr) >= len(s):
        return None  # No digits after the substring
    s = s[idx+len(substr)+1:] if s[idx+len(substr)] == "=" else s[idx+len(substr):]

    # Special characters that are used for scientific notation or negation can occur
    # only a finite number of times. If they occur more often, then stop parsing.
    finite_chars2remaining = {".": 1, "e": 1, "-": 2}
    possible_s = ""
    for idx,c in enumerate(s):
        if c in finite_chars2remaining and finite_chars2remaining[c]:
            finite_chars2remaining[c] -= 1
            possible_s += c
        elif c.isnumeric():
            possible_s += c
        else:
            break
    s = possible_s
    if not s:
        return None

    # Return the longest left-aligned substring of [s] that is a number
    numeric_substrings = [s[:idx] for idx in range(1, len(s) + 1)]
    for n in reversed(numeric_substrings):
        try:
            result = float(n)
            return strip_right(n, ".") # Remove the training dot since probably the thing isn't meant to represent a float
        except ValueError:
            continue
    return None  # No numeric substring found

def try_make_number(s):
    """Tries to convert [s] to an int or float, and returns [s] otherwise."""
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return s
    
def list_to_pretty_str(l, one_per_line=False, sep="\t", terminal_size=None):
    """Returns list [l] as a pretty string. The intended usage is to get its elements
    nicely displayed to the terminal.
    
    l               -- List of elements to display
    one_per_line    -- If True, then each element is displayed on its own line
    sep             -- Separator between elements if not one_per_line
    terminal_size   -- If provided, use this as the terminal size instead of querying
    """
    l = [str(ll) for ll in l]
    if one_per_line:
        return "\n\t".join(l)

    max_len = max([len(ll) for ll in l]) if l else 0
    terminal_size = terminal_size if terminal_size else os.get_terminal_size().columns 

    num_cols = max(1, terminal_size // (max_len + 2))
    num_rows = math.ceil(len(l) / num_cols)
    chars_per_col = terminal_size // num_cols
    
    sublists = [l[idx * num_cols:max(len(l), (idx + 1) * num_cols)] for idx in range(num_rows)]
    sublists = [sep.join([s.ljust(chars_per_col) for s in sublist]) for sublist in sublists]
    return "\n".join(sublists)






######################################################################################
######################################################################################
######################################################################################

###### Argparse and Datastructure Functions ##########################################

def truthy_type(s, make_bool=False):
    """Type for argparse value that represents a truth value."""
    s = s.lower()
    if s in ("false", "no", "0"):
        return False if make_bool else 0
    elif s in ("true", "yes", "1"):
        return True if make_bool else 1
    else:
        raise argparse.ArgumentTypeError(f"Invalid truthy value: {s}. Must be one of 'true', 'false', 'yes', 'no', '1', '0'.")

def file_exists_type(s, allow_none=False):
    """Type for argparse value that represents a file that must exist."""
    if (allow_none and s is None) or osp.exists(s):
        return s
    else:
        raise argparse.ArgumentTypeError(f"File does not exist: {s}. Please provide a valid file path or 'None'.")

def updated_namespace(extant, *updated, **kwargs):
    """Returns a new argparse Namespace that updates [extant] with [updated] and [kwargs]."""
    assert len(updated) <= 1, "Only one updated argument is allowed"
    extant = vars(extant) if isinstance(extant, argparse.Namespace) else extant
    updated = dict() if len(updated) == 0 else updated[0]
    updated = vars(updated) if isinstance(updated, argparse.Namespace) else updated
    return argparse.Namespace(**extant | updated | kwargs)

def dict_to_namespace(d):
    """Returns possibly-nested dictionary [d] as an argparse Namespace."""
    d = vars(d) if isinstance(d, argparse.Namespace) else d
    if isinstance(d, dict):
        return argparse.Namespace(**{k: dict_to_namespace(v) for k,v in d.items()})
    elif isinstance(d, list | tuple): # Obviously this shouldn't be the outer call!
        return d.__class__([dict_to_namespace(v) for v in d])
    else:
        return d

def namespace_to_dict(n):
    """Returns the namespace [n] as a dictionary."""
    n = vars(n) if isinstance(n, argparse.Namespace) else n
    if isinstance(n, dict):
        return {k: namespace_to_dict(v) for k,v in n.items()}
    elif isinstance(n, list | tuple | set):
        return n.__class__([namespace_to_dict(v) for v in n])
    elif isinstance(n, str | int | float | bool | None):
        return n
    else:
        raise ValueError(f"Could not convert {n} to dictionary: {type(n)}")

def flatten(xs):
    """Returns collection [xs] after recursively flattening into a list."""
    type_map = {type({}.items()): list, type({}.values()): list, type({}.keys()): set}
    xs = type_map[type(xs)](xs) if type(xs) in type_map else xs

    if isinstance(xs, list | set | tuple):
        result = []
        for x in xs:
            result += flatten(x) if isinstance(x, list | set | tuple) else [x]
        return xs.__class__(result)
    else:
        return xs

def reverse_dict(d, use_defaultdict=False):
    """Returns dictionary [d] with the keys and values swapped. Set [use_defaultdict]
    to handle when multiple keys could the same value.
    """
    if use_defaultdict:
        result = defaultdict(list)
        for k,v in d.items():
            rd[v].append(k)
        return result
    else:
        return {v: k for k,v in d.items()}

def unparse_args(args, return_as="str"):
    """Returns a string that would recreate the argparse Namespace [args]."""
    args = vars(args) if isinstance(args, argparse.Namespace) else args
    
    for k,v in args.items():
        
        # It's inherently dangerous to represent booleans this way, as it's not clear
        # what not having the flag would mean. Still, this is the most likely option
        if isinstance(v, bool):
            result += f" --{k}" if v else ""
                
        elif isinstance(v, list | tuple):
            v = " ".join([f"\'{vv}\'" if isinstance(vv, str) else str(vv) for vv in v])
            arg_strs.append(f"--{k} {v}")
        else:
            arg_strs.append(f"--{k} {v}")
    return " ".join(arg_strs) if return_as == "str" else arg_strs


######################################################################################
######################################################################################
######################################################################################


###### Time Functions ################################################################
def seconds_since_time(start_time):
    if isinstance(start_time, datetime):
        return (datetime.now() - start_time).total_seconds()
    elif isinstance(start_time, str):
        return (datetime.now() - time_stamp_to_datetime(start_time)).total_seconds()
    else:
        return time.time() - start_time  
def hours_since_time(start_time): return seconds_since_time(start_time) / 3600
def minutes_since_time(start_time): return seconds_since_time(start_time) / 60
def seconds_to_minutes(seconds): return seconds / 60
def seconds_to_hours(seconds): return seconds / 3600

def time_stamp_to_datetime(time_stamp):
    """Converts a time stamp to a datetime object."""
    if isinstance(time_stamp, datetime):
        return time_stamp
    
    time_stamp = time_stamp.strip()
    # Common custom time stamp format to make life easier
    if time_stamp.find("-") in [1,2]:
        dt = datetime.strptime(time_stamp, "%m-%d-%H:%M")
        dt = dt.replace(year=datetime.now().year)
        return dt
    elif "T" in time_stamp:
        return datetime.strptime(time_stamp, "%Y-%m-%dT%H:%M:%S")
    elif "-" in time_stamp:
        return datetime.strptime(time_stamp, "%Y-%m-%d-%H:%M:%S")
    else:
        raise ValueError(f"Could not parse time stamp: {time_stamp}")

def time_to_seconds(time_str):
    """Returns [time_str] as a number of seconds. Tries to fit as many possible ways
    [time_str] could be interpreted as a duration; it need not actually be a string.

    This sort of time string would indicate a duration.
    """
    if isinstance(time_str, int | float):
        return time_str

    time_str = time_str.strip()
    if time_str.lower().endswith("s"):
        return float(time_str[:-1])
    elif time_str.lower().endswith("m"):
        return float(time_str[:-1]) * 60
    elif time_str.lower().endswith("h"):
        return float(time_str[:-1]) * 3600
    elif time_str.lower().endswith("d"):
        return float(time_str[:-1]) * 24 * 3600
    elif "-" in time_str:
        days, time_str = time_str.split("-")
        
        # If there is only a single colon in [time_str] now, then assume that the
        # seconds are not included.
        time_str = f"{time_str}:00" if time_str.count(":") == 1 else time_str
        
        return int(days) * 24 * 3600 + time_to_seconds(time_str)
    # Usually output by SLURM. Assumes that seconds are present!
    elif ":" in time_str:
        times = time_str.split(":")
        return sum([int(t) * (60 ** idx) for idx,t in enumerate(reversed(times))])
    else:
        time_suffix2seconds = dict(H=3600, M=60, S=1, D=24*3600)
        s, cur_num = 0, ""
        for c in time_str:
            if c.isnumeric():
                cur_num += c
            elif c in time_suffix2seconds and cur_num:
                s += int(cur_num) * time_suffix2seconds[c.upper()]
                cur_num = ""
            else:
                raise ValueError(f"Invalid character in time string: {c} in {time_str}")
        return s
        
def time_to_hours(t): return time_to_seconds(t) / 3600
def time_to_minutes(t): return time_to_seconds(t) / 60
def time_to_str(t):
    """Returns time string [time_str] in our default way, ie. without days."""
    s = time_to_seconds(t)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{int(h)}:{int(m):02}:{int(s):02}"

def time_to_pretty_str(t):
    """Returns XXHYYM for XX hours and YY minutes. Days are collapsed to hours."""
    s = time_to_seconds(t)
    h = s // 3600
    m = (s % 3600) // 60
    h = str(h).zfill(max(2, len(str(h))+1))
    return f"{h}H{m:02d}M"

######################################################################################
######################################################################################
######################################################################################


###### Persisted State ###############################################################
# For when environment variables have semantics not suitable for the task at hand. In
# this case, we will generally assume that either there is one process running per
# machine or that a machine isolates processes that would use this via $SLURM_TMPDIR.
def get_persisted_state_file():
    """Returns the path to the persisted state file."""
    return f"{os.environ['SLURM_TMPDIR'] if 'SLURM_JOB_ID' in os.environ else '.'}/persisted_state.json"

def persisted_state_get_all():
    """Returns all persisted state as a dictionary."""
    f = get_persisted_state_file()
    return json_to_dict(f) if osp.exists(f) else dict()

def persisted_state_get(k, default=None):
    """Returns the value of [k] in persisted state or [default] if it isn't found."""
    persisted_state = persisted_state_get_all()
    return persisted_state.get(k, (default() if callable(default) else default))

def persisted_state_contains(k):
    """Returns if [k] is in the persisted state."""
    return k in persisted_state_get_all()

def persisted_state_update(**kwargs):
    """Sets key [k] to value [v] in persisted state."""
    _ = dict_append_json(kwargs, get_persisted_state_file())

def persistent_state_del(k):
    """Deletes key [k] from the persisted state."""
    persisted_state = persisted_state_get_all()
    persisted_state = {k1: v for k1,v in persisted_state.items() if not k1 == k}
    return dict_to_json(persisted_state, get_persisted_state_file())

def persisted_state_clear():
    """Removes the persisted state file."""
    _ = dict_to_json(dict(), get_persisted_state_file())

######################################################################################
######################################################################################
######################################################################################





