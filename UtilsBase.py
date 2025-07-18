"""Utility functions I think should by in Python's standard library."""
import argparse
from datetime import datetime
import functools
import json
import os
import os.path as osp
import uuid

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
        elif fname.endswith(".txt") or fname.endswith(".sh"):
            return f.read()
        else:
            raise NotImplementedError(f"Unknown file extension for {fname}")

def atomic_save_lite(*, data, fname, **kwargs):
    """Atomically saves [data] to [fname] with [kwargs]. The kind of save function is
    inferred from the file extension. This version does not support .pt files.
    """
    _ = os.makedirs(osp.dirname(fname), exist_ok=True)
    fname_base, ext = osp.splitext(fname)
    tmp_file = f"__tempfile__{str(uuid.uuid4()).replace('-', '')}_{osp.basename(fname_base)}.tmp"
    tmp_file = osp.join(osp.dirname(fname), tmp_file)

    if fname.endswith(".pt"):
        raise NotImplementedError("Saving .pt files is not supported")
    elif fname.endswith(".json"):
        kwargs["indent"] = 4 if not "indent" in kwargs else kwargs["indent"]
        with open(tmp_file, "w+") as f:
            json.dump(data, f, **kwargs)
    elif fname.endswith(".txt") or fname.endswith(".sh"):
        with open(tmp_file, "w+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")
    os.rename(tmp_file, fname)

def dict_to_json(d, f): return atomic_save_lite(data=d, fname=f, indent=4, sort_keys=True)
def json_to_dict(f): return load_file_lite(f)
def dict_append_json(d, f): return atomic_save_lite(data=load_file_lite(f) | d, fname=f, indent=4, sort_keys=True)

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
    if isinstance(xs, list | set | tuple):
        result = []
        for x in xs:
            result += [flatten(x)]
        return xs.__class__(result)
    else:
        return xs

######################################################################################
######################################################################################
######################################################################################


###### Time Functions ################################################################
def time_since_time(start_time): return time.time() - start_time    
def hours_since_time(start_time): return time_since_time(start_time) / 3600
def minutes_since_time(start_time): return time_since_time(start_time) / 60
def seconds_to_minutes(seconds): return seconds / 60
def seconds_to_hours(seconds): return seconds / 3600


def time_str_to_time(time_str):
    """Returns the number of seconds in a time string, formatted in the various ways
    SLURM tends to do it, eg. DD-HH:MM:SS, HH:MM:SS. It is assumed that seconds are
    included.
    """
    if "-" in time_str:
        days, time_str = time_str.split("-")
        return int(days) * 24 * 3600 + time_str_to_time(time_str)
    else:
        times = time_str.split(":")
        return sum([int(t) * (60 ** idx) for idx,t in enumerate(reversed(times))])
def time_str_to_hours(time_str): return time_str_to_time(time_str) / 3600
def time_str_to_minutes(time_str): return time_str_to_time(time_str) / 60
def time_str_to_str(time_str):
    """Returns time string [time_str] in our default way, ie. without days."""
    s = time_str_to_time(time_str)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02}:{s:02}"

def pretty_time_str(time_str):
    """Returns XXHYYM for XX hours and YY minutes. Days are collapsed to hours."""
    s = time_str_to_seconds(time_str)
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





