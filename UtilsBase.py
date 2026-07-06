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
from threading import Thread

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

@functools.cache
def torch_available():
    """Returns whether PyTorch is available."""
    try:
        import torch
        return True
    except ImportError:
        return False

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

def load_file_lite(fpath, json_kwargs=dict(), weights_only=False, map_location="cpu", **kwargs):
    """Loads a file [fpath] with [kwargs]. The kind of load function is inferred from
    the file extension.
    """
    txt_extensions = [".txt", ".sh", ".py", ".log", ".enc"]
    if (fpath.endswith(".pt") or fpath.endswith(".pth")) and torch_available():
        return torch.load(fpath, weights_only=weights_only, map_location=map_location, **kwargs)
    elif (fpath.endswith(".pt") or fpath.endswith(".pth")) and not torch_available():
        raise ImportError("PyTorch is not available, so .pt files cannot be loaded")
    else:
        with open(fpath, "r") as f:
            if fpath.endswith(".pt"):
                raise NotImplementedError("Loading .pt files is not supported")
            elif fpath.endswith(".json"):
                return json.load(f, **json_kwargs)
            elif any([fpath.endswith(ext) for ext in txt_extensions]):
                return f.read()
            else:
                twrite(f"[WARNING] load_file_lite(): unknown extension for fpath={fpath} -> assume string-like")
                return f.read()

def atomic_save_lite(*, data, fpath, **kwargs):
    """Atomically saves [data] to [fpath] with [kwargs]. The kind of save function is
    inferred from the file extension.
    """
    txt_extensions = [".txt", ".sh", ".py", ".log", ".enc"]
    _ = os.makedirs(osp.dirname(fpath), exist_ok=True) if osp.dirname(fpath) else None
    fpath_base, ext = osp.splitext(fpath)
    tmp_file = f"__tempfile__{str(uuid.uuid4()).replace('-', '')}_{osp.basename(fpath_base)}.tmp"
    tmp_file = osp.join(osp.dirname(fpath), tmp_file)

    if fpath.endswith(".pt") and torch_available():
        torch.save(data, tmp_file, **kwargs)
    elif fpath.endswith(".pt") and not torch_available():
        raise ImportError("PyTorch is not available, so .pt files cannot be saved")
    elif fpath.endswith(".json"):
        kwargs["indent"] = 4 if not "indent" in kwargs else kwargs["indent"]
        with open(tmp_file, "w+") as f:
            json.dump(data, f, **kwargs)
    elif any([fpath.endswith(ext) for ext in txt_extensions]):
        with open(tmp_file, "w+") as f:
            f.write(data)
    else:
        twrite(f"[WARNING] atomic_save_lite(): unknown extension for fpath={fpath} -> assume string-like")
        with open(tmp_file, "w+") as f:
            f.write(data)
    os.rename(tmp_file, fpath)

def atomic_append_lite(*, data, fname, weights_only=False, map_location="cpu", **kwargs):
    fname = osp.expanduser(fname)
    if fname.endswith(".pt") or fname.endswith(".pth"):
        return atomic_save_lite(data=load_file_lite(fname, weights_only=weights_only, map_location=map_location) | data, fpath=fname, **kwargs)
    elif fname.endswith(".json"):
        return atomic_save_lite(data=load_file_lite(fname) | data, fpath=fname, indent=4, sort_keys=True)
    elif fname.endswith(".txt") or fname.endswith(".sh") or fname.endswith(".py"):
        with open(fname, "a+") as f:
            f.write(data)
    else:
        raise NotImplementedError(f"Unknown file extension for {fname}")


def dict_to_json(d, f): return atomic_save_lite(data=d, fpath=f, indent=4, sort_keys=True)
def json_to_dict(f): return load_file_lite(f)
def dict_append_json(d, f): atomic_append_lite(data=d, fname=f)

def path_from_home(f):
    """Returns a path to [f] that will work from any home directory."""
    abspath = osp.abspath(osp.expanduser(f))
    home = osp.abspath(osp.expanduser("~"))
    return f"~/{abspath[len(home)+1:]}"
    
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
    """Returns [s] with all non-numeric characters removed or [s] if it is numeric."""
    return s if isinstance(s, float | int) else "".join([c for c in s if c.isnumeric()])

def str_to_nonnumeric_prefix(s):
    """Returns the longest prefix of [s] that is not numeric."""
    for idx,c in enumerate(s):
        if c.isnumeric():
            return s[:idx]
    return s

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

def try_make_jsonable(x):
    """Returns [x] converted to a JSONable thing as needed."""
    if isinstance(x, dict):
        return {try_make_jsonable(k): try_make_jsonable(v) for k,v in x.items()}
    elif isinstance(x, argparse.Namespace):
        return try_make_jsonable(vars(x))
    elif isinstance(x, list):
        return [try_make_jsonable(xi) for xi in x]
    elif isinstance(x, tuple):
        return tuple([try_make_jsonable(xi) for xi in x])
    elif isinstance(x, set):
        return list([try_make_jsonable(xi) for xi in x])
    elif isinstance(x, (int, float, str, bool)) or x is None:
        return x
    else:
        return str(x)

def unit_conversion(x, desc=None, source=None, target=None):
    """Returns [x] converted from [source] units to [target] units."""
    # Multiplier types that are unambiguous. 'm' and 'M' are ambiguous.
    number_multipliers = {"K", "G", "T", "KB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"}
    time_multipliers = {"seconds", "minutes", "hours", "days", "s", "h", "d", "S", "H", "D"}
    inferred_multipliers = {"M"}
    all_multipliers = number_multipliers | time_multipliers | inferred_multipliers
    all_multipliers = sorted(all_multipliers, key=lambda am: len(am), reverse=True) # longest-first, so first match is longest and thus most meaningful
    
    unit2multiplier_wrt_base = dict(
        K=1e3, G=1e9, T=1e12,
        KB=1e3, MB=1e6, GB=1e9, TB=1e12,
        KiB=(2**10), MiB=(2**20), GiB=(2**30), TiB=(2**40),
        seconds=1, minutes=60, hours=3600, days=3600*24,
        s=1, h=3600, d=3600*24,
        S=1, H=3600, D=3600*24,
        none=1)

    # Found source
    if isinstance(x, str) and source is None:
        for am in all_multipliers:
            if x.endswith(am):
                x, source = int(strip_right(x, am)), am
                break
    # In this case, interpret the source as an override
    elif isinstance(x, str) and source is not None:
        x = int(remove_nonnumeric(x))


    # Parse [source]
    if not desc is None and source is None and target is None:
        source, target = [d.strip() for d in desc.split("->")] # Einops style is nice!
    elif source is None and not target is None:
        source = "none"
    elif target is None and not source is None:
        target = "none"
    assert not target is None
    assert not source is None

    

    # twrite("AAA", x=x, source=source)
        

    if target in number_multipliers and source == "M":
        source = "MB"
    elif target in time_multipliers and source == "M":
        source = "minutes"
    
    if source in number_multipliers and target == "M":
        target = "MB"
    elif source in time_multipliers and target == "M":
        target = "minutes"

    # twrite(x=x, source=source, target=target)

    
        
    # def infer_m_meaning(*, source, target):
    #     if ((source == "M" and target in number_multipliers)
    #         or (target == "M" and source in number_multipliers)):
    #         return 1e6
    #     elif ((source == "M" and target in time_multipliers)
    #         or (target == "M" and source in time_multipliers)):
    #         return 60
    #     else:
    #         raise ValueError(f"Ambiguous multiplier 'M' with source={source} and target={target}")

    # if source is None and target is None and not desc is None:
    #     source, target = [d.strip() for d in desc.split("->")] # Einops style is nice!
    # elif source is None and not target is None and desc is None:
    #     source = "none"

    # if source in inferred_multipliers and target in unit2multiplier_wrt_base:
    #     raise ValueError(f"Cannot infer meaning of multiplier 'M' when target unit is provided: source={source}, target={target}")

    # source_multiplier = infer_meaning(source=source, target=target) if source in inferred_multipliers else unit2multiplier_wrt_base.get(source, None)
    # target_multiplier = infer_meaning(source=target, target=source) if target in inferred_multipliers else unit2multiplier_wrt_base.get(target, None)

    source_multiplier = unit2multiplier_wrt_base.get(source, 1)
    target_multiplier = unit2multiplier_wrt_base.get(target, 1)
    if source in unit2multiplier_wrt_base and target in unit2multiplier_wrt_base:
        # twrite(f"[DEBUG] unit_conversion(): x={x}, source={source}, target={target}, source_multiplier={source_multiplier}, target_multiplier={target_multiplier}")
        return x * source_multiplier / target_multiplier
    else:
        raise ValueError(f"Unknown source or target unit for conversion: {source} -> {target}")
    
def try_make_number(s, suffixes=False):
    """Tries to convert [s] to an int or float, and returns [s] otherwise."""
    def try_make_number_(s):
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    if not suffixes:
        return try_make_number_(s)
    elif isinstance(s, str) and not suffixes and len(s) > 0:
        suffix2mul = dict(K=1e3, M=1e6, G=1e9, T=1e12,
            KB=1e3, MB=1e6, GB=1e9, TB=1e12,
            KiB=1024, MiB=1024**2, GiB=1024**3, TiB=1024**4)
        
        s_tail = s[-1 * min(len(s), 3):]
        s_head = s[:-len(s_tail)] if len(s) > len(s_tail) else ""
        s_head_maybe_numeric = try_make_number_(s_head)

        if s_tail in suffix2mul and isinstance(s_head_maybe_numeric, (int, float)):
            return s_head_maybe_numeric * suffix2mul[s_tail]
        else:
            return s_head_maybe_numeric
    else:
        raise NotImplementedError(f"try_make_number() s={s} with type(s)={type(s)} and suffixes={suffixes} is not supported")

def is_numeric(s): return not isinstance(try_make_number(s), str)

def last_numeric_substring(s):
    """Returns the longest tail of [s] that is numeric."""
    for idx in range(len(s)):
        if is_numeric(s[idx:]):
            return s[idx:]
    return ""

def first_numeric_substring(s):
    """Returns the longest head of [s] that is numeric."""
    for idx in range(len(s), 0, -1):
        if is_numeric(s[:idx]):
            return s[:idx]
    return ""
    
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

def comma_separated_list_to_list(s):
    """Converts a comma-separated string or list thereof to a list of strings. The key
    use case is in handling various SLURM commands where lists are represented as
    comma-separated strings.
    """
    if isinstance(s, list | tuple | set):
        return UtilsBase.flatten([comma_separated_list_to_list(x) for x in s])
    elif isinstance(s, str):
        return [x.strip() for x in s.split(",") if len(x.strip()) > 0]
    elif isinstance(s, int | float):
        return [str(s)]
    else:
        raise ValueError(f"Unexpected type for comma_separated_list_to_list: {type(s)}. Value: {s}")

######################################################################################
######################################################################################
######################################################################################


###### Colorization utilities ########################################################
# See: https://jakob-bagterp.github.io/colorist-for-python/ansi-escape-codes/extended-256-colors/#extended-palette
color2value_base = dict(
    blue=21,
    green=46,
    yellow=226,
    red=196,
    purple=201,
    lightblue=51,
    white=231,
)

def get_color_scale(*, start, end, mid=None, num_colors=11, light_bias=0):
    """Returns a list of [num_colors] going between [start] and [end].

    Args:
    start       -- start color name
    end         -- end color name 
    mid         -- mid color name. Not required for [num_colors] <= 6
    num_colors  -- number of colors to return
    light_bias  -- shifts the colors to be more grayscale (looks better on black background)
    """
    if num_colors < 2 or num_colors > 11:
        raise ValueError(f"num_colors={num_colors} must be between 2 and 11")

    start = start if isinstance(start, int) else color2value_base[start]
    end = end if isinstance(end, int) else color2value_base[end]
    end_color = reverse_dict(color2value_base)[end]
    start_color = reverse_dict(color2value_base)[start]

    # twrite(start=start, end=end, mid=mid, num_colors=num_colors, end_color=end_color, start_color=start_color)
    
    if num_colors >= 6 and mid is None:
        mid = (end - start) // 2 + start
    elif not mid is None:
        mid = mid if isinstance(mid, int) else color2value_base[mid]
    else:
        pass
        
    if num_colors >= 6:
        scale_delta1 = (mid - start) / 5
        light_bias1_mul = (5+ abs(scale_delta1)) // 6 
        scale_delta2 = (end - mid) / 5
        light_bias2_mul = (5 + abs(scale_delta2)) // 6 
        scale1 = [start + i * scale_delta1 + light_bias * light_bias1_mul for i in range(6)] # Total of six values, puts more resolution near start
        scale2 = [mid + i * scale_delta2 + light_bias * light_bias2_mul for i in range(1,6)] # Total of five values, puts less resolution near end
        scale = scale1 + scale2


        # twrite(light_bias=light_bias, light_bias1_mul=light_bias1_mul, light_bias2_mul=light_bias2_mul, start_color=start_color, end_color=end_color, scale_delta1=scale_delta1, scale_delta2=scale_delta2, )
    else:
        scale_delta = (end - start) / 5
        light_bias_mul = (5 + abs(scale_delta)) // 6
        scale = [start + i * scale_delta + light_bias_mul * light_bias for i in range(7)]

    
    scale = [int(s) for s in scale]
    # twrite(scale=scale, len_scale=len(scale))

    scale_inner = scale[1:-1]
    num_to_select = len(scale_inner) // (num_colors - 2)
    scale_inner = scale_inner[::num_to_select]
    scale_inner = scale_inner[:min(num_colors - 2, len(scale_inner))]

    result = [scale[0]] + scale_inner + [scale[-1]]
    return result

color2value = {c: f"\033[38;5;{v}m" for c,v in (color2value_base | dict(
    reset=0,
    green1=46,
    green2=40,
    green3=34,
    green4=118,
    green5=154,
    yellow1=190,
    yellow2=226,
    yellow3=220,
    blue1=39,
    blue2=27,
    purple1=129,
    purple2=165,  
    orange=214,
    red1=208,
    red2=202,
    red3=196,
)).items()}

def colorize(s, color="no_change"):
    """Returns [s] colorized with ANSI escape codes."""
    color = color2value[color] if color in color2value else color
    color = f"\033[38;5;{color}m" if isinstance(color, int | float) else color
    return s if color == "no_change" else f"{color}{s}\033[0m".strip()

def decolorize(s):
    """Returns [s] with ANSI escape codes removed, eg. so its length is correct."""
    s = copy.deepcopy(s)
    decolorized_s = ""
    while len(s):
        if s.startswith("\x1b["):
            next_valid_idx = s.index("m") + 1
        else:
            next_valid_idx = 1
            decolorized_s += s[0]
        s = s[next_valid_idx:]
    return decolorized_s



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
        # Extract the last numeric characters before the 's', convert these to a float
        # and return time_to_seconds(remainder) + that number of seconds.
        last_numeric_tail = last_numeric_substring(time_str[:-1])
        time_str_head = time_str[:-len(last_numeric_tail)-1]
        time_head = 0 if not time_str_head else time_to_seconds(time_str_head)
        seconds = float(last_numeric_tail) if last_numeric_tail else 0
        return time_head + seconds
    elif time_str.lower().endswith("m"):
        last_numeric_tail = last_numeric_substring(time_str[:-1])
        time_str_head = time_str[:-len(last_numeric_tail)-1]
        time_head = 0 if not time_str_head else time_to_seconds(time_str_head)
        minutes = float(last_numeric_tail) if last_numeric_tail else 0
        return time_head + minutes * 60
    elif time_str.lower().endswith("h"):
        last_numeric_tail = last_numeric_substring(time_str[:-1])
        time_str_head = time_str[:-len(last_numeric_tail)-1]
        time_head = 0 if not time_str_head else time_to_seconds(time_str_head)
        hours = float(last_numeric_tail) if last_numeric_tail else 0
        return time_head + hours * 3600
    elif time_str.lower().endswith("d"):
        last_numeric_tail = last_numeric_substring(time_str[:-1])
        time_str_head = time_str[:-len(last_numeric_tail)-1]
        time_head = 0 if not time_str_head else time_to_seconds(time_str_head)
        days = float(last_numeric_tail) if last_numeric_tail else 0
        return time_head + days * 24 * 3600
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
    s = int(t) if isinstance(t, float | int) else time_to_seconds(t)
    h, m, s = s // 3600, (s % 3600) // 60, s % 60
    return f"{int(h)}:{int(m):02}:{int(s):02}"

def time_to_pretty_str(t):
    """Returns XXHYYM for XX hours and YY minutes. Days are collapsed to hours."""
    s = int(t) if isinstance(t, float | int) else time_to_seconds(t)
    h = s // 3600
    m = (s % 3600) // 60
    h = str(h).zfill(max(2, len(str(h))+1))
    return f"{h}H{m:02d}M"

def format_timestamp(t, time_format="default"):
    """Formats a timestamp [t] as a string. Our canonical format is """
    t_as_datetime = time_stamp_to_datetime(t)

    if time_format == "default": # YYYY_MM_DD_XXhYYm
        return t_as_datetime.strftime("%Y_%m_%d_%Hh%Mm")
    elif time_format == "default_long" or time_format == "default_seconds": # YYYY_MM_DD_XXhYYmZZs
        return t_as_datetime.strftime("%Y_%m_%d_%Hh%Mm%Ss")
    else:
        return t_as_datetime.strftime(time_format)
        
def try_format_timestamp(t, time_format="default"):
    """Tries to format [t] as a timestamp. If [t] cannot be parsed as a timestamp, then
    returns [t] as a string."""
    try:
        return format_timestamp(t, time_format=time_format)
    except Exception:
        return str(t)

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



###### User Query Functions ##########################################################
def query_among_list(*, prompt, options):
    """Returns the element of [options] chosen by the user given [prompt]."""
    assert len(options) > 0, "Cannot query among empty list"
    print(prompt)
    for idx,m in enumerate(options):
        print(f"\t{idx+1}: {m}")
    
    while True:
        choice = input(f"Enter the number of the choice (1-{len(options)}), or 0 to cancel: ")
        if choice.isdigit() and int(choice) == 0:
            raise KeyboardInterrupt()
        elif choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice)-1]
        else:
            print(f"[WARNING] Invalid choice: {choice} -> try again")

def query_yes_no(msg="Proceed? (y/n): "):
    """Queries the user to proceed. Returns True if the user wants to proceed, False otherwise."""
    print(msg)
    while True:
        choice = input("")
        if choice.lower() in ["y", "yes"]:
            return True
        elif choice.lower() in ["n", "no"]:
            return False
        else:
            print(f"[WARNING] Invalid choice: {choice} -> try again")
######################################################################################
######################################################################################
######################################################################################

###### Miscellaneous Functions #######################################################
def run_in_new_thread(fn, *args, **kwargs):
    """Runs [fn] with [args] and [kwargs] in a new thread."""
    thread = Thread(target=fn, args=args, kwargs=kwargs,)
    thread.start()  # Start the thread, it will run in the background
    return thread  # Return the thread object if you need to join or manage it later



def warn_once(message):
    def decorator(fn):
        warned = False

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            nonlocal warned
            if not warned:
                print(message)
                warned = True
            return fn(*args, **kwargs)

        return wrapper
    return decorator
