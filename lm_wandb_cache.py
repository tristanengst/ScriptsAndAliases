"""lm_wandb_cache: a local cache of WandB runs, updated incrementally, built for LM agents.

Each sync fetches only runs WandB says were updated since the previous one (with a margin),
so repeated result checks take seconds and print little. The cache is one JSON file per
project in this machine's $TMP/lm_wandb_cache; delete it (or sync with --full 1) to rebuild
it from WandB. It keeps each run's id, name, state, tags, config and scalar summary values
(nested summary dicts are flattened to dotted keys; media are dropped). Runs deleted in
WandB stay cached until a full sync.

Commands:
sync  --project P [--full 1]          One line: runs cached and updated.
table --project P --tags T ... --keys K ... [--groupby G ...] [--agg mean std ...]
                                      A compact table of summary keys over runs with any of
                                      the tags, grouped by config values (syncs first).

Python use:
import sys, os.path as osp; sys.path.insert(0, osp.expanduser("~/.ScriptsAndAliases"))
import lm_wandb_cache
runs = lm_wandb_cache.get_runs("apex-lab/BettercIMLE", tags=["Oct08-Toy2D-1-5"])  # List of dicts

Example:
python3 ~/.ScriptsAndAliases/lm_wandb_cache.py table --project apex-lab/BettercIMLE --tags Oct08-Toy2D-1-6 --groupby ns imle_latent_rv --keys kl.mean.jsd --agg mean min count
"""
import argparse, datetime, json, os, os.path as osp, statistics, sys, time
from collections import defaultdict

MARGIN_S = 900  # Seconds subtracted from the last sync time, so runs updated during it aren't missed

def cache_path(project):
    """Returns this machine's cache file for WandB [project] (ENTITY/PROJECT)."""
    user = os.environ.get("USER", "user")
    tmp = f"/localscratch/{user}/tmp" if osp.isdir(f"/localscratch/{user}") else f"/tmp/{user}"
    _ = os.makedirs(osp.join(tmp, "lm_wandb_cache"), exist_ok=True)
    return osp.join(tmp, "lm_wandb_cache", project.replace("/", "__") + ".json")

def flatten(d, prefix=""):
    """Returns the scalar entries of nested dict [d] with dotted keys, dropping media."""
    out = dict()
    for k, v in d.items():
        if isinstance(v, dict) and not "_type" in v:
            out |= flatten(v, f"{prefix}{k}.")
        elif v is None or isinstance(v, bool | int | float | str):
            out[f"{prefix}{k}"] = v
    return out

def run_to_dict(r):
    """Returns the cached form of WandB run [r]."""
    return dict(id=r.id, name=r.name, state=r.state, tags=list(r.tags), created=r._attrs.get("createdAt"),
        config={k: v for k, v in r.config.items() if not k.startswith("_")}, summary=flatten(r.summary._json_dict))

def sync(project, *, full=False):
    """Updates the cache of [project] from WandB and returns (dict mapping run IDs to runs,
    number of runs fetched).
    """
    import wandb
    f = cache_path(project)
    cache = json.load(open(f)) if osp.exists(f) and not full else dict(synced_at=None, runs=dict())
    start = datetime.datetime.now(datetime.timezone.utc)
    filters = dict()
    if cache["synced_at"]:
        since = datetime.datetime.fromisoformat(cache["synced_at"]) - datetime.timedelta(seconds=MARGIN_S)
        filters = {"updatedAt": {"$gt": since.strftime("%Y-%m-%dT%H:%M:%S")}}
    fetched = [run_to_dict(r) for r in wandb.Api(timeout=120).runs(project, filters=filters, per_page=500)]
    cache["runs"] |= {r["id"]: r for r in fetched}
    cache["synced_at"] = start.isoformat()
    tmp = f"{f}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(cache, fh)
    os.replace(tmp, f)
    return cache["runs"], len(fetched)

def get_runs(project, *, tags=None, states=None, refresh=True):
    """Returns the cached runs of [project] (synced first if [refresh]) with any of [tags]
    and a state in [states] (all if None), as dicts with keys id, name, state, tags,
    created, config and summary.
    """
    if refresh or not osp.exists(cache_path(project)):
        id2run, _ = sync(project)
    else:
        id2run = json.load(open(cache_path(project)))["runs"]
    return [r for r in id2run.values() if (tags is None or set(tags) & set(r["tags"])) and (states is None or r["state"] in states)]

################################################################################
# Table
################################################################################
agg2fn = dict(mean=statistics.mean, std=lambda vs: statistics.stdev(vs) if len(vs) > 1 else float("nan"),
    min=min, max=max, count=len, vals=lambda vs: " ".join(format_value(v) for v in vs))

def format_value(v):
    """Returns [v] as a short string."""
    if isinstance(v, bool | int) or not isinstance(v, float):
        return str(v)
    return f"{v:.3g}" if v == 0 or 1e-3 <= abs(v) < 1e4 else f"{v:.2e}"

def sort_key(v):
    """Returns a key sorting config values numerically where possible, with None first."""
    return (0, 0.0, "") if v is None else (1, float(v), "") if isinstance(v, int | float) else (2, 0.0, str(v))

def table(runs, *, groupby, keys, aggs, uids=False):
    """Returns the lines of a table of [keys] in [runs]' summaries aggregated by [aggs] over
    groups of runs sharing the config values of [groupby].
    """
    group2runs = defaultdict(list)
    for r in runs:
        group2runs[tuple(r["config"].get(g) for g in groupby)].append(r)
    header = groupby + [f"{k}:{a}" for k in keys for a in aggs] + (["uids"] if uids else [])
    rows = []
    for group in sorted(group2runs, key=lambda g: [sort_key(v) for v in g]):
        k2vals = {k: [v for r in group2runs[group] if isinstance(v := r["summary"].get(k), int | float) and not isinstance(v, bool)] for k in keys}
        cells = [format_value(agg2fn[a](k2vals[k])) if k2vals[k] else "-" for k in keys for a in aggs]
        ids = " ".join(f"{r['id']}{'' if r['state'] == 'finished' else '*'}" for r in group2runs[group])
        rows.append([format_value(v) for v in group] + cells + ([ids] if uids else []))
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    return [f"{len(runs)} runs in {len(rows)} groups" + (" (* = not finished)" if uids else "")] + \
        ["  ".join(str(x).ljust(w) for x, w in zip(row, widths)).rstrip() for row in [header] + rows]

def get_args():
    P = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    C = P.add_subparsers(dest="cmd", required=True)
    PS = C.add_parser("sync")
    PS.add_argument("--project", required=True, help="WandB ENTITY/PROJECT")
    PS.add_argument("--full", type=int, default=0, choices=[0, 1], help="Rebuild the cache from scratch")
    PT = C.add_parser("table")
    PT.add_argument("--project", required=True, help="WandB ENTITY/PROJECT")
    PT.add_argument("--tags", nargs="+", required=True, help="Runs with any of these tags are included")
    PT.add_argument("--keys", nargs="+", required=True, help="Summary keys to show, eg. kl.mean.jsd")
    PT.add_argument("--groupby", nargs="+", default=[], help="Config arguments whose values define the groups")
    PT.add_argument("--agg", nargs="+", choices=list(agg2fn), default=["mean"], help="Aggregations over each group's runs")
    PT.add_argument("--state", nargs="+", default=["finished"], help="Run states to include")
    PT.add_argument("--uids", type=int, choices=[0, 1], default=0, help="Whether to list each group's run IDs")
    PT.add_argument("--refresh", type=int, choices=[0, 1], default=1, help="Sync with WandB first")
    return P.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.cmd == "sync":
        t = time.time()
        id2run, n = sync(args.project, full=bool(args.full))
        print(f"{len(id2run)} runs cached, {n} fetched in {time.time() - t:.0f}s")
    else:
        runs = get_runs(args.project, tags=args.tags, states=args.state, refresh=bool(args.refresh))
        print("\n".join(table(runs, groupby=args.groupby, keys=args.keys, aggs=args.agg, uids=bool(args.uids))))
