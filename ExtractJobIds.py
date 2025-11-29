"""Extracts line numbers from an `squeue`-derived input."""
import argparse

def line_to_jobid(l):
    """Returns the JobID from an squeue line. By assumption, it is contained in the
    first all-numeric string when the line is split by whitespace. If it contains an
    underscore, only the portion before the underscore is returned. If not such
    substring is found, None is returned.
    """
    for ll in l.split():
        if ll.isnumeric():
            return ll[:ll.index("_")] if "_" in ll else ll
    return None

if __name__ == "__main__":
    P = argparse.ArgumentParser()
    P.add_argument("sq", type=str, help="squeue output")
    P.add_argument("--require_substring", type=str, default=None,
        help="Only include lines with this substring")
    args = P.parse_args()

    lines = args.sq.split("\n")
    lines = [l for l in lines if args.require_substring is None or args.require_substring in l]
    jobids = [line_to_jobid(l) for l in lines]
    jobids = [j for j in jobids if not j is None]
    print(" ".join(jobids))