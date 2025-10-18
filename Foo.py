import argparse
P = argparse.ArgumentParser()
P.add_argument("--foo", "--bar", type=int, default=42, help="An example argument.")
args = P.parse_args()
print(f"The value of foo/bar is: {args.foo}")