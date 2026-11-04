# Check if data-like files staged for commit

import subprocess
import sys

BLOCKED_EXT = (".csv", ".xlsx", ".xls", ".parquet", ".feather", ".pkl", ".npz", ".zip")
BLOCKED_NAMES = ("fills.py", "recall.py", "build_table.py", "build_xlsx.py", ".env")
BLOCKED_PREFIX = ("data/", "models/aircraft", "data_private/")

ALLOWED_CSV_DIR = "results/metrics/"
AIRCRAFT_COLUMNS = {"name", "aircraft", "aircraft_id", "manufacturer", "callsign", "country"}

def allowed_metrics_csv(path):
    if not (path.startswith(ALLOWED_CSV_DIR) and path.endswith(".csv") and "/" not in path[len(ALLOWED_CSV_DIR):]):
        return False
    staged = subprocess.run(["git", "show", f":{path}"], capture_output=True, text=True).stdout
    header = {c.strip().strip('"').lower() for c in staged.splitlines()[0].split(",")} if staged else set()
    return not (header & AIRCRAFT_COLUMNS)

staged = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"], capture_output=True, text=True).stdout.split()
bad = [f for f in staged
       if (f.endswith(BLOCKED_EXT) and not allowed_metrics_csv(f))
       or f.split("/")[-1] in BLOCKED_NAMES
       or (f.startswith(BLOCKED_PREFIX) and not f.endswith("README.md"))
       or "per_aircraft" in f or "aircraft_confusion" in f]
if bad:
    print("Blocked: data-like files staged:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("OK: no data-like files staged")