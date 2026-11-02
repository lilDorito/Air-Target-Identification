import subprocess
import sys

BLOCKED_EXT = (".csv", ".xlsx", ".xls", ".parquet", ".feather", ".pkl", ".npz")
BLOCKED_NAMES = ("fills.py", "recall.py", "build_table.py", "build_xlsx.py")
BLOCKED_PREFIX = ("data/", "models/aircraft", "data_private/")

staged = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True, text=True).stdout.split()
bad = [f for f in staged if f.endswith(BLOCKED_EXT) or f.split("/")[-1] in BLOCKED_NAMES or (f.startswith(BLOCKED_PREFIX) and not f.endswith("README.md")) or "per_aircraft" in f or "aircraft_confusion" in f]
if bad:
    print("Blocked: data-like files staged:\n  " + "\n  ".join(bad))
    sys.exit(1)
print("OK: no data-like files staged.")
