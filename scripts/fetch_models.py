# Download the PUBLIC model bundle into models/

import io
import os
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = re.compile(r"^(role|type)_[A-Za-z_]+\.(txt|json)$")

url = os.environ.get("MODELS_URL")
if not url:
    sys.exit("MODELS_URL is not set")
out = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
out.mkdir(parents=True, exist_ok=True)

with urllib.request.urlopen(url, timeout=120) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))

installed, refused = [], []
for member in archive.namelist():
    name = Path(member).name
    if not name:
        continue
    if ALLOWED.match(name):
        (out / name).write_bytes(archive.read(member))
        installed.append(name)
    else:
        refused.append(name)

if refused:
    print("skipped (not public model files):", ", ".join(refused))
if not installed:
    sys.exit("no role_/type_ model files found in the zip")
print(f"installed {len(installed)} files into {out}")
