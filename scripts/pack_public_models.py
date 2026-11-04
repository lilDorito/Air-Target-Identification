# Zip only the public models for a GitHub release

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
files = sorted(p for p in (ROOT / "models").iterdir() if p.name.startswith(("role_", "type_")) and p.suffix in {".txt", ".json"})
stems = {p.stem for p in files}
missing = [s for s in stems if not (ROOT / "models" / f"{s}.json").exists() or not (ROOT / "models" / f"{s}.txt").exists()]
if not files or missing:
    raise SystemExit(f"missing model files: {missing or 'no role_/type_ models'} - run scripts/build_bundle.py first")
target = ROOT / "public_models.zip"
with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.name)
print(f"{target.name}: {len(files)} files, {target.stat().st_size / 1e6:.1f} MB")
