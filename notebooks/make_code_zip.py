"""Build the code zip that accompanies a portal upload.

    python notebooks/make_code_zip.py submissions/v4/v4_code.zip
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
out = Path(sys.argv[1])
files = [p for p in (ROOT / "code/business_entity_resolution").rglob("*")
         if p.is_file() and "__pycache__" not in p.parts]
extra = {ROOT / "infra/ec2_worker.py": "code/infra/ec2_worker.py"}
extra |= {p: f"code/kaggle/embed_knn/{p.name}" for p in (ROOT / "kaggle/embed_knn").glob("*") if p.is_file()}
extra |= {ROOT / "notebooks" / n: f"code/notebooks/{n}" for n in ("run_v3.py", "run_v4.py", "make_submission.py", "make_code_zip.py")}
out.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for p in files:
        z.write(p, p.relative_to(ROOT).as_posix())
    for p, arc in extra.items():
        if p.exists():
            z.write(p, arc)
print(f"{out}: {len(z.namelist()) if False else len(files) + sum(p.exists() for p in extra)} files")
