"""Small immutable source bundles for experiment provenance, not runtime resume."""
import json,tarfile
from pathlib import Path
from .protocol import file_sha
ROOT=Path(__file__).resolve().parents[2]


def snapshot_source(output):
    output=Path(output).resolve()
    if not output.is_relative_to(ROOT): raise ValueError("source bundle outside project")
    paths=sorted((ROOT/"src").rglob("*.py"))+sorted((ROOT/"configs").glob("*.json"))+[ROOT/"run.sh"]
    paths=[p for p in paths if p.is_file()]
    hashes={str(p.relative_to(ROOT)):file_sha(p) for p in paths}
    archive=output/"source.tar.gz"
    with tarfile.open(archive,"x:gz") as tar:
        for p in paths: tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    record={"files":hashes,"archive_sha256":file_sha(archive),
            "scope":"source/config snapshot; raw data referenced separately by hash"}
    with (output/"source-manifest.json").open("x") as f: json.dump(record,f,indent=2)
    return record["archive_sha256"]
