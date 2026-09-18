"""Freeze an audited development ocean inventory for the formal study."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import time

from .forcing_scenarios import validate_split_support
from .protocol import file_sha


ROOT = Path(__file__).resolve().parents[2]


def configured_data_root():
    """Return the user-configured data root without embedding a host path."""
    return Path(os.environ.get("PEVA_DATA_ROOT", ROOT / "data")).expanduser().resolve()


def validate_candidate(candidate, candidate_path):
    if candidate.get("schema") != "ocean-forcing-inventory-v1":
        raise ValueError("candidate must use ocean-forcing-inventory-v1")
    if not str(candidate.get("status", "")).startswith("development candidate"):
        raise ValueError("only a development candidate can be frozen")
    field_path = Path(candidate.get("field_path", "")).resolve()
    data_root = configured_data_root()
    if not field_path.is_file() or not field_path.is_relative_to(data_root):
        raise ValueError(f"field_path must be an existing file below {data_root}")
    if file_sha(field_path) != candidate.get("field_sha256"):
        raise ValueError("ocean field SHA-256 mismatch")
    scenarios = candidate.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise ValueError("candidate has no scenarios")
    ids = [row.get("id") for row in scenarios]
    if None in ids or len(set(ids)) != len(ids):
        raise ValueError("scenario ids must be present and unique")
    counts = Counter(row.get("split") for row in scenarios)
    if set(counts) != {"train", "validation", "test"}:
        raise ValueError("candidate must contain train, validation and test splits")
    if counts["train"] < 4 or counts["validation"] < 2 or counts["test"] < 2:
        raise ValueError("insufficient scenarios in one or more splits")
    recomputed = validate_split_support(scenarios)
    if recomputed.get("status") != "disjoint":
        raise ValueError("scenario split support is not disjoint")
    if candidate.get("audit", {}).get("status") != "disjoint":
        raise ValueError("candidate does not record a successful split audit")
    return field_path, counts, recomputed


def freeze(candidate, candidate_path, frozen_at=None):
    field_path, counts, audit = validate_candidate(candidate, candidate_path)
    result = copy.deepcopy(candidate)
    result["schema"] = "ocean-forcing-inventory-v2"
    result["status"] = "frozen formal protocol"
    result["parent_inventory_path"] = str(candidate_path.resolve())
    result["parent_inventory_sha256"] = file_sha(candidate_path)
    result["audit"] = audit
    result["formal_freeze"] = {
        "frozen_unix": time.time() if frozen_at is None else frozen_at,
        "host": socket.gethostname(),
        "tool": "peva_sim.freeze_forcing_inventory",
        "tool_sha256": file_sha(__file__),
        "field_path": str(field_path),
        "field_sha256_verified": True,
        "split_counts": dict(counts),
        "selection_stage": "before final multi-seed training and all test evaluation",
        "test_access": "locked by formal-study driver unless --unlock-test is explicit",
    }
    limitations = list(result.get("limitations", []))
    note = ("The original scenario-spec path is unavailable in this project; "
            "the immutable parent inventory and its recorded spec SHA-256 are retained.")
    if note not in limitations:
        limitations.append(note)
    result["limitations"] = limitations
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    candidate_path = (ROOT / args.candidate).resolve()
    output = (ROOT / args.out).resolve()
    if not candidate_path.is_file() or not candidate_path.is_relative_to(ROOT):
        parser.error("candidate must be an existing project file")
    if not output.is_relative_to(ROOT) or output.exists():
        parser.error("output must be a new project file")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    result = freeze(candidate, candidate_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"],
                      "output": str(output),
                      "parent_sha256": result["parent_inventory_sha256"],
                      "field_sha256": result["field_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
