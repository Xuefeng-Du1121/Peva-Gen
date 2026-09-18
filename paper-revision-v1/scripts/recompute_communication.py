"""Recompute communication bytes from audited episode summaries."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ("peer_payload", "peer_header", "shared_uplink", "shared_downlink")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    mismatches = []
    for job in sorted((args.test_root / "test").iterdir()):
        summary_path = job / "summary.json"
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        sums = {key: 0.0 for key in FIELDS}
        for record in summary["records"]:
            for key in FIELDS:
                sums[key] += float(record["info"]["bytes"][key])
        count = len(summary["records"])
        means = {key: value / count for key, value in sums.items()}
        for key in FIELDS:
            if abs(means[key] - float(summary["communication_bytes_mean"][key])) > 1e-7:
                mismatches.append({"job": job.name, "field": key, "recomputed": means[key], "reported": summary["communication_bytes_mean"][key]})
        cfg = summary["evaluation_config"]
        total = sum(means.values())
        rows.append({"job": job.name, "method": job.name.rsplit("-seed", 1)[0], "seed": job.name.rsplit("-seed", 1)[1], **{f"{key}_bytes": means[key] for key in FIELDS}, "communication_bytes_total": total, "bytes_per_uav_step": total / ((float(cfg["horizon_s"]) / float(cfg["dt_s"])) * int(cfg["n_uavs"]))})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {"status": "passed" if not mismatches else "failed", "jobs": len(rows), "mismatches": mismatches}
    args.out.with_suffix(".json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if mismatches:
        raise SystemExit("communication audit failed")


if __name__ == "__main__":
    main()
