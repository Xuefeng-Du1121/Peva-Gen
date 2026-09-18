"""Build canonical paper tables and audit communication from frozen traces."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from itertools import permutations
from pathlib import Path

from scipy.stats import wilcoxon


METHOD_ORDER = ["mappo", "pvf-mappo", "critic-only-eva-gen", "peva-gen"]
METHOD_LABELS = {
    "mappo": "MAPPO",
    "pvf-mappo": "PVF+MAPPO",
    "critic-only-eva-gen": "critic-only EVA-Gen",
    "peva-gen": "PEVA-Gen",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def float_or_blank(value):
    return "" if value in (None, "") else float(value)


def canonical_main(test_root: Path, out: Path):
    source = test_root / "test-results-table.csv"
    rows = []
    for row in read_csv(source):
        method = row["method"]
        rows.append({
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "success_rate_mean": row["success_rate_mean"],
            "success_rate_sd": row["success_rate_sd"],
            "success_rate_ci95_low": row["success_rate_ci95_low"],
            "success_rate_ci95_high": row["success_rate_ci95_high"],
            "survival_score_mean": row["survival_score_mean"],
            "survival_score_sd": row["survival_score_sd"],
            "survival_score_ci95_low": row["survival_score_ci95_low"],
            "survival_score_ci95_high": row["survival_score_ci95_high"],
            "ttd_all_mean": row["ttd_all_mean"],
            "ttd_all_sd": row["ttd_all_sd"],
            "ttd_all_ci95_low": row["ttd_all_ci95_low"],
            "ttd_all_ci95_high": row["ttd_all_ci95_high"],
            "boundary_fraction_1km_mean": row["boundary_fraction_1km_mean"],
            "boundary_fraction_1km_sd": row["boundary_fraction_1km_sd"],
            "communication_bytes_total_mean": row["communication_bytes_total_mean"],
            "communication_bytes_total_sd": row["communication_bytes_total_sd"],
            "inference_mean_ms_mean": row["inference_mean_ms_mean"],
            "inference_mean_ms_sd": row["inference_mean_ms_sd"],
        })
    rows.sort(key=lambda item: METHOD_ORDER.index(item["method"]))
    write_csv(out / "main-test.csv", rows)
    return source


def copy_canonical(source: Path, destination: Path, method_field="method"):
    rows = read_csv(source)
    for row in rows:
        if method_field in row:
            row["label"] = METHOD_LABELS.get(row[method_field], row[method_field])
    write_csv(destination, rows)


def audit_communication(test_root: Path, out: Path):
    rows = []
    mismatches = []
    config = None
    for job in sorted((test_root / "test").iterdir()):
        summary_path = job / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        method = job.name.rsplit("-seed", 1)[0]
        seed = int(job.name.rsplit("-seed", 1)[1])
        config = config or summary["evaluation_config"]
        n_uavs = int(summary["evaluation_config"]["n_uavs"])
        dt_s = float(summary["evaluation_config"]["dt_s"])
        horizon_s = float(summary["evaluation_config"]["horizon_s"])
        expected_mean = summary["communication_bytes_mean"]
        sums = {key: 0.0 for key in ("peer_payload", "peer_header", "shared_uplink", "shared_downlink")}
        for record in summary["records"]:
            for key in sums:
                sums[key] += float(record["info"]["bytes"][key])
        count = len(summary["records"])
        means = {key: value / count for key, value in sums.items()}
        for key in sums:
            if abs(means[key] - float(expected_mean[key])) > 1e-7:
                mismatches.append({"job": job.name, "field": key, "recomputed": means[key], "reported": expected_mean[key]})
        total = sum(means.values())
        steps_per_episode = horizon_s / dt_s
        rows.append({
            "method": method,
            "label": METHOD_LABELS.get(method, method),
            "seed": seed,
            "episodes": count,
            "peer_payload_bytes": means["peer_payload"],
            "peer_header_bytes": means["peer_header"],
            "shared_uplink_bytes": means["shared_uplink"],
            "shared_downlink_bytes": means["shared_downlink"],
            "peer_bytes_total": means["peer_payload"] + means["peer_header"],
            "shared_bytes_total": means["shared_uplink"] + means["shared_downlink"],
            "communication_bytes_total": total,
            "total_bytes_per_uav_step": total / (steps_per_episode * n_uavs),
            "steps_per_episode": steps_per_episode,
            "n_uavs": n_uavs,
            "message_payload_bytes": 42 if method in {"peva-gen", "critic-only-eva-gen"} else "",
            "radio_header_bytes": 16 if method in {"peva-gen", "critic-only-eva-gen"} else "",
        })
    write_csv(out / "communication-audit.csv", rows)
    payload = {
        "status": "passed" if not mismatches else "failed",
        "jobs": len(rows),
        "mismatches": mismatches,
        "wire_protocol": {
            "quantized_message_payload_bytes": 42,
            "radio_header_bytes": 16,
            "quantized_peer_packet_bytes": 58,
            "quantized_peer_packet_bits": 464,
            "unquantized_fp16_32d_bytes": 64,
            "unquantized_fp16_32d_bits": 512,
        },
        "evaluation_config": config,
    }
    (out / "communication-audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return test_root / "test" / "peva-gen-seed10" / "summary.json"


def all_pairwise(test_root: Path, out: Path):
    rows = read_csv(test_root / "aggregate-test-v1" / "seed-means.csv")
    metrics = ["success_rate", "survival_score", "ttd_all", "boundary_fraction_1km", "peer_bytes_total", "shared_bytes_total", "communication_bytes_total", "inference_mean_ms"]
    by_method = {method: sorted((r for r in rows if r["method"] == method), key=lambda r: int(r["training_seed"])) for method in METHOD_ORDER}
    output = []
    for reference, method in permutations(METHOD_ORDER, 2):
        for metric in metrics:
            ref = [float(r[metric]) for r in by_method[reference]]
            cur = [float(r[metric]) for r in by_method[method]]
            diff = [b - a for a, b in zip(ref, cur)]
            test = wilcoxon(diff, alternative="two-sided", method="exact")
            output.append({"reference": reference, "method": method, "metric": metric, "paired_training_seeds": len(diff), "raw_difference_method_minus_reference": sum(diff) / len(diff), "p_two_sided_exact": float(test.pvalue)})
    write_csv(out / "pairwise-all.csv", output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=False)
    parser.add_argument("--test-root", type=Path)
    parser.add_argument("--validation-root", type=Path)
    parser.add_argument("--ablation-root", type=Path)
    parser.add_argument("--robustness-csv", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.artifacts:
        test_root = args.test_root or (args.artifacts / "formal-main-test-seeds10-14-v1-20260918")
        validation_root = args.validation_root or (args.artifacts / "formal-main-validation")
        ablation_root = args.ablation_root or (args.artifacts / "formal-ablation")
        robustness_csv = args.robustness_csv or (args.artifacts / "robustness-summary.csv")
    else:
        for value, name in ((args.test_root, "--test-root"), (args.validation_root, "--validation-root"),
                            (args.ablation_root, "--ablation-root"), (args.robustness_csv, "--robustness-csv")):
            if value is None:
                parser.error(f"{name} is required when --artifacts is omitted")
        test_root, validation_root, ablation_root, robustness_csv = args.test_root, args.validation_root, args.ablation_root, args.robustness_csv
    canonical_main(test_root, args.out)
    pairwise_csv = test_root / "aggregate-test-v1" / "pairwise.csv"
    if pairwise_csv.exists():
        copy_canonical(pairwise_csv, args.out / "main-pairwise.csv", method_field="method")
    validation_csv = validation_root / "aggregate.csv"
    if not validation_csv.exists():
        validation_csv = validation_root / "aggregate-main-v1" / "aggregate.csv"
    ablation_csv = ablation_root / "aggregate.csv"
    if not ablation_csv.exists():
        ablation_csv = ablation_root / "aggregate-ablation-v1" / "aggregate.csv"
    copy_canonical(validation_csv, args.out / "main-validation.csv")
    copy_canonical(ablation_csv, args.out / "ablation-validation.csv")
    copy_canonical(robustness_csv, args.out / "robustness-validation.csv", method_field="method")
    audit_communication(test_root, args.out)
    all_pairwise(test_root, args.out)
    manifest = {}
    for path in sorted(args.out.glob("*")):
        if path.is_file() and path.name != "results-manifest.json":
            manifest[path.name] = sha256(path)
    (args.out / "results-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
