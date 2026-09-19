"""Audited seed-level aggregation for formal maritime-SAR experiments."""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np


VERSION = "formal_seed_hierarchical_statistics_v1"
METRICS = {
    "success_rate": 1,
    "survival_score": 1,
    "ttd_all": -1,
    "boundary_fraction_1km": -1,
    "peer_bytes_total": -1,
    "shared_bytes_total": -1,
    "communication_bytes_total": -1,
    "inference_mean_ms": -1,
    "belief_mean_error_m": -1,
    "belief_ess_fraction": 1,
    "belief_remaining_mass": 1,
}


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_metric(record, metric):
    if metric in ("success_rate", "survival_score", "ttd_all"):
        return float(record["info"][metric])
    if metric == "boundary_fraction_1km":
        return float(record["trajectory_diagnostics"][metric])
    if metric == "inference_mean_ms":
        return float(record["runtime"][metric])
    if metric == "belief_mean_error_m":
        return float(record["belief_diagnostics"]["posterior_mean_error_m"])
    if metric == "belief_ess_fraction":
        return float(record["belief_diagnostics"]["effective_sample_size_fraction"])
    if metric == "belief_remaining_mass":
        return float(record["belief_diagnostics"]["remaining_mass_mean"])
    byte_counts = record["info"]["bytes"]
    peer = float(byte_counts["peer_payload"] + byte_counts["peer_header"])
    shared = float(byte_counts["shared_uplink"] + byte_counts["shared_downlink"])
    if metric == "peer_bytes_total":
        return peer
    if metric == "shared_bytes_total":
        return shared
    if metric == "communication_bytes_total":
        return peer + shared
    raise KeyError(metric)


def extract_records(summary, method):
    records = summary["records"]
    if isinstance(records, dict):
        if method not in records:
            raise ValueError(f"Method {method!r} absent from classical summary")
        records = records[method]
    if not isinstance(records, list) or not records:
        raise ValueError("Expected nonempty episode records")
    extracted = []
    seen = set()
    for record in records:
        key = (str(record["scenario_id"]), int(record["seed"]))
        if key in seen:
            raise ValueError(f"Duplicate evaluation episode key {key}")
        seen.add(key)
        values = {metric: _record_metric(record, metric) for metric in METRICS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Nonfinite episode metric")
        extracted.append({"key": key, "values": values})
    return extracted


def average_ranks(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2
        start = stop
    return ranks


def exact_wilcoxon_two_sided(differences):
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[~np.isclose(differences, 0, rtol=0, atol=1e-15)]
    if len(differences) == 0:
        return {"n_nonzero": 0, "w_plus": 0.0, "p_two_sided": 1.0}
    if len(differences) > 20:
        raise ValueError("Exact Wilcoxon enumeration is limited to 20 paired seeds")
    ranks = average_ranks(np.abs(differences))
    observed = float(ranks[differences > 0].sum())
    midpoint = float(ranks.sum() / 2)
    distance = abs(observed - midpoint)
    extreme = 0
    total = 1 << len(ranks)
    for mask in range(total):
        value = sum(rank for index, rank in enumerate(ranks)
                    if mask & (1 << index))
        extreme += abs(value - midpoint) >= distance - 1e-12
    return {
        "n_nonzero": int(len(differences)),
        "w_plus": observed,
        "p_two_sided": float(extreme / total),
    }


def hierarchical_group_bootstrap(runs, metric, samples, rng):
    result = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.integers(len(runs), size=len(runs))
        seed_means = []
        for index in selected:
            values = np.asarray([row["values"][metric] for row in runs[index]["records"]])
            seed_means.append(values[rng.integers(len(values), size=len(values))].mean())
        result[draw] = np.mean(seed_means)
    return result


def paired_hierarchical_bootstrap(left, right, metric, samples, rng):
    pairs = []
    for seed in sorted(set(left) & set(right)):
        left_rows = {row["key"]: row for row in left[seed]["records"]}
        right_rows = {row["key"]: row for row in right[seed]["records"]}
        if left_rows.keys() != right_rows.keys():
            raise ValueError(f"Episode keys differ for paired seed {seed}")
        differences = np.asarray([
            right_rows[key]["values"][metric] - left_rows[key]["values"][metric]
            for key in sorted(left_rows)])
        pairs.append(differences)
    result = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        selected = rng.integers(len(pairs), size=len(pairs))
        seed_effects = []
        for index in selected:
            values = pairs[index]
            seed_effects.append(values[rng.integers(len(values), size=len(values))].mean())
        result[draw] = np.mean(seed_effects)
    return result


def percentile_interval(values):
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", nargs=3, action="append", required=True,
        metavar=("METHOD", "TRAIN_SEED", "SUMMARY_JSON"))
    parser.add_argument("--reference", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260914)
    parser.add_argument("--allow-single-seed", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    args = parser.parse_args(argv)
    if args.bootstrap_samples < 1000:
        parser.error("At least 1000 bootstrap samples are required")

    output = Path(args.out).resolve()
    if output.exists():
        parser.error("Refusing to overwrite output")
    loaded = []
    protocol = None
    for method, seed_text, path_text in args.run:
        path = Path(path_text).resolve()
        summary = json.loads(path.read_text())
        split = summary["split"]
        if split == "test" and not args.allow_test:
            parser.error("Test summaries require explicit --allow-test")
        signature = {
            "split": split,
            "scenarios": summary["scenarios"],
            "evaluation_config": summary["evaluation_config"],
            "environment_uncertainty": summary.get("environment_uncertainty"),
            "robustness_overrides": summary.get("robustness_overrides", {}),
        }
        if protocol is None:
            protocol = signature
        elif signature != protocol:
            raise ValueError("Evaluation protocols differ across inputs")
        seed = int(seed_text)
        loaded.append({
            "method": method,
            "seed": seed,
            "path": str(path),
            "sha256": file_sha(path),
            "records": extract_records(summary, method),
            "policy_communication": summary.get("policy_communication", {}),
        })

    grouped = {}
    for run in loaded:
        method_runs = grouped.setdefault(run["method"], {})
        if run["seed"] in method_runs:
            raise ValueError("Duplicate method/training-seed input")
        method_runs[run["seed"]] = run
    if args.reference not in grouped:
        parser.error("Reference method is absent")
    if not args.allow_single_seed and any(len(runs) < 2 for runs in grouped.values()):
        parser.error("Every method requires at least two training seeds")
    seed_sets = {method: set(runs) for method, runs in grouped.items()}
    if any(seeds != seed_sets[args.reference] for seeds in seed_sets.values()):
        raise ValueError("Methods must use the same training seeds")

    rng = np.random.default_rng(args.bootstrap_seed)
    seed_rows = []
    aggregates = {}
    for method, by_seed in grouped.items():
        runs = [by_seed[seed] for seed in sorted(by_seed)]
        aggregates[method] = {
            "n_training_seeds": len(runs),
            "episodes_per_seed": [len(run["records"]) for run in runs],
            "metrics": {},
            "policy_communication": runs[0]["policy_communication"],
        }
        for seed, run in sorted(by_seed.items()):
            row = {"method": method, "training_seed": seed,
                   "episodes": len(run["records"])}
            for metric in METRICS:
                row[metric] = float(np.mean([
                    record["values"][metric] for record in run["records"]]))
            seed_rows.append(row)
        for metric in METRICS:
            values = np.asarray([
                next(row[metric] for row in seed_rows
                     if row["method"] == method and row["training_seed"] == seed)
                for seed in sorted(by_seed)])
            bootstrap = hierarchical_group_bootstrap(runs, metric, args.bootstrap_samples, rng)
            aggregates[method]["metrics"][metric] = {
                "mean_of_seed_means": float(values.mean()),
                "std_across_seed_means": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "hierarchical_bootstrap_95_ci": percentile_interval(bootstrap),
                "direction": "higher_is_better" if METRICS[metric] > 0 else "lower_is_better",
            }

    pairwise = []
    reference = grouped[args.reference]
    for method, candidate in grouped.items():
        if method == args.reference:
            continue
        for metric, direction in METRICS.items():
            seed_differences = []
            for seed in sorted(reference):
                left = np.mean([row["values"][metric] for row in reference[seed]["records"]])
                right = np.mean([row["values"][metric] for row in candidate[seed]["records"]])
                seed_differences.append(right - left)
            bootstrap = paired_hierarchical_bootstrap(
                reference, candidate, metric, args.bootstrap_samples, rng)
            wilcoxon = exact_wilcoxon_two_sided(seed_differences)
            pairwise.append({
                "reference": args.reference,
                "method": method,
                "metric": metric,
                "paired_training_seeds": len(seed_differences),
                "raw_difference_method_minus_reference": float(np.mean(seed_differences)),
                "favorable_effect": float(direction * np.mean(seed_differences)),
                "paired_hierarchical_bootstrap_95_ci": percentile_interval(bootstrap),
                **wilcoxon,
            })

    output.mkdir(parents=True)
    aggregate_rows = []
    for method, payload in aggregates.items():
        for metric, values in payload["metrics"].items():
            aggregate_rows.append({
                "method": method,
                "metric": metric,
                "n_training_seeds": payload["n_training_seeds"],
                "mean_of_seed_means": values["mean_of_seed_means"],
                "std_across_seed_means": values["std_across_seed_means"],
                "ci95_low": values["hierarchical_bootstrap_95_ci"][0],
                "ci95_high": values["hierarchical_bootstrap_95_ci"][1],
                "direction": values["direction"],
            })
    pairwise_csv = [{
        **{key: value for key, value in row.items()
           if key != "paired_hierarchical_bootstrap_95_ci"},
        "ci95_low": row["paired_hierarchical_bootstrap_95_ci"][0],
        "ci95_high": row["paired_hierarchical_bootstrap_95_ci"][1],
    } for row in pairwise]
    write_csv(output / "seed-means.csv", seed_rows)
    write_csv(output / "aggregate.csv", aggregate_rows)
    write_csv(output / "pairwise.csv", pairwise_csv)
    result = {
        "version": VERSION,
        "protocol": protocol,
        "reference": args.reference,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.bootstrap_seed,
                      "unit": "training seed, then episode within seed"},
        "inputs": [{key: run[key] for key in ("method", "seed", "path", "sha256")}
                   for run in loaded],
        "aggregates": aggregates,
        "pairwise": pairwise,
        "source_sha256": file_sha(__file__),
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
