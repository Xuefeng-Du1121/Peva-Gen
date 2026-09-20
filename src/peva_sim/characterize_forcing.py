"""Characterize held-out ocean-forcing distributions without touching training.

The reported split labels remain protocol labels. This tool only measures how
candidate scenarios compare with the training forcing support, so an IID/OOD
claim is not inferred from date or geographic distance alone.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .forcing_scenarios import support
from .ocean_field import OceanField
from .protocol import file_sha

ROOT = Path(__file__).resolve().parents[2]


def _sample_features(field, row, config, time_count, side, seed):
    region = float(config["region_m"])
    horizon = float(config["horizon_s"])
    origin = row["origin_lonlat"]
    # Use the same local-frame corners as the split-support contract, then
    # sample a deterministic tensor product inside the interpolation domain.
    from .real_env import LocalFrame
    frame = LocalFrame(origin, region)
    corners = frame.inverse(np.array([[0.0, 0.0], [region, region]]))
    lon = np.linspace(corners[:, 0].min(), corners[:, 0].max(), side)
    lat = np.linspace(corners[:, 1].min(), corners[:, 1].max(), side)
    times = np.linspace(float(row["start_utc"]),
                        float(row["start_utc"]) + horizon, time_count)
    tt, yy, xx = np.meshgrid(times, lat, lon, indexing="ij")
    uv = field.query(tt, yy, xx).reshape(-1, 2)
    values = np.column_stack((uv, np.linalg.norm(uv, axis=1)))
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite forcing sample for {row['id']}")
    return values


def summarize(values):
    names = ("u", "v", "speed")
    result = {"samples": int(len(values))}
    for index, name in enumerate(names):
        column = np.asarray(values[:, index], dtype=float)
        result.update({
            f"{name}_mean": float(column.mean()),
            f"{name}_sd": float(column.std(ddof=1)),
            f"{name}_q10": float(np.quantile(column, .10)),
            f"{name}_q50": float(np.quantile(column, .50)),
            f"{name}_q90": float(np.quantile(column, .90)),
        })
    return result


def compare_to_training(summary, training_summary):
    names = ("u", "v", "speed")
    mean = np.array([training_summary[f"{n}_mean"] for n in names])
    scale = np.array([training_summary[f"{n}_sd"] for n in names])
    scale = np.maximum(scale, 1e-8)
    candidate = np.array([summary[f"{n}_mean"] for n in names])
    z = (candidate - mean) / scale
    summary["standardized_mean_distance_l2"] = float(np.linalg.norm(z))
    summary["standardized_mean_z"] = [float(x) for x in z]
    summary["speed_sd_ratio"] = float(
        summary["speed_sd"] / max(training_summary["speed_sd"], 1e-8))
    return summary


def characterize(inventory_path, output_path, csv_path, time_count=8, side=16,
                  seed=20260920, scenario_inventory_path=None):
    inventory_path = Path(inventory_path).resolve()
    scenario_path = (inventory_path if scenario_inventory_path is None else
                     Path(scenario_inventory_path).resolve())
    output_path = Path(output_path).resolve()
    csv_path = Path(csv_path).resolve()
    if not inventory_path.is_relative_to(ROOT) or not scenario_path.is_relative_to(ROOT):
        raise ValueError("inventory must stay under project root")
    if not output_path.is_relative_to(ROOT) or not csv_path.is_relative_to(ROOT):
        raise ValueError("outputs must stay under project root")
    if output_path.exists() or csv_path.exists():
        raise ValueError("refusing to overwrite characterization output")
    if time_count < 2 or side < 2:
        raise ValueError("time-count and side must be at least two")
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    scenarios_inventory = (inventory if scenario_path == inventory_path else
                           json.loads(scenario_path.read_text(encoding="utf-8")))
    config = inventory["config"]
    if scenarios_inventory.get("config") != config:
        raise ValueError("scenario inventory config differs from training")
    if scenarios_inventory.get("field_sha256") != inventory["field_sha256"]:
        raise ValueError("scenario inventory field differs from training")
    field_path = Path(inventory["field_path"]).resolve()
    if file_sha(field_path) != inventory["field_sha256"]:
        raise ValueError("forcing field hash mismatch")
    field = OceanField(field_path)
    rows = scenarios_inventory["scenarios"]
    for row in rows:
        expected = support(field, row["origin_lonlat"], row["start_utc"],
                           config["region_m"], config["horizon_s"])
        if expected != row["support"]:
            raise ValueError(f"scenario support mismatch: {row['id']}")
    train = [row for row in inventory["scenarios"] if row["split"] == "train"]
    if not train:
        raise ValueError("inventory has no training scenarios")
    all_train = np.concatenate([
        _sample_features(field, row, config, time_count, side, seed + i)
        for i, row in enumerate(train)])
    train_summary = summarize(all_train)
    records = []
    for i, row in enumerate(rows):
        values = _sample_features(field, row, config, time_count, side,
                                  seed + 1000 + i)
        summary = compare_to_training(summarize(values), train_summary)
        records.append({"scenario_id": row["id"], "protocol_split": row["split"],
                        **summary})
    report = {
        "schema": "forcing-distribution-characterization-v1",
        "status": "descriptive; does not relabel IID/OOD protocol splits",
        "training_inventory_sha256": file_sha(inventory_path),
        "scenario_inventory_sha256": file_sha(scenario_path),
        "field_sha256": inventory["field_sha256"],
        "source_sha256": file_sha(Path(__file__).resolve()),
        "sampling": {"time_count": time_count, "grid_side": side,
                     "seed": seed, "training_scenarios": [r["id"] for r in train]},
        "training_reference": train_summary,
        "scenarios": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    fields = ["scenario_id", "protocol_split"] + [
        key for key in records[0] if key not in ("scenario_id", "protocol_split")]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--scenario-inventory", default=None)
    parser.add_argument("--time-count", type=int, default=8)
    parser.add_argument("--grid-side", type=int, default=16)
    args = parser.parse_args(argv)
    report = characterize(args.inventory, args.out, args.csv,
                          args.time_count, args.grid_side,
                          scenario_inventory_path=args.scenario_inventory)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
