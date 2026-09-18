"""Generate paper figures from canonical tables."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ["mappo", "pvf-mappo", "critic-only-eva-gen", "peva-gen"]
LABELS = {"mappo": "MAPPO", "pvf-mappo": "PVF+MAPPO", "critic-only-eva-gen": "critic-only EVA-Gen", "peva-gen": "PEVA-Gen"}
COLORS = {"mappo": "#4C78A8", "pvf-mappo": "#F58518", "critic-only-eva-gen": "#54A24B", "peva-gen": "#E45756"}


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_source(out, name, header, rows):
    out.mkdir(parents=True, exist_ok=True)
    with (out / f"{name}-source.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def main_results(results, out):
    rows = {row["method"]: row for row in read_csv(results / "main-test.csv")}
    panels = [
        ("success_rate", "Success rate", "success_rate_mean", "success_rate_ci95_low", "success_rate_ci95_high"),
        ("survival_score", "Survival score", "survival_score_mean", None, None),
        ("ttd_all", "Normalized TTD", "ttd_all_mean", "ttd_all_ci95_low", "ttd_all_ci95_high"),
        ("communication", "Total communication (bytes)", "communication_bytes_total_mean", None, None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.6))
    for ax, (_, title, mean_key, low_key, high_key) in zip(axes.ravel(), panels):
        xs = np.arange(len(ORDER))
        values = np.array([float(rows[m][mean_key]) for m in ORDER])
        ax.bar(xs, values, color=[COLORS[m] for m in ORDER], width=0.72)
        if low_key:
            low = np.array([float(rows[m][low_key]) for m in ORDER])
            high = np.array([float(rows[m][high_key]) for m in ORDER])
            ax.errorbar(xs, values, yerr=[values - low, high - values], fmt="none", color="#222222", capsize=3, lw=1)
        ax.set_title(title)
        ax.set_xticks(xs, [LABELS[m] for m in ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    write_source(out, "fig1_main_test_results", ["method", "success_rate", "survival_score", "ttd_all", "communication_bytes_total"],
                 [[LABELS[m], rows[m]["success_rate_mean"], rows[m]["survival_score_mean"], rows[m]["ttd_all_mean"], rows[m]["communication_bytes_total_mean"]] for m in ORDER])
    save(fig, out, "fig1_main_test_results")


def paired_improvement(results, out):
    main = {r["method"]: r for r in read_csv(results / "main-test.csv")}
    refs = ["mappo", "pvf-mappo", "critic-only-eva-gen"]
    metrics = ["success_rate", "survival_score", "ttd_all"]
    keys = {"success_rate": "success_rate_mean", "survival_score": "survival_score_mean", "ttd_all": "ttd_all_mean"}
    rows = [{"reference": ref, "metric": metric, "raw_difference_method_minus_reference": str(float(main["peva-gen"][keys[metric]]) - float(main[ref][keys[metric]]))} for ref in refs for metric in metrics]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    x = np.arange(len(refs))
    width = 0.24
    for j, metric in enumerate(metrics):
        vals = [float(next(r["raw_difference_method_minus_reference"] for r in rows if r["reference"] == ref and r["metric"] == metric)) for ref in refs]
        ax.bar(x + (j - 1) * width, vals, width, label={"success_rate": "SR", "survival_score": "Survival", "ttd_all": "TTD (negative is better)"}[metric])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x, [LABELS[r] for r in refs])
    ax.set_ylabel("PEVA-Gen minus reference")
    ax.set_title("Paired improvement on the independent test split")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    write_source(out, "fig2_paired_improvement", ["reference", "metric", "raw_difference_method_minus_reference"], [[r["reference"], r["metric"], r["raw_difference_method_minus_reference"]] for r in rows])
    save(fig, out, "fig2_paired_improvement")


def ablation(results, out):
    rows = read_csv(results / "ablation-validation.csv")
    methods = ["peva-gen", "peva-no-rsr", "peva-deterministic-belief", "peva-no-coverage", "peva-no-action-prior", "peva-raw-context"]
    labels = {"peva-gen": "PEVA-Gen", "peva-no-rsr": "no RSR", "peva-deterministic-belief": "deterministic belief", "peva-no-coverage": "no coverage", "peva-no-action-prior": "no action prior", "peva-raw-context": "raw context"}
    by = {(r["method"], r["metric"]): r for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, metric, title in [(axes[0], "success_rate", "Success rate"), (axes[1], "ttd_all", "Normalized TTD")]:
        vals = [float(by[(m, metric)]["mean_of_seed_means"]) for m in methods]
        errs = [float(by[(m, metric)]["std_across_seed_means"]) for m in methods]
        ax.bar(np.arange(len(methods)), vals, yerr=errs, capsize=3, color=[COLORS["peva-gen"] if m == "peva-gen" else "#9D9D9D" for m in methods])
        ax.set_title(title)
        ax.set_xticks(np.arange(len(methods)), [labels[m] for m in methods], rotation=35, ha="right")
        ax.grid(axis="y", alpha=0.25)
    write_source(out, "fig3_ablation_validation", ["method", "metric", "mean", "sd"], [[m, metric, by[(m, metric)]["mean_of_seed_means"], by[(m, metric)]["std_across_seed_means"]] for m in methods for metric in ("success_rate", "ttd_all")])
    save(fig, out, "fig3_ablation_validation")


def robustness(results, out):
    rows = [r for r in read_csv(results / "robustness-validation.csv") if r["metric"] == "success_rate"]
    stressors = list(dict.fromkeys(r["stressor"] for r in rows))
    methods = ["mappo", "pvf-mappo", "peva-gen"]
    matrix = np.array([[float(next(r["mean_of_seed_means"] for r in rows if r["stressor"] == s and r["method"] == m)) for m in methods] for s in stressors])
    fig, ax = plt.subplots(figsize=(8.5, max(4.5, len(stressors) * 0.34)))
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(methods)), [LABELS[m] for m in methods])
    ax.set_yticks(np.arange(len(stressors)), stressors)
    for i in range(len(stressors)):
        for j in range(len(methods)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white" if matrix[i, j] < 0.55 else "black", fontsize=8)
    ax.set_title("Validation stress-test success rate")
    fig.colorbar(image, ax=ax, label="Success rate")
    write_source(out, "fig4_robustness_validation", ["stressor", "method", "success_rate"], [[s, m, matrix[i, j]] for i, s in enumerate(stressors) for j, m in enumerate(methods)])
    save(fig, out, "fig4_robustness_validation")


def communication(results, out):
    rows = {r["method"]: r for r in read_csv(results / "communication-audit.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    xs = np.arange(len(ORDER))
    totals = [float(rows[m]["communication_bytes_total"]) for m in ORDER]
    normalized = [float(rows[m]["total_bytes_per_uav_step"]) for m in ORDER]
    axes[0].bar(xs, totals, color=[COLORS[m] for m in ORDER])
    axes[0].set_title("Total communication per episode")
    axes[0].set_ylabel("bytes")
    axes[1].bar(xs, normalized, color=[COLORS[m] for m in ORDER])
    axes[1].set_title("Communication per UAV-step")
    axes[1].set_ylabel("bytes / UAV-step")
    for ax in axes:
        ax.set_xticks(xs, [LABELS[m] for m in ORDER], rotation=25, ha="right")
        ax.grid(axis="y", alpha=0.25)
    write_source(out, "fig5_communication_audit", ["method", "total_bytes", "bytes_per_uav_step"], [[LABELS[m], totals[i], normalized[i]] for i, m in enumerate(ORDER)])
    save(fig, out, "fig5_communication_audit")


def qualitative(test_root: Path, out):
    peva = np.load(test_root / "test" / "peva-gen-seed10" / "episode-0000.npz")
    mappo = np.load(test_root / "test" / "mappo-seed10" / "episode-0000.npz")
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3), sharex=True, sharey=True)
    for ax, data, title in [(axes[0], mappo, "MAPPO"), (axes[1], peva, "PEVA-Gen")]:
        uavs = data["uavs_m"]
        targets = data["targets_m"]
        for i in range(uavs.shape[1]):
            ax.plot(uavs[:, i, 0] / 1000, uavs[:, i, 1] / 1000, lw=0.8, alpha=0.75)
        for i in range(targets.shape[1]):
            ax.plot(targets[:, i, 0] / 1000, targets[:, i, 1] / 1000, "k--", lw=1.0, alpha=0.7)
            ax.scatter(targets[0, i, 0] / 1000, targets[0, i, 1] / 1000, c="black", s=12)
        ax.set_title(title)
        ax.set_xlabel("east (km)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("north (km)")
    fig.suptitle("Audited test episode trajectory; dashed lines are drifting targets")
    write_source(out, "fig6_qualitative_trajectory", ["episode", "methods", "trace_source"], [["0000", "MAPPO;PEVA-Gen", "audited NPZ traces"]])
    save(fig, out, "fig6_qualitative_trajectory")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    main_results(args.results, args.out)
    paired_improvement(args.results, args.out)
    ablation(args.results, args.out)
    robustness(args.results, args.out)
    communication(args.results, args.out)
    qualitative(args.test_root, args.out)


if __name__ == "__main__":
    main()
