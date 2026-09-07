"""Generate the three final numerical figures from fail-closed replay reductions.

The architecture diagram remains Fig. 1 in the TeX source.  This script emits
the dynamic shedding--safety--speed map, the equal-hour adequacy figure,
and the paired M1/M6--Safety tail figure.  It never contains hard-coded results.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tomllib

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd


MODELS = [
    "M0", "M1", "M2", "M3", "M3-ClassMax", "M4", "M5",
    "Proposed-Tradeoff", "Proposed-Safety",
]
LABELS = {
    "M0": "M0", "M1": "M1", "M2": "M2", "M3": "M3",
    "M3-ClassMax": "M3-CM", "M4": "M4", "M5": "M5",
    "Proposed-Tradeoff": "M6-T", "Proposed-Safety": "M6-S",
}
EVENTS = [
    "RU_event", "RD_event", "Fplus_event", "Fminus_event",
    "Sminus_event", "Splus_event",
]


def truth(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def read_summary(path: Path, system: str, scope: str, states: int) -> pd.DataFrame:
    manifest_path = path.with_name(path.name.replace("_summary.csv", "_manifest.toml"))
    if not manifest_path.is_file():
        raise ValueError(f"missing summary manifest: {manifest_path}")
    with manifest_path.open("rb") as stream:
        manifest = tomllib.load(stream)
    if manifest.get("test_wind_gate_version") != "post_replay_test_wind_v1":
        raise ValueError(f"final figures require corrected test-wind acceptance: {path}")
    if manifest.get("summary_file_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError(f"summary hash mismatch: {path}")
    if manifest.get("status") != "complete":
        raise ValueError(f"incomplete summary manifest: {manifest_path}")
    if manifest.get("system") != system or manifest.get("scope") != scope:
        raise ValueError(f"manifest scope mismatch: {manifest_path}")
    if int(manifest.get("state_count", -1)) != states:
        raise ValueError(f"manifest state-count mismatch: {manifest_path}")
    if manifest.get("models") != MODELS:
        raise ValueError(f"manifest model mismatch: {manifest_path}")
    if scope == "stress":
        if not manifest.get("all_rows_model_called", False):
            raise ValueError(f"stress substitution detected: {manifest_path}")
    else:
        if manifest.get("policy_mode") != "common_normal_dispatch_plus_emergency_activation":
            raise ValueError(f"unverified equal-hour adequacy operating policy: {manifest_path}")
        if not manifest.get("raw_full_domain_all_models_called", False):
            raise ValueError(f"missing all-model applicability audit: {manifest_path}")
        if not manifest.get("equal_hour_weights", False):
            raise ValueError(f"nonuniform equal-hour weights: {manifest_path}")
        covered = (int(manifest.get("activation_state_count", -1)) +
                   int(manifest.get("outside_activation_state_count", -1)))
        if covered != states:
            raise ValueError(f"activation-state accounting mismatch: {manifest_path}")
    table = pd.read_csv(path)
    if table["model"].tolist() != MODELS:
        raise ValueError(f"model order mismatch: {path}")
    if set(table["system"].astype(str)) != {system}:
        raise ValueError(f"system mismatch: {path}")
    if set(table["scope"].astype(str)) != {scope}:
        raise ValueError(f"scope mismatch: {path}")
    if set(table["n_states"].astype(int)) != {states}:
        raise ValueError(f"state-count mismatch: {path}")
    return table.set_index("model", drop=False)


def read_long(path: Path, states: int) -> pd.DataFrame:
    table = pd.read_csv(path)
    if not table["test_wind_gate_version"].eq("post_replay_test_wind_v1").all():
        raise ValueError(f"uncorrected long table: {path}")
    accepted_excess = table.loc[truth(table["accepted"]), "wind_curtail_excess_mw"]
    if not (np.isfinite(accepted_excess) & accepted_excess.le(1e-3)).all():
        raise ValueError(f"accepted test-wind violation: {path}")
    if len(table) != states * len(MODELS):
        raise ValueError(f"long-table row-count mismatch: {path}")
    if set(table["model"].astype(str)) != set(MODELS):
        raise ValueError(f"long-table model mismatch: {path}")
    if table[["state_id", "model"]].duplicated().any():
        raise ValueError(f"duplicate state/model row: {path}")
    if not truth(table["model_called"]).all():
        raise ValueError(f"precheck-substituted row detected: {path}")
    return table


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.titlesize": 9.0,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.3,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig: mpl.figure.Figure, output_dir: Path, stem: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(output_dir / f"{stem}.{suffix}", dpi=450,
                    bbox_inches="tight", pad_inches=0.025, transparent=True)
    plt.close(fig)


def plot_operating_map(summaries: list[tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    all_times = np.concatenate([
        table.loc[MODELS, "median_online_s"].to_numpy(float) for _, table in summaries
    ])
    positive_times = all_times[np.isfinite(all_times) & (all_times > 0)]
    norm = LogNorm(vmin=positive_times.min(), vmax=positive_times.max())
    cmap = mpl.colormaps["viridis"]
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), constrained_layout=True)
    offset_sets = [
        [(6, 5), (-19, 7), (-4, 21), (19, 5), (4, 34),
         (6, 5), (-6, 6), (-7, 9), (9, 8)],
        [(6, 5), (16, 25), (-14, 8), (45, 30), (-7, 37),
         (6, 5), (-6, 6), (-8, 17), (18, 8)],
    ]

    for panel, (ax, (title, table)) in enumerate(zip(axes, summaries)):
        x = table.loc[MODELS, "mean_shed_mw"].to_numpy(float)
        y = 100.0 * table.loc[MODELS, "safety_event_rate"].to_numpy(float)
        times = table.loc[MODELS, "median_online_s"].to_numpy(float)
        baseline_count = 7
        ax.scatter(x[:baseline_count], y[:baseline_count], c=times[:baseline_count],
                   cmap=cmap, norm=norm, marker="o", s=48, edgecolors="black",
                   linewidths=0.45, zorder=3)
        ax.scatter([x[7]], [y[7]], c=[times[7]], cmap=cmap, norm=norm,
                   marker="D", s=72, edgecolors="#0b3c5d", linewidths=1.1, zorder=5)
        ax.scatter([x[8]], [y[8]], c=[times[8]], cmap=cmap, norm=norm,
                   marker="*", s=120, edgecolors="#8b1a1a", linewidths=1.0, zorder=6)
        ax.annotate("", xy=(x[8], y[8]), xytext=(x[7], y[7]),
                    arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 0.9,
                                "shrinkA": 7, "shrinkB": 8}, zorder=2)
        for model, xi, yi, offset in zip(MODELS, x, y, offset_sets[panel]):
            ax.annotate(LABELS[model], (xi, yi), xytext=offset,
                        textcoords="offset points",
                        ha="left" if offset[0] >= 0 else "right",
                        va="bottom",
                        fontsize=7.1,
                        fontweight="bold" if model.startswith("Proposed") else "normal",
                        arrowprops={"arrowstyle": "-", "color": "0.50", "lw": 0.45,
                                    "shrinkA": 1.5, "shrinkB": 3.5}
                        if abs(offset[0]) + abs(offset[1]) >= 22 else None,
                        zorder=7)
        ax.set_title(title, pad=4)
        ax.set_xlabel("Stress-conditional mean shedding (MW)")
        ax.set_ylabel("Conservative six-direction event frequency (%)")
        ax.set_ylim(-3.0, max(1.0, y.max() * 1.16))
        ax.grid(True, color="0.88", linewidth=0.55, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    legend_handles = [
        mpl.lines.Line2D([], [], marker="o", linestyle="", markerfacecolor="0.65",
                         markeredgecolor="black", markersize=5.5, label="M0-M5 and M3-CM"),
        mpl.lines.Line2D([], [], marker="D", linestyle="", markerfacecolor="0.65",
                         markeredgecolor="#0b3c5d", markersize=6.0, label="M6-Tradeoff"),
        mpl.lines.Line2D([], [], marker="*", linestyle="", markerfacecolor="0.65",
                         markeredgecolor="#8b1a1a", markersize=8.5, label="M6-Safety"),
    ]
    axes[0].legend(handles=legend_handles, loc="upper right", framealpha=0.95,
                   borderpad=0.35, handletextpad=0.4)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(scalar, ax=axes, location="right", pad=0.015, fraction=0.035)
    cbar.set_label("Median accepted online time (s, log scale)")
    save(fig, output_dir, "FIG2_SHEDDING_SAFETY_SPEED")


def plot_reliability(summaries: list[tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.15, 3.95), constrained_layout=True,
                             sharex="col")
    colors = ["#909090", "#6f6f6f", "#4676a9", "#315a87", "#78a6cc",
              "#b58b4a", "#9a6540", "#227c74", "#b43c39"]
    labels = [LABELS[m] for m in MODELS]
    positions = np.arange(len(MODELS))
    for column, (title, table) in enumerate(summaries):
        edns = table.loc[MODELS, "mean_shed_mw"].to_numpy(float)
        edns_error = 1.959963984540054 * table.loc[MODELS, "shed_se_mw"].to_numpy(float)
        lolp = 100.0 * table.loc[MODELS, "lolp"].to_numpy(float)
        lolp_low = 100.0 * table.loc[MODELS, "lolp_wilson95_low"].to_numpy(float)
        lolp_high = 100.0 * table.loc[MODELS, "lolp_wilson95_high"].to_numpy(float)
        error_style = {"elinewidth": 0.65, "ecolor": "0.18", "capsize": 1.8,
                       "capthick": 0.65}
        axes[0, column].bar(positions, edns, color=colors, edgecolor="0.2", linewidth=0.35,
                            yerr=edns_error, error_kw=error_style, zorder=2)
        axes[1, column].bar(positions, lolp, color=colors, edgecolor="0.2", linewidth=0.35,
                            yerr=np.vstack((lolp - lolp_low, lolp_high - lolp)),
                            error_kw=error_style, zorder=2)
        axes[0, column].set_title(title)
        axes[0, column].set_ylabel("EDNS (MW)")
        axes[1, column].set_ylabel("LOLP (%)")
        axes[1, column].set_xticks(positions, labels, rotation=45, ha="right")
        for row in range(2):
            axes[row, column].grid(axis="y", color="0.88", linewidth=0.5, zorder=0)
            axes[row, column].spines[["top", "right"]].set_visible(False)
    save(fig, output_dir, "FIG3_CHRONOLOGICAL_RELIABILITY")


def empirical_tail(long_table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    m1 = long_table[long_table["model"].eq("M1")].set_index("state_id")
    safety = long_table[long_table["model"].eq("Proposed-Safety")].set_index("state_id")
    m1_ok = truth(m1["accepted"])
    safety_ok = truth(safety["accepted"])
    common_ids = m1.index[m1_ok & safety_ok]
    m1_values = np.maximum(0.0, m1.loc[common_ids, "max_safety_excess_mw"].to_numpy(float))
    safety_values = np.maximum(0.0, safety.loc[common_ids, "max_safety_excess_mw"].to_numpy(float))
    maximum = max(float(m1_values.max(initial=0.0)), float(safety_values.max(initial=0.0)), 1e-5)
    thresholds = np.concatenate(([0.0], np.logspace(-6, np.log10(maximum * 1.05), 240)))
    return (thresholds,
            100.0 * np.array([(m1_values > value).mean() for value in thresholds]),
            100.0 * np.array([(safety_values > value).mean() for value in thresholds]),
            len(common_ids), int((~m1_ok).sum()), int((~safety_ok).sum()))


def plot_tail(longs: list[tuple[str, pd.DataFrame]], output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.55), constrained_layout=True)
    for ax, (title, table) in zip(axes, longs):
        thresholds, m1_tail, safety_tail, common_n, m1_fail, safety_fail = empirical_tail(table)
        ax.step(thresholds, m1_tail, where="post", color="#3f6f9f", lw=1.35, label="M1")
        ax.step(thresholds, safety_tail, where="post", color="#b43c39", lw=1.35,
                label="M6-Safety")
        ax.set_xscale("symlog", linthresh=1e-6, linscale=0.5)
        ax.set_yscale("log")
        positive = np.concatenate((m1_tail[m1_tail > 0], safety_tail[safety_tail > 0]))
        lower = max(0.05, positive.min() * 0.75) if positive.size else 0.05
        ax.set_ylim(lower, 120.0)
        ax.set_title(title)
        ax.set_xlabel("Accepted maximum six-direction excess threshold (MW)")
        ax.set_ylabel("Empirical exceedance probability (%)")
        ax.grid(True, which="both", color="0.88", linewidth=0.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.98, 0.96, f"common accepted N={common_n}\nnonaccepted M1/M6-S={m1_fail}/{safety_fail}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.0,
                bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
                      "edgecolor": "0.75", "alpha": 0.92})
    axes[0].legend(loc="lower left", framealpha=0.95)
    save(fig, output_dir, "FIG4_M1_SAFETY_TAIL")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbts-stress-summary", type=Path, required=True)
    parser.add_argument("--rbts-stress-long", type=Path, required=True)
    parser.add_argument("--rts79-stress-summary", type=Path, required=True)
    parser.add_argument("--rts79-stress-long", type=Path, required=True)
    parser.add_argument("--rbts-unconditional-summary", type=Path, required=True)
    parser.add_argument("--rts79-unconditional-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rbts_stress = read_summary(args.rbts_stress_summary, "RBTS", "stress", 1000)
    rts_stress = read_summary(args.rts79_stress_summary, "RTS79", "stress", 500)
    rbts_unconditional = read_summary(args.rbts_unconditional_summary, "RBTS", "unconditional", 500)
    rts_unconditional = read_summary(args.rts79_unconditional_summary, "RTS79", "unconditional", 500)
    rbts_long = read_long(args.rbts_stress_long, 1000)
    rts_long = read_long(args.rts79_stress_long, 500)

    plot_operating_map([("RBTS ($N=1000$)", rbts_stress),
                        ("RTS79 ($N=500$)", rts_stress)], args.output_dir)
    plot_reliability([("RBTS equal-hour adequacy test", rbts_unconditional),
                      ("RTS79 equal-hour adequacy test", rts_unconditional)], args.output_dir)
    plot_tail([("RBTS stress test", rbts_long),
               ("RTS79 stress test", rts_long)], args.output_dir)
    print(f"Final numerical figures: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
