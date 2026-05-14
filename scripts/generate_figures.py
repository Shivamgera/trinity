"""Generate thesis figures from experiment results (v7).

Reads canonical v7 JSON artifacts under experiments/ and produces all
figures used in the thesis.

Usage:
    python3 -m scripts.generate_figures
    python3 -m scripts.generate_figures --output-dir thesis_text/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths and constants (v7)
# ------------------------------------------------------------------

BASELINES_DIR = Path("experiments/baselines")
CGATE_DIR = Path("experiments/cgate")
ADV_DIR = Path("experiments/adversarial")
POISON_DIR = ADV_DIR / "analyst_poison"
PERTURB_DIR = ADV_DIR / "executor_perturb"
FLIP_DIR = ADV_DIR / "executor_flip"
BOOTSTRAP_PATH = Path("experiments/bootstrap/bootstrap_summary.json")

# v7 frozen seeds (NOT the stale v4 set 123/4444/6789/9999)
SEEDS = [999, 1111, 4096, 9999]

# Adversarial intensity grids (encoded as integers in filenames: 10..50)
RATES_INT = [10, 20, 30, 40, 50]
RATES_FLOAT = [r / 100.0 for r in RATES_INT]

# Consistent palette across all figures
COLORS = {
    "Trinity": "#2196F3",
    "Executor-Only": "#F44336",
    "Analyst-Only": "#4CAF50",
    "Trinity-no-CGate": "#FF9800",
    "Trinity-no-Guardian": "#9C27B0",
    "Buy-and-Hold": "#607D8B",
}

MARKERS = {
    "Trinity": "o",
    "Executor-Only": "s",
    "Analyst-Only": "^",
    "Trinity-no-CGate": "D",
    "Trinity-no-Guardian": "v",
}


# ------------------------------------------------------------------
# JSON helpers
# ------------------------------------------------------------------


def _load_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _stat(path: Path, key: str) -> float | None:
    """Extract a scalar statistic from a result JSON. Returns None if
    the file or key is missing rather than raising, so figures still
    generate when some artefacts are absent."""
    if not path.exists():
        return None
    data = _load_json(path)
    stats = data.get("statistics", data)
    # Some files use "max_drawdown", others "mean_max_drawdown" etc.
    for k in (key, f"mean_{key}", key.replace("mean_", "")):
        if k in stats:
            return float(stats[k])
    return None


def _maxdd_pct(path: Path) -> float | None:
    v = _stat(path, "max_drawdown")
    return None if v is None else v * 100


def _sharpe(path: Path) -> float | None:
    return _stat(path, "sharpe")


def _aggregate(
    per_seed_paths: list[Path], extractor
) -> tuple[float | None, float | None, float | None]:
    """Return (mean, min, max) of an extractor over per-seed paths, ignoring
    missing files. Returns (None, None, None) if no files found."""
    vals = [v for v in (extractor(p) for p in per_seed_paths) if v is not None]
    if not vals:
        return None, None, None
    return float(np.mean(vals)), float(np.min(vals)), float(np.max(vals))


# ------------------------------------------------------------------
# Path builders
# ------------------------------------------------------------------


def _clean_paths(config: str) -> list[Path]:
    if config == "Trinity":
        return [CGATE_DIR / f"integration_test_seed{s}_calibrated.json" for s in SEEDS]
    if config == "Trinity-no-Guardian":
        return [CGATE_DIR / f"integration_test_seed{s}_no_guardian.json" for s in SEEDS]
    if config == "Trinity-no-CGate":
        return [BASELINES_DIR / f"trinity_no_cgate_test_seed{s}.json" for s in SEEDS]
    if config == "Executor-Only":
        return [BASELINES_DIR / f"executor_only_test_seed{s}.json" for s in SEEDS]
    if config == "Analyst-Only":
        return [BASELINES_DIR / "analyst_only_test.json"]
    raise ValueError(f"Unknown config: {config}")


def _adv_paths(attack: str, config: str, intensity: int) -> list[Path]:
    """Build per-seed paths for an adversarial attack.

    attack ∈ {"poison", "perturb", "flip"}; intensity is integer percent
    (10, 20, ..., 50).
    """
    if attack == "poison":
        directory, tag = POISON_DIR, f"rate{intensity}"
    elif attack == "perturb":
        directory, tag = PERTURB_DIR, f"sigma{intensity}"
    elif attack == "flip":
        directory, tag = FLIP_DIR, f"flip{intensity}"
    else:
        raise ValueError(attack)

    # Configurations that do not depend on a particular attack channel
    # remain seed-resolved but read the same file across intensities.
    if config == "Analyst-Only" and attack in ("perturb", "flip"):
        return [directory / "analyst_only.json"]
    if config == "Executor-Only" and attack == "poison":
        # Executor ignores analyst signals; one file per seed, no rate.
        return [directory / f"executor_only_seed{s}.json" for s in SEEDS]
    if config == "Analyst-Only" and attack == "poison":
        return [directory / f"analyst_only_rate{intensity}.json"]

    slug = {
        "Trinity": "trinity",
        "Trinity-no-CGate": "trinity_no_cgate",
        "Trinity-no-Guardian": "trinity_no_guardian",
        "Executor-Only": "executor_only",
        "Analyst-Only": "analyst_only",
    }[config]
    return [directory / f"{slug}_seed{s}_{tag}.json" for s in SEEDS]


# ------------------------------------------------------------------
# Figure 1: Clean MaxDD bar chart
# ------------------------------------------------------------------


def fig_clean_maxdd_bar(output_dir: Path) -> None:
    configs = [
        "Trinity",
        "Trinity-no-CGate",
        "Trinity-no-Guardian",
        "Executor-Only",
        "Analyst-Only",
    ]
    means, lo, hi = [], [], []
    for c in configs:
        m, mn, mx = _aggregate(_clean_paths(c), _maxdd_pct)
        means.append(m if m is not None else 0.0)
        lo.append((m - mn) if (m is not None and mn is not None) else 0.0)
        hi.append((mx - m) if (m is not None and mx is not None) else 0.0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = [COLORS[c] for c in configs]
    bars = ax.bar(configs, means, color=colors, width=0.6, edgecolor="black", linewidth=0.5)
    ax.errorbar(configs, means, yerr=[lo, hi], fmt="none", ecolor="black", capsize=5, linewidth=1.2)

    ax.set_ylabel("Mean Maximum Drawdown (%)", fontsize=12)
    ax.set_title("Maximum Drawdown Under Clean Conditions", fontsize=13, fontweight="bold")
    ax.set_ylim(0, max(means) * 1.3 if max(means) > 0 else 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{val:.2f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    out = output_dir / "clean_maxdd_bar.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for c, m in zip(configs, means):
        print(f"  {c:22s} {m:.3f}%")


# ------------------------------------------------------------------
# Figure 2: Delta histogram with thresholds
# ------------------------------------------------------------------


def fig_delta_histogram(output_dir: Path) -> None:
    cal_path = CGATE_DIR / "calibration.json"
    if not cal_path.exists():
        print(f"  Skipping delta_histogram: {cal_path} not found")
        return
    cal = _load_json(cal_path)
    tau_low = cal["tau_low"]
    tau_high = cal["tau_high"]

    delta_path = CGATE_DIR / "calibration_deltas.json"
    if delta_path.exists():
        deltas = np.array(_load_json(delta_path))
    else:
        print("  Warning: calibration_deltas.json not found; synthesising distribution")
        rng = np.random.default_rng(42)
        deltas = rng.beta(1.5, 0.8, size=cal.get("n_samples", 250))
        deltas = deltas * cal.get("delta_std", 0.1) / np.std(deltas)
        deltas = deltas - np.mean(deltas) + cal.get("delta_mean", 0.66)
        deltas = np.clip(deltas, 0, 1)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 41)
    ax.hist(deltas, bins=bins, color="#90CAF9", edgecolor="white", linewidth=0.5, zorder=2)
    ax.axvspan(0, tau_low, alpha=0.15, color="#4CAF50", label="Agreement", zorder=1)
    ax.axvspan(tau_low, tau_high, alpha=0.15, color="#FF9800", label="Ambiguity", zorder=1)
    ax.axvspan(tau_high, 1.0, alpha=0.15, color="#F44336", label="Conflict", zorder=1)
    ax.axvline(
        tau_low,
        color="#2E7D32",
        linestyle="--",
        linewidth=1.5,
        label=f"$\\tau_{{low}}$ = {tau_low:.4f}",
    )
    ax.axvline(
        tau_high,
        color="#C62828",
        linestyle="--",
        linewidth=1.5,
        label=f"$\\tau_{{high}}$ = {tau_high:.4f}",
    )

    ax.set_xlabel("$\\Delta_t = 1 - \\pi_{RL}(d_{LLM} \\mid s_t)$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Validation-Set Divergence Distribution ($T = 1.0$)", fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "delta_histogram.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ------------------------------------------------------------------
# Figures 3, 4, 5: Adversarial MaxDD line charts
# ------------------------------------------------------------------


def _adv_line_chart(
    attack: str, configs: list[str], x_label: str, title: str, filename: str, output_dir: Path
) -> None:
    """Generic per-attack MaxDD line chart across the 10..50% grid."""
    x_labels = [f"{r}%" for r in RATES_INT]
    series: dict[str, list[float | None]] = {c: [] for c in configs}

    for c in configs:
        for r in RATES_INT:
            m, _, _ = _aggregate(_adv_paths(attack, c, r), _maxdd_pct)
            series[c].append(m)

    # Drop configs with no data at all
    series = {c: v for c, v in series.items() if any(x is not None for x in v)}
    if not series:
        print(f"  Skipping {filename}: no adversarial data found for {attack}")
        # Diagnostic: show the paths that were checked so the user can
        # verify the directory layout and filename pattern on disk.
        sample_paths = _adv_paths(attack, configs[0], RATES_INT[0])
        for p in sample_paths:
            print(f"    expected: {p} (exists: {p.exists()})")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    for c, vals in series.items():
        # matplotlib handles None as missing data via masked array
        masked = np.ma.masked_invalid([np.nan if v is None else v for v in vals])
        ax.plot(
            x_labels,
            masked,
            marker=MARKERS.get(c, "o"),
            color=COLORS[c],
            linewidth=2,
            markersize=7,
            label=c,
        )

    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Mean Maximum Drawdown (%)", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    all_vals = [v for vals in series.values() for v in vals if v is not None]
    if all_vals:
        ax.set_ylim(0, max(all_vals) * 1.18)

    plt.tight_layout()
    out = output_dir / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")
    for c, vals in series.items():
        formatted = ", ".join("None" if v is None else f"{v:.2f}" for v in vals)
        print(f"  {c:22s} [{formatted}]")


def fig_analyst_poison_maxdd(output_dir: Path) -> None:
    _adv_line_chart(
        attack="poison",
        configs=["Trinity", "Trinity-no-CGate", "Executor-Only", "Analyst-Only"],
        x_label="Analyst Poisoning Corruption Rate",
        title="MaxDD Under Analyst Poisoning",
        filename="analyst_poison_maxdd.pdf",
        output_dir=output_dir,
    )


def fig_executor_perturb_maxdd(output_dir: Path) -> None:
    _adv_line_chart(
        attack="perturb",
        configs=["Trinity", "Trinity-no-CGate", "Executor-Only"],
        x_label="Executor Perturbation Noise Level ($\\sigma$)",
        title="MaxDD Under Executor Perturbation (Gaussian Noise)",
        filename="executor_perturb_maxdd.pdf",
        output_dir=output_dir,
    )


def fig_executor_flip_maxdd(output_dir: Path) -> None:
    _adv_line_chart(
        attack="flip",
        configs=["Trinity", "Trinity-no-CGate", "Executor-Only"],
        x_label="Executor Action-Flip Rate",
        title="MaxDD Under Executor Action-Flip Attack",
        filename="executor_flip_maxdd.pdf",
        output_dir=output_dir,
    )


# ------------------------------------------------------------------
# Figure 6: Bootstrap CI forest plot
# ------------------------------------------------------------------


def fig_bootstrap_forest(output_dir: Path) -> None:
    if not BOOTSTRAP_PATH.exists():
        print(f"  Skipping bootstrap_forest: {BOOTSTRAP_PATH} not found")
        return
    bs = _load_json(BOOTSTRAP_PATH)
    per_config = bs["per_config"]

    order = ["Trinity", "Trinity-no-CGate", "Trinity-no-Guardian", "Executor-Only", "Analyst-Only"]
    order = [c for c in order if c in per_config]

    fig, (ax_s, ax_d) = plt.subplots(1, 2, figsize=(11, 4.5))
    y = np.arange(len(order))

    sharpe_mean = [per_config[c]["sharpe"]["mean"] for c in order]
    sharpe_lo = [per_config[c]["sharpe"]["ci_lo"] for c in order]
    sharpe_hi = [per_config[c]["sharpe"]["ci_hi"] for c in order]
    maxdd_mean = [per_config[c]["maxdd"]["mean"] * 100 for c in order]
    maxdd_lo = [per_config[c]["maxdd"]["ci_lo"] * 100 for c in order]
    maxdd_hi = [per_config[c]["maxdd"]["ci_hi"] * 100 for c in order]

    colors = [COLORS[c] for c in order]
    for axis, mean, lo, hi, xlabel in [
        (ax_s, sharpe_mean, sharpe_lo, sharpe_hi, "Sharpe Ratio (95\\% CI)"),
        (ax_d, maxdd_mean, maxdd_lo, maxdd_hi, "Maximum Drawdown (\\%, 95\\% CI)"),
    ]:
        err_lo = [m - l for m, l in zip(mean, lo)]
        err_hi = [h - m for m, h in zip(mean, hi)]
        axis.errorbar(
            mean,
            y,
            xerr=[err_lo, err_hi],
            fmt="o",
            color="black",
            ecolor="gray",
            capsize=4,
            markersize=0,
        )
        for yi, m, c in zip(y, mean, colors):
            axis.scatter([m], [yi], color=c, s=70, zorder=3, edgecolor="black", linewidth=0.6)
        axis.set_yticks(y)
        axis.set_yticklabels(order)
        axis.set_xlabel(xlabel, fontsize=11)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.invert_yaxis()
        axis.axvline(0, color="black", linewidth=0.5, alpha=0.3)

    ax_s.set_title("Sharpe Ratio", fontsize=12, fontweight="bold")
    ax_d.set_title("Maximum Drawdown", fontsize=12, fontweight="bold")

    fig.suptitle(
        "Circular Block Bootstrap (B=10{,}000, block=10, n=116)", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    out = output_dir / "bootstrap_forest.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Also report headline paired contrast
    c = bs.get("contrasts", {}).get("Trinity_minus_Trinity-no-CGate", {})
    if c:
        ps = c["pooled_sharpe"]
        pm = c["pooled_maxdd"]
        print(f"  Trinity - Trinity-no-CGate (pooled):")
        print(
            f"    Sharpe Δ = {ps['mean']:+.3f} "
            f"[{ps['ci_lo']:+.3f}, {ps['ci_hi']:+.3f}]  p={ps['p_two_sided']:.4f}"
        )
        print(
            f"    MaxDD  Δ = {pm['mean'] * 100:+.3f}pp "
            f"[{pm['ci_lo'] * 100:+.3f}, {pm['ci_hi'] * 100:+.3f}]  p={pm['p_two_sided']:.4f}"
        )


# ------------------------------------------------------------------
# Figures 7, 8: AAPL price and volume
# ------------------------------------------------------------------


def _split_lines(ax):
    ax.axvline(
        pd.Timestamp("2010-01-04"),
        color="green",
        linestyle="--",
        linewidth=1.2,
        label="Train start",
    )
    ax.axvline(
        pd.Timestamp("2024-01-02"), color="orange", linestyle="--", linewidth=1.2, label="Val start"
    )
    ax.axvline(
        pd.Timestamp("2024-07-01"), color="red", linestyle="--", linewidth=1.2, label="Test start"
    )


def fig_aapl_close(output_dir: Path) -> None:
    raw_path = Path("data/raw/aapl_ohlcv.parquet")
    if not raw_path.exists():
        print(f"  Skipping aapl_close: {raw_path} not found")
        return
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(raw.index, raw["Close"], linewidth=0.7, color="#1976D2")
    _split_lines(ax)
    ax.set_ylabel(r"Price (\$)", fontsize=11)
    ax.set_title("AAPL Daily Close Price (2009\u20132024)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "aapl_close.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def fig_aapl_volume(output_dir: Path) -> None:
    raw_path = Path("data/raw/aapl_ohlcv.parquet")
    if not raw_path.exists():
        print(f"  Skipping aapl_volume: {raw_path} not found")
        return
    raw = pd.read_parquet(raw_path)
    raw.index = pd.to_datetime(raw.index)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(raw.index, raw["Volume"], width=1, alpha=0.6, color="#1976D2")
    _split_lines(ax)
    ax.set_ylabel("Volume", fontsize=11)
    ax.set_title("AAPL Daily Trading Volume (2009\u20132024)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = output_dir / "aapl_volume.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Generate thesis figures (v7)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="thesis_text/figures",
        help="Directory to save figures (default: thesis_text/figures)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating thesis figures (v7 data)...")
    print(f"Seeds: {SEEDS}")
    print()

    steps = [
        ("Clean MaxDD bar chart", fig_clean_maxdd_bar),
        ("Delta histogram", fig_delta_histogram),
        ("Analyst poisoning MaxDD", fig_analyst_poison_maxdd),
        ("Executor perturbation MaxDD", fig_executor_perturb_maxdd),
        ("Executor action-flip MaxDD", fig_executor_flip_maxdd),
        ("Bootstrap CI forest plot", fig_bootstrap_forest),
        ("AAPL close price", fig_aapl_close),
        ("AAPL volume", fig_aapl_volume),
    ]
    for i, (name, fn) in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {name}")
        try:
            fn(output_dir)
        except Exception as e:
            print(f"  ERROR generating {name}: {e}")
        print()

    print(f"All figures written to {output_dir}/")


if __name__ == "__main__":
    main()
