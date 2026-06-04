"""Block bootstrap analysis of test-window daily returns for the Robust Trinity.

Produces 95% confidence intervals for Sharpe and MaxDD per configuration, and
a paired bootstrap test of the Trinity vs Trinity-no-CGate contrast on matched
test days. Resampling is performed over time (circular block bootstrap of
Politis & Romano, 1994), so the procedure does not depend on the seed budget
and is appropriate for short evaluation windows with serial dependence.

Usage (run on the cluster, where v7 per-day result JSONs live):

    python scripts/bootstrap_analysis.py \\
        --results-root experiments \\
        --seeds 999 1111 4096 9999 \\
        --block-length 10 \\
        --n-bootstrap 10000 \\
        --output experiments/bootstrap/bootstrap_summary.json

Input file schema (matches scripts/run_baselines.py and
scripts/cgate_integration.py output):

    {
      "statistics": {...},
      "results": [
        {"date": "...", "step_return": <float>, "portfolio_value": <float>, ...},
        ...
      ]
    }

Configurations discovered automatically by globbing:

    Trinity (full):        experiments/cgate/integration_test_seed{seed}_calibrated.json
    Trinity-no-CGate:      experiments/baselines/trinity_no_cgate_test_seed{seed}.json
    Trinity-no-Guardian:   experiments/cgate/integration_test_seed{seed}_no_guardian.json
    Executor-Only:         experiments/baselines/executor_only_test_seed{seed}.json
    Analyst-Only:          experiments/baselines/analyst_only_test.json (seed-independent)

Path templates are CLI-overridable. Only stdlib + numpy required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence

import numpy as np

# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------

ANNUALISATION = np.sqrt(252.0)


def sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe with ddof=0 (population SD), matching the thesis."""
    sigma = returns.std(ddof=0)
    if sigma < 1e-12:
        return 0.0
    return float(returns.mean() / sigma * ANNUALISATION)


def maxdd(returns: np.ndarray) -> float:
    """Maximum drawdown of the equity curve implied by `returns`.

    Returns are arithmetic per-step. Equity starts at 1.0.
    """
    equity = np.cumprod(1.0 + returns)
    peaks = np.maximum.accumulate(equity)
    drawdowns = 1.0 - equity / peaks
    return float(drawdowns.max())


# ----------------------------------------------------------------------------
# Block bootstrap
# ----------------------------------------------------------------------------


def circular_block_bootstrap_indices(
    n: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Return n indices drawn by circular block bootstrap.

    Politis & Romano (1994). The series is wrapped circularly so every starting
    index is equally likely, removing the boundary bias of the basic
    moving-block bootstrap. Blocks are concatenated until length n is reached.
    """
    if block_length < 1:
        raise ValueError("block_length must be >= 1")
    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    idx = (starts[:, None] + np.arange(block_length)[None, :]) % n
    return idx.reshape(-1)[:n]


def bootstrap_metric(
    returns: np.ndarray,
    metric_fn,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return a (n_bootstrap,) array of metric values on circular-block resamples."""
    n = len(returns)
    out = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = circular_block_bootstrap_indices(n, block_length, rng)
        out[b] = metric_fn(returns[idx])
    return out


def paired_bootstrap_contrast(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    metric_fn,
    block_length: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return (n_bootstrap,) array of metric(a) - metric(b) on shared resamples.

    Indices are drawn once per bootstrap replicate and applied to BOTH series,
    preserving the day-by-day pairing. Sharpe and MaxDD are non-linear, so
    bootstrapping the daily difference would be wrong --- we resample the
    matched series, recompute each metric, then take the difference.
    """
    if len(returns_a) != len(returns_b):
        raise ValueError(f"Series length mismatch: {len(returns_a)} vs {len(returns_b)}")
    n = len(returns_a)
    out = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = circular_block_bootstrap_indices(n, block_length, rng)
        out[b] = metric_fn(returns_a[idx]) - metric_fn(returns_b[idx])
    return out


def ci_and_pvalue(samples: np.ndarray, alpha: float = 0.05) -> dict:
    """Return percentile CI and a two-sided p-value via CI inversion.

    p = 2 * min(P(samples <= 0), P(samples >= 0)), bounded to [1/B, 1].
    """
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    p_below = float((samples <= 0).mean())
    p_above = float((samples >= 0).mean())
    p_two_sided = max(2.0 * min(p_below, p_above), 1.0 / len(samples))
    return {
        "mean": float(samples.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "p_two_sided": p_two_sided,
    }


# ----------------------------------------------------------------------------
# I/O
# ----------------------------------------------------------------------------


def load_returns(path: Path) -> np.ndarray:
    with path.open() as f:
        data = json.load(f)
    results = data.get("results", [])
    if not results:
        raise ValueError(f"No 'results' in {path}")
    return np.array([r["step_return"] for r in results], dtype=float)


def load_dates(path: Path) -> list[str]:
    with path.open() as f:
        data = json.load(f)
    return [r["date"] for r in data.get("results", [])]


# ----------------------------------------------------------------------------
# Config discovery
# ----------------------------------------------------------------------------


@dataclass
class ConfigSpec:
    name: str
    template: str  # contains {seed} placeholder, or none for seedless configs
    seedless: bool = False


DEFAULT_CONFIGS: list[ConfigSpec] = [
    ConfigSpec(
        "Trinity",
        "{root}/cgate/integration_test_seed{seed}_calibrated.json",
    ),
    ConfigSpec(
        "Trinity-no-CGate",
        "{root}/baselines/trinity_no_cgate_test_seed{seed}.json",
    ),
    ConfigSpec(
        "Trinity-no-Guardian",
        "{root}/cgate/integration_test_seed{seed}_no_guardian.json",
    ),
    ConfigSpec(
        "Executor-Only",
        "{root}/baselines/executor_only_test_seed{seed}.json",
    ),
    ConfigSpec(
        "Analyst-Only",
        "{root}/baselines/analyst_only_test.json",
        seedless=True,
    ),
]


def resolve_paths(spec: ConfigSpec, root: str, seeds: Sequence[int]) -> dict[int | None, Path]:
    out: dict[int | None, Path] = {}
    if spec.seedless:
        p = Path(spec.template.format(root=root))
        if p.exists():
            out[None] = p
        return out
    for s in seeds:
        p = Path(spec.template.format(root=root, seed=s))
        if p.exists():
            out[s] = p
    return out


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results-root", default="experiments", type=str)
    ap.add_argument("--seeds", nargs="+", type=int, default=[999, 1111, 4096, 9999])
    ap.add_argument("--block-length", type=int, default=10)
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=42, help="Bootstrap RNG seed")
    ap.add_argument(
        "--output",
        default="experiments/bootstrap/bootstrap_summary.json",
        type=str,
    )
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ------------------------------------------------------------------
    # 1. Load all available per-config, per-seed return series
    # ------------------------------------------------------------------
    series: dict[str, dict[int | None, np.ndarray]] = {}
    dates_ref: list[str] | None = None
    for spec in DEFAULT_CONFIGS:
        paths = resolve_paths(spec, args.results_root, args.seeds)
        if not paths:
            print(f"[warn] no files found for {spec.name}", file=sys.stderr)
            continue
        series[spec.name] = {}
        for s, p in paths.items():
            r = load_returns(p)
            series[spec.name][s] = r
            if dates_ref is None:
                dates_ref = load_dates(p)
            print(
                f"[ok] {spec.name:22s} seed={s} n={len(r):4d} from {p.name}",
                file=sys.stderr,
            )

    if not series:
        print("[error] no result files found. Check --results-root and --seeds.")
        return 2

    n_days = len(dates_ref) if dates_ref else None
    print(f"[info] test window: {n_days} days", file=sys.stderr)

    # ------------------------------------------------------------------
    # 2. Per-config, per-seed bootstrap CIs for Sharpe and MaxDD
    # ------------------------------------------------------------------
    summary: dict = {
        "meta": {
            "block_length": args.block_length,
            "n_bootstrap": args.n_bootstrap,
            "rng_seed": args.seed,
            "n_test_days": n_days,
            "seeds": args.seeds,
        },
        "per_seed": {},
        "per_config": {},
        "contrasts": {},
    }

    for cfg, per_seed in series.items():
        summary["per_seed"][cfg] = {}
        # Pool bootstrap replicates across seeds for a config-level CI:
        # for each replicate b, average the metric across seeds (matched
        # day indices not required here, since each seed has its own series).
        sharpe_reps_per_seed: list[np.ndarray] = []
        maxdd_reps_per_seed: list[np.ndarray] = []
        for s, r in per_seed.items():
            sh = bootstrap_metric(r, sharpe, args.block_length, args.n_bootstrap, rng)
            md = bootstrap_metric(r, maxdd, args.block_length, args.n_bootstrap, rng)
            sharpe_reps_per_seed.append(sh)
            maxdd_reps_per_seed.append(md)
            summary["per_seed"][cfg][str(s)] = {
                "sharpe_point": sharpe(r),
                "maxdd_point": maxdd(r),
                "sharpe": {
                    "mean": float(sh.mean()),
                    "ci_lo": float(np.percentile(sh, 2.5)),
                    "ci_hi": float(np.percentile(sh, 97.5)),
                },
                "maxdd": {
                    "mean": float(md.mean()),
                    "ci_lo": float(np.percentile(md, 2.5)),
                    "ci_hi": float(np.percentile(md, 97.5)),
                },
            }
        # Config-level: average across seeds within each bootstrap replicate.
        sharpe_cfg = np.mean(np.stack(sharpe_reps_per_seed), axis=0)
        maxdd_cfg = np.mean(np.stack(maxdd_reps_per_seed), axis=0)
        summary["per_config"][cfg] = {
            "n_seeds": len(per_seed),
            "sharpe": {
                "mean": float(sharpe_cfg.mean()),
                "ci_lo": float(np.percentile(sharpe_cfg, 2.5)),
                "ci_hi": float(np.percentile(sharpe_cfg, 97.5)),
            },
            "maxdd": {
                "mean": float(maxdd_cfg.mean()),
                "ci_lo": float(np.percentile(maxdd_cfg, 2.5)),
                "ci_hi": float(np.percentile(maxdd_cfg, 97.5)),
            },
        }

    # ------------------------------------------------------------------
    # 3. Paired contrast: Trinity vs Trinity-no-CGate, per seed and pooled
    # ------------------------------------------------------------------
    if "Trinity" in series and "Trinity-no-CGate" in series:
        shared_seeds = sorted(
            set(series["Trinity"].keys()) & set(series["Trinity-no-CGate"].keys())
        )
        contrast = {"per_seed_sharpe": {}, "per_seed_maxdd": {}}
        sharpe_diffs_per_seed: list[np.ndarray] = []
        maxdd_diffs_per_seed: list[np.ndarray] = []
        for s in shared_seeds:
            ra = series["Trinity"][s]
            rb = series["Trinity-no-CGate"][s]
            sh_diff = paired_bootstrap_contrast(
                ra, rb, sharpe, args.block_length, args.n_bootstrap, rng
            )
            md_diff = paired_bootstrap_contrast(
                ra, rb, maxdd, args.block_length, args.n_bootstrap, rng
            )
            sharpe_diffs_per_seed.append(sh_diff)
            maxdd_diffs_per_seed.append(md_diff)
            contrast["per_seed_sharpe"][str(s)] = ci_and_pvalue(sh_diff)
            contrast["per_seed_maxdd"][str(s)] = ci_and_pvalue(md_diff)
        if sharpe_diffs_per_seed:
            pooled_sh = np.mean(np.stack(sharpe_diffs_per_seed), axis=0)
            pooled_md = np.mean(np.stack(maxdd_diffs_per_seed), axis=0)
            contrast["pooled_sharpe"] = ci_and_pvalue(pooled_sh)
            contrast["pooled_maxdd"] = ci_and_pvalue(pooled_md)
        summary["contrasts"]["Trinity_minus_Trinity-no-CGate"] = contrast

    # ------------------------------------------------------------------
    # 3b. Paired contrast: Trinity vs Executor-Only, per seed and pooled
    # ------------------------------------------------------------------
    if "Trinity" in series and "Executor-Only" in series:
        shared_seeds = sorted(
            set(series["Trinity"].keys()) & set(series["Executor-Only"].keys())
        )
        contrast_ex = {"per_seed_sharpe": {}, "per_seed_maxdd": {}}
        sharpe_diffs_ex: list[np.ndarray] = []
        maxdd_diffs_ex: list[np.ndarray] = []
        for s in shared_seeds:
            ra = series["Trinity"][s]
            rb = series["Executor-Only"][s]
            sh_diff = paired_bootstrap_contrast(
                ra, rb, sharpe, args.block_length, args.n_bootstrap, rng
            )
            md_diff = paired_bootstrap_contrast(
                ra, rb, maxdd, args.block_length, args.n_bootstrap, rng
            )
            sharpe_diffs_ex.append(sh_diff)
            maxdd_diffs_ex.append(md_diff)
            contrast_ex["per_seed_sharpe"][str(s)] = ci_and_pvalue(sh_diff)
            contrast_ex["per_seed_maxdd"][str(s)] = ci_and_pvalue(md_diff)
        if sharpe_diffs_ex:
            pooled_sh = np.mean(np.stack(sharpe_diffs_ex), axis=0)
            pooled_md = np.mean(np.stack(maxdd_diffs_ex), axis=0)
            contrast_ex["pooled_sharpe"] = ci_and_pvalue(pooled_sh)
            contrast_ex["pooled_maxdd"] = ci_and_pvalue(pooled_md)
        summary["contrasts"]["Trinity_minus_Executor-Only"] = contrast_ex

    # ------------------------------------------------------------------
    # 4. Write summary
    # ------------------------------------------------------------------
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[ok] wrote {out_path}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 5. Human-readable Markdown summary on stdout
    # ------------------------------------------------------------------
    print("\n## Per-configuration bootstrap (pooled across seeds)\n")
    print("| Config | Sharpe mean | Sharpe 95% CI | MaxDD mean | MaxDD 95% CI |")
    print("|---|---|---|---|---|")
    for cfg, d in summary["per_config"].items():
        s = d["sharpe"]
        m = d["maxdd"]
        print(
            f"| {cfg} "
            f"| {s['mean']:+.3f} "
            f"| [{s['ci_lo']:+.3f}, {s['ci_hi']:+.3f}] "
            f"| {m['mean'] * 100:.2f}% "
            f"| [{m['ci_lo'] * 100:.2f}%, {m['ci_hi'] * 100:.2f}%] |"
        )

    if "Trinity_minus_Trinity-no-CGate" in summary["contrasts"]:
        c = summary["contrasts"]["Trinity_minus_Trinity-no-CGate"]
        print("\n## Paired contrast: Trinity \u2212 Trinity-no-CGate\n")
        if "pooled_sharpe" in c:
            ps = c["pooled_sharpe"]
            pm = c["pooled_maxdd"]
            print(
                f"Pooled Sharpe diff: {ps['mean']:+.3f} "
                f"95% CI [{ps['ci_lo']:+.3f}, {ps['ci_hi']:+.3f}], "
                f"p = {ps['p_two_sided']:.4f}"
            )
            print(
                f"Pooled MaxDD diff:  {pm['mean'] * 100:+.2f}pp "
                f"95% CI [{pm['ci_lo'] * 100:+.2f}pp, {pm['ci_hi'] * 100:+.2f}pp], "
                f"p = {pm['p_two_sided']:.4f}"
            )

    if "Trinity_minus_Executor-Only" in summary["contrasts"]:
        c = summary["contrasts"]["Trinity_minus_Executor-Only"]
        print("\n## Paired contrast: Trinity \u2212 Executor-Only\n")
        if "pooled_sharpe" in c:
            ps = c["pooled_sharpe"]
            pm = c["pooled_maxdd"]
            print(
                f"Pooled Sharpe diff: {ps['mean']:+.3f} "
                f"95% CI [{ps['ci_lo']:+.3f}, {ps['ci_hi']:+.3f}], "
                f"p = {ps['p_two_sided']:.4f}"
            )
            print(
                f"Pooled MaxDD diff:  {pm['mean'] * 100:+.2f}pp "
                f"95% CI [{pm['ci_lo'] * 100:+.2f}pp, {pm['ci_hi'] * 100:+.2f}pp], "
                f"p = {pm['p_two_sided']:.4f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
