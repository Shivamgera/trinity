#!/usr/bin/env python3
"""Generate per-seed appendix tables for the thesis.

Reads per-seed JSON outputs from experiments/baselines, experiments/cgate,
and experiments/adversarial/{analyst_poison,executor_perturb,executor_flip},
and writes a single LaTeX file containing one longtable per attack
condition, each row a seed.

Output: thesis_text/Contents/Appendix/perseed_results.tex

Run from project root on the cluster, where the canonical v7 JSONs live.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments"
OUT = ROOT / "thesis_text" / "Contents" / "Appendices" / "PerSeedResults.tex"

SEEDS = [999, 1111, 4096, 9999]
RATES = [10, 20, 30, 40, 50]


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  warn: failed to parse {path}: {e}")
        return None


def _metrics(d: dict | None) -> tuple[str, str, str]:
    """Return (Sharpe, Return%, MaxDD%) as formatted strings."""
    if d is None:
        return ("--", "--", "--")
    m = d.get("statistics") or d.get("metrics") or d
    sh = m.get("sharpe_ratio") or m.get("sharpe")
    rt = m.get("total_return") or m.get("return")
    dd = m.get("max_drawdown") or m.get("maxdd")
    if rt is not None and abs(rt) < 1.5:
        rt = rt * 100  # fraction -> percent
    if dd is not None and abs(dd) < 1.5:
        dd = dd * 100
    f = lambda x, sign=False: "--" if x is None else (f"{x:+.3f}" if sign else f"{x:.2f}")
    return (f(sh, sign=True), f(rt, sign=True), f(dd))


# ------------------------------------------------------------------
# Per-config / per-seed paths


def clean_paths(config: str, seed: int) -> Path:
    if config == "Trinity":
        return EXP / "cgate" / f"integration_test_seed{seed}_calibrated.json"
    if config == "Trinity-no-CGate":
        return EXP / "baselines" / f"trinity_no_cgate_test_seed{seed}.json"
    if config == "Trinity-no-Guardian":
        return EXP / "cgate" / f"integration_test_seed{seed}_no_guardian.json"
    if config == "Executor-Only":
        return EXP / "baselines" / f"executor_only_test_seed{seed}.json"
    raise ValueError(config)


def adv_paths(attack: str, config: str, seed: int, rate: int) -> Path:
    if attack == "poison":
        d, tag = EXP / "adversarial" / "analyst_poison", f"rate{rate}"
    elif attack == "perturb":
        d, tag = EXP / "adversarial" / "executor_perturb", f"sigma{rate}"
    elif attack == "flip":
        d, tag = EXP / "adversarial" / "executor_flip", f"flip{rate}"
    else:
        raise ValueError(attack)
    # Channel-independent configs are seed-resolved but rate-invariant.
    if config == "Executor-Only" and attack == "poison":
        return d / f"executor_only_seed{seed}.json"
    slug = {
        "Trinity": "trinity",
        "Trinity-no-CGate": "trinity_no_cgate",
        "Executor-Only": "executor_only",
    }[config]
    return d / f"{slug}_seed{seed}_{tag}.json"


# ------------------------------------------------------------------
# Table emitters


def clean_table() -> str:
    configs = ["Trinity", "Trinity-no-CGate", "Trinity-no-Guardian", "Executor-Only"]
    lines = [
        r"\begin{table}[h]\centering",
        r"\caption{Per-seed clean test-split metrics.}",
        r"\label{tab:perseed-clean}",
        r"\small\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"\textsc{Config} & \textsc{Seed} & "
        r"\textsc{Sharpe} & \textsc{Return} & \textsc{MaxDD} \\",
        r"\midrule",
    ]
    for c in configs:
        for s in SEEDS:
            sh, rt, dd = _metrics(_load(clean_paths(c, s)))
            lines.append(f"{c} & {s} & ${sh}$ & ${rt}\\%$ & {dd}\\% \\\\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def attack_table(attack: str, label_tag: str, intensity_sym: str) -> str:
    configs = ["Trinity", "Trinity-no-CGate", "Executor-Only"]
    lines = [
        r"\begin{table}[h]\centering",
        rf"\caption{{Per-seed test metrics under {label_tag}.}}",
        rf"\label{{tab:perseed-{attack}}}",
        r"\small\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"\textsc{Config} & \textsc{Seed} & "
        rf"${intensity_sym}=10\%$ & $20\%$ & $30\%$ & $40\%$ & $50\%$ \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{Sharpe ratio}} \\",
    ]
    for c in configs:
        for s in SEEDS:
            row = [c, str(s)]
            for r in RATES:
                sh, _, _ = _metrics(_load(adv_paths(attack, c, s, r)))
                row.append(f"${sh}$")
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\midrule", r"\multicolumn{7}{l}{\textit{MaxDD (\%)}} \\"]
    for c in configs:
        for s in SEEDS:
            row = [c, str(s)]
            for r in RATES:
                _, _, dd = _metrics(_load(adv_paths(attack, c, s, r)))
                row.append(dd)
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    body = [
        r"% Auto-generated by scripts/build_perseed_appendix.py.",
        r"% Do not edit by hand.",
        r"",
        r"\chapter{Per-Seed Results}",
        r"\label{app:per-seed-results}",
        r"",
        r"This appendix reports the per-seed metrics underlying the",
        r"averaged values in Chapter~\ref{cha:results}. Each table",
        r"resolves the four PPO seeds ($999$, $1111$, $4096$, $9999$)",
        r"separately for one evaluation condition.",
        r"",
        clean_table(),
        attack_table("poison", r"analyst poisoning at corruption rate $\rho$", r"\rho"),
        attack_table("perturb", r"executor perturbation at noise level $\sigma$", r"\sigma"),
        attack_table("flip", r"executor action-flipping at flip rate $\rho$", r"\rho"),
    ]
    OUT.write_text("\n".join(body))
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
