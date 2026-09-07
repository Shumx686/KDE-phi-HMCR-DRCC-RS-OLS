"""Generate paired EDNS intervals from the frozen equal-hour adequacy replays.

The reported difference is baseline minus M6, so a positive interval favors
the named M6 endpoint on EDNS.  All 500 aligned states remain in the contrast,
including conservative full-load penalties for nonaccepted solves.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


MODELS = [
    "M0", "M1", "M2", "M3", "M3-ClassMax", "M4", "M5",
    "Proposed-Tradeoff", "Proposed-Safety",
]
ENDPOINTS = ["Proposed-Tradeoff", "Proposed-Safety"]
BASELINES = ["M4", "M5"]
Z_95 = 1.959963984540054


def validate(table: pd.DataFrame, system: str, states: int) -> None:
    required = {"system", "scope", "model", "state_id", "shed_mw", "accepted"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"{system}: missing columns {sorted(missing)}")
    if len(table) != states * len(MODELS):
        raise ValueError(f"{system}: row-count mismatch")
    if set(table["system"].astype(str)) != {system}:
        raise ValueError(f"{system}: system mismatch")
    if set(table["scope"].astype(str)) != {"unconditional"}:
        raise ValueError(f"{system}: scope mismatch")
    if set(table["model"].astype(str)) != set(MODELS):
        raise ValueError(f"{system}: model mismatch")
    if table[["state_id", "model"]].duplicated().any():
        raise ValueError(f"{system}: duplicate state/model rows")
    identifiers = sorted(table["state_id"].astype(int).unique())
    if identifiers != list(range(1, states + 1)):
        raise ValueError(f"{system}: state coverage mismatch")


def interval(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, se, mean - Z_95 * se, mean + Z_95 * se


def reduce(path: Path, system: str, states: int) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    if not table["test_wind_gate_version"].eq("post_replay_test_wind_v1").all():
        raise ValueError(f"{system}: uncorrected test-wind acceptance")
    validate(table, system, states)
    by_model = {
        model: table[table["model"].eq(model)].sort_values("state_id").reset_index(drop=True)
        for model in MODELS
    }
    output: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        endpoint_rows = by_model[endpoint]
        endpoint_shed = endpoint_rows["shed_mw"].to_numpy(float)
        for baseline in BASELINES:
            baseline_rows = by_model[baseline]
            if not np.array_equal(
                    endpoint_rows["state_id"].to_numpy(int),
                    baseline_rows["state_id"].to_numpy(int)):
                raise ValueError(f"{system}: unaligned {baseline}/{endpoint} states")
            baseline_shed = baseline_rows["shed_mw"].to_numpy(float)
            mean, se, low, high = interval(baseline_shed - endpoint_shed)
            output.append({
                "system": system,
                "states": states,
                "endpoint": endpoint,
                "baseline": baseline,
                "endpoint_edns_mw": float(endpoint_shed.mean()),
                "baseline_edns_mw": float(baseline_shed.mean()),
                "baseline_minus_endpoint_edns_mw": mean,
                "paired_difference_se_mw": se,
                "paired_difference_95_low_mw": low,
                "paired_difference_95_high_mw": high,
                "interval_method": "two-sided 95% large-sample normal interval of aligned state differences",
            })
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbts-long", type=Path, required=True)
    parser.add_argument("--rts79-long", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = reduce(args.rbts_long, "RBTS", 500)
    rows.extend(reduce(args.rts79_long, "RTS79", 500))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"Final equal-hour adequacy claims: {args.output.resolve()}")


if __name__ == "__main__":
    main()
