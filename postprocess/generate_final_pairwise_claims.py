"""Generate baseline-specific paired evidence for the final M6 claims.

The output is descriptive evidence for prose generation, not a model-selection
routine.  Endpoints and test streams must already be frozen.
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
BASELINES = MODELS[:7]
ENDPOINTS = MODELS[7:]
EVENT_COLUMNS = [
    "RU_event", "RD_event", "Fplus_event", "Fminus_event",
    "Sminus_event", "Splus_event",
]


def truth(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().eq("true")


def validate(table: pd.DataFrame, system: str, states: int) -> None:
    if not table["test_wind_gate_version"].eq("post_replay_test_wind_v1").all():
        raise ValueError(f"{system}: uncorrected test-wind acceptance")
    excess = table.loc[truth(table["accepted"]), "wind_curtail_excess_mw"]
    if not (np.isfinite(excess) & excess.le(1e-3)).all():
        raise ValueError(f"{system}: accepted test-wind violation")
    if len(table) != states * len(MODELS):
        raise ValueError(f"{system}: row-count mismatch")
    if set(table["model"].astype(str)) != set(MODELS):
        raise ValueError(f"{system}: model mismatch")
    if table[["state_id", "model"]].duplicated().any():
        raise ValueError(f"{system}: duplicate state/model rows")
    if not truth(table["model_called"]).all():
        raise ValueError(f"{system}: substituted row detected")
    identifiers = sorted(table["state_id"].astype(int).unique())
    if len(identifiers) != states or identifiers != list(
            range(identifiers[0], identifiers[0] + states)):
        raise ValueError(f"{system}: state coverage mismatch")


def event_vector(table: pd.DataFrame) -> np.ndarray:
    accepted = truth(table["accepted"]).to_numpy()
    audited = np.column_stack([truth(table[column]).to_numpy()
                               for column in EVENT_COLUMNS]).any(axis=1)
    return (~accepted) | audited


def paired_interval(values: np.ndarray) -> tuple[float, float, float, float]:
    mean = float(values.mean())
    if len(values) < 2:
        return mean, float("nan"), float("nan"), float("nan")
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    return mean, se, mean - 1.959963984540054 * se, mean + 1.959963984540054 * se


def reduce_system(path: Path, system: str, states: int) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    validate(table, system, states)
    by_model = {
        model: table[table["model"].eq(model)].sort_values("state_id").reset_index(drop=True)
        for model in MODELS
    }
    output: list[dict[str, object]] = []
    for endpoint in ENDPOINTS:
        endpoint_rows = by_model[endpoint]
        endpoint_shed = endpoint_rows["shed_mw"].to_numpy(float)
        endpoint_event = event_vector(endpoint_rows)
        endpoint_ok = truth(endpoint_rows["accepted"]).to_numpy()
        endpoint_time = endpoint_rows.loc[endpoint_ok, "T_on_s"].to_numpy(float)
        for baseline in BASELINES:
            baseline_rows = by_model[baseline]
            baseline_shed = baseline_rows["shed_mw"].to_numpy(float)
            baseline_event = event_vector(baseline_rows)
            baseline_ok = truth(baseline_rows["accepted"]).to_numpy()
            baseline_time = baseline_rows.loc[baseline_ok, "T_on_s"].to_numpy(float)
            shed_difference = baseline_shed - endpoint_shed
            event_difference = baseline_event.astype(float) - endpoint_event.astype(float)
            shed_mean, shed_se, shed_low, shed_high = paired_interval(shed_difference)
            event_mean, event_se, event_low, event_high = paired_interval(event_difference)
            common_ok = endpoint_ok & baseline_ok
            common_shed_difference = shed_difference[common_ok]
            (common_shed_mean, common_shed_se, common_shed_low,
             common_shed_high) = paired_interval(common_shed_difference)
            baseline_mean = float(baseline_shed.mean())
            endpoint_mean = float(endpoint_shed.mean())
            output.append({
                "system": system,
                "states": states,
                "endpoint": endpoint,
                "baseline": baseline,
                "endpoint_mean_shed_mw": endpoint_mean,
                "baseline_mean_shed_mw": baseline_mean,
                "baseline_minus_endpoint_shed_mw": shed_mean,
                "shed_difference_se_mw": shed_se,
                "shed_difference_95_low_mw": shed_low,
                "shed_difference_95_high_mw": shed_high,
                "endpoint_relative_shed_reduction": (
                    shed_mean / baseline_mean if baseline_mean != 0 else np.nan
                ),
                "endpoint_lower_shed_states": int((shed_difference > 1.0e-9).sum()),
                "equal_shed_states": int((np.abs(shed_difference) <= 1.0e-9).sum()),
                "endpoint_higher_shed_states": int((shed_difference < -1.0e-9).sum()),
                "common_accepted_states": int(common_ok.sum()),
                "common_endpoint_mean_shed_mw": float(endpoint_shed[common_ok].mean()),
                "common_baseline_mean_shed_mw": float(baseline_shed[common_ok].mean()),
                "common_baseline_minus_endpoint_shed_mw": common_shed_mean,
                "common_shed_difference_se_mw": common_shed_se,
                "common_shed_difference_95_low_mw": common_shed_low,
                "common_shed_difference_95_high_mw": common_shed_high,
                "common_endpoint_relative_shed_reduction": (
                    common_shed_mean / float(baseline_shed[common_ok].mean())
                    if common_ok.any() and float(baseline_shed[common_ok].mean()) != 0
                    else np.nan
                ),
                "common_endpoint_lower_shed_states": int(
                    (common_shed_difference > 1.0e-9).sum()
                ),
                "endpoint_safety_events": int(endpoint_event.sum()),
                "baseline_safety_events": int(baseline_event.sum()),
                "baseline_minus_endpoint_event_rate": event_mean,
                "event_rate_difference_se": event_se,
                "event_rate_difference_95_low": event_low,
                "event_rate_difference_95_high": event_high,
                "endpoint_nonaccepted": int((~endpoint_ok).sum()),
                "baseline_nonaccepted": int((~baseline_ok).sum()),
                "endpoint_median_online_s": float(np.median(endpoint_time)),
                "baseline_median_online_s": float(np.median(baseline_time)),
                "baseline_over_endpoint_median_speed_ratio": (
                    float(np.median(baseline_time) / np.median(endpoint_time))
                    if np.median(endpoint_time) > 0 else np.nan
                ),
            })
    return output


def reduce_directional(path: Path, system: str, states: int) -> list[dict[str, object]]:
    table = pd.read_csv(path)
    validate(table, system, states)
    output: list[dict[str, object]] = []
    for model in MODELS:
        rows = table[table["model"].eq(model)].sort_values("state_id")
        accepted = truth(rows["accepted"]).to_numpy()
        record: dict[str, object] = {
            "system": system,
            "states": states,
            "model": model,
            "accepted": int(accepted.sum()),
            "nonaccepted": int((~accepted).sum()),
        }
        for column in EVENT_COLUMNS:
            observed = truth(rows[column]).to_numpy()
            conservative = (~accepted) | observed
            stem = column.removesuffix("_event")
            record[f"{stem}_count"] = int(conservative.sum())
            record[f"{stem}_rate"] = float(conservative.mean())
            record[f"{stem}_accepted_count"] = int((accepted & observed).sum())
        output.append(record)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rbts-long", type=Path, required=True)
    parser.add_argument("--rts79-long", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--directional-output", type=Path, required=True)
    args = parser.parse_args()
    rows = reduce_system(args.rbts_long, "RBTS", 1000)
    rows.extend(reduce_system(args.rts79_long, "RTS79", 500))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)
    directional = reduce_directional(args.rbts_long, "RBTS", 1000)
    directional.extend(reduce_directional(args.rts79_long, "RTS79", 500))
    args.directional_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(directional).to_csv(args.directional_output, index=False)
    print(f"Final paired claims: {args.output.resolve()}")
    print(f"Final directional events: {args.directional_output.resolve()}")


if __name__ == "__main__":
    main()
