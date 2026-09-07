module TestWindAcceptance

# Post-replay physical acceptance, not a change to frozen optimization models.
export apply_test_wind_gate!, TEST_WIND_GATE_VERSION, TEST_WIND_TOL_MW
const TEST_WIND_GATE_VERSION = "post_replay_test_wind_v1"
const TEST_WIND_TOL_MW = 1.0e-3
const DIRECTIONS = ("RU", "RD", "Fplus", "Fminus", "Sminus", "Splus")
const CORRECTED_FIELDS = vcat(
    ["accepted", "failure_reason", "shed_mw", "curtail_mw", "LOLP_event",
     "max_safety_excess_mw", "wind_curtail_event", "sample_safe_curtailment_margin_mw"],
    ["$(direction)_$(suffix)" for direction in DIRECTIONS
     for suffix in ("event", "excess_mw")])
truth(x) = x isa Bool ? x : lowercase(strip(string(x))) == "true"
number(x) = x isa Real ? Float64(x) : something(tryparse(Float64, string(x)), NaN)

function apply_test_wind_gate!(row::AbstractDict, full_load::Real)
    load = Float64(full_load)
    isfinite(load) && load >= 0 || error("invalid realized full load")
    if haskey(row, "test_wind_gate_version")
        row["test_wind_gate_version"] == TEST_WIND_GATE_VERSION || error("gate version mismatch")
        number(row["test_wind_audit_tol_mw"]) == TEST_WIND_TOL_MW || error("gate tolerance mismatch")
        number(row["realized_full_load_mw"]) == load || error("full-load mismatch")
        if truth(row["test_wind_rejected"])
            !truth(row["accepted"]) && number(row["shed_mw"]) == load ||
                error("corrupted wind-rejection penalty")
            all(truth(row["$(direction)_event"]) for direction in DIRECTIONS) ||
                error("corrupted wind-rejection event penalty")
        end
        return row
    end
    accepted = truth(row["accepted"])
    excess = number(get(row, "wind_curtail_excess_mw", missing))
    rejected = accepted && (!isfinite(excess) || excess > TEST_WIND_TOL_MW)
    for field in CORRECTED_FIELDS
        row["pre_test_wind_gate_" * field] = get(row, field, missing)
    end
    row["test_wind_gate_version"] = TEST_WIND_GATE_VERSION
    row["test_wind_audit_tol_mw"] = TEST_WIND_TOL_MW
    row["realized_full_load_mw"] = load
    row["solver_accepted_before_test_wind_gate"] = accepted
    row["test_wind_rejected"] = rejected
    if rejected
        row["accepted"] = false
        reason = get(row, "failure_reason", "")
        reason = ismissing(reason) ? "" : string(reason)
        row["failure_reason"] = isempty(reason) ? "test_wind_availability_excess" :
            reason * ";test_wind_availability_excess"
        row["shed_mw"] = load
        row["curtail_mw"] = 0.0
        row["LOLP_event"] = load > TEST_WIND_TOL_MW
        row["max_safety_excess_mw"] = Inf
        row["wind_curtail_event"] = true
        if haskey(row, "sample_safe_curtailment_cap_mw")
            row["sample_safe_curtailment_margin_mw"] = row["sample_safe_curtailment_cap_mw"]
        end
        for direction in DIRECTIONS
            row["$(direction)_event"] = true
            row["$(direction)_excess_mw"] = Inf
        end
    end
    # Keep the original solver status, timing, and observed wind excess intact.
    return row
end
end
