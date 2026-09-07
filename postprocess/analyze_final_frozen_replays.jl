#!/usr/bin/env julia

# Final, scope-frozen reduction for the RBTS/RTS79 common-state replays.
# It refuses precheck-substituted rows and keeps stress-conditioned metrics
# distinct from unconditional reliability metrics.

using CSV
using DataFrames
using Statistics
using TOML
using SHA
include(joinpath(@__DIR__, "TestWindAcceptance.jl"))
using .TestWindAcceptance

const MODELS = ["M0", "M1", "M2", "M3", "M3-ClassMax", "M4", "M5",
                "Proposed-Tradeoff", "Proposed-Safety"]
const EVENT_COLUMNS = [:RU_event, :RD_event, :Fplus_event, :Fminus_event,
                       :Sminus_event, :Splus_event]

length(ARGS) == 6 || error(
    "usage: analyze_final_frozen_replays.jl SYSTEM SCOPE INPUT_DIR FIRST_STATE LAST_STATE OUTPUT_DIR")

asbool(x) = x isa Bool ? x : lowercase(strip(string(x))) == "true"

function numeric(x; default = NaN)
    ismissing(x) && return default
    x isa Number && return Float64(x)
    value = tryparse(Float64, strip(string(x)))
    return isnothing(value) ? default : value
end

function finite_quantile(values, q)
    isempty(values) && return NaN
    return quantile(sort(values), q)
end

function wilson_interval(successes::Integer, trials::Integer; z = 1.959963984540054)
    trials > 0 || return (NaN, NaN)
    p = successes / trials
    denominator = 1 + z^2 / trials
    center = (p + z^2 / (2trials)) / denominator
    radius = z * sqrt(p * (1 - p) / trials + z^2 / (4trials^2)) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))
end

function main(system::String, scope::String, input_dir::String,
              first_state::Int, last_state::Int, output_dir::String)
    scope in ("stress", "unconditional") ||
        error("scope must be stress or unconditional")
    isdir(input_dir) || error("input directory does not exist")
    files = sort(filter(path -> endswith(lowercase(path), ".csv"),
                        readdir(input_dir; join = true)))
    isempty(files) && error("no replay CSV files found")
    tables = [CSV.read(path, DataFrame) for path in files]
    rows = reduce((a, b) -> vcat(a, b; cols = :union), tables)

    required = vcat([:model, :state_id, :accepted, :model_called, :shed_mw,
                     :T_on_s, :max_safety_excess_mw, :curtail_mw,
                     :wind_curtail_excess_mw, :wind_curtail_event], EVENT_COLUMNS)
    missing_columns = setdiff(required, Symbol.(names(rows)))
    isempty(missing_columns) || error("missing columns: $(join(missing_columns, ','))")

    rows.model = String.(rows.model)
    rows.state_id = Int.(rows.state_id)
    filter!(row -> first_state <= row.state_id <= last_state && row.model in MODELS, rows)
    n_expected = last_state - first_state + 1
    nrow(rows) == n_expected * length(MODELS) ||
        error("expected $(n_expected * length(MODELS)) rows, found $(nrow(rows))")
    all(asbool.(rows.model_called)) || error("precheck-substituted row detected")
    allunique(zip(rows.state_id, rows.model)) || error("duplicate model-state row detected")
    Set(rows.state_id) == Set(first_state:last_state) || error("state coverage mismatch")
    Set(rows.model) == Set(MODELS) || error("model coverage mismatch")
    sort!(rows, [:state_id, :model])
    :test_wind_gate_version in propertynames(rows) ||
        error("apply_test_wind_acceptance_correction.jl must precede final reduction")
    all(==(TEST_WIND_GATE_VERSION), rows.test_wind_gate_version) || error("gate version mismatch")
    all(==(TEST_WIND_TOL_MW), rows.test_wind_audit_tol_mw) || error("gate tolerance mismatch")
    for row in eachrow(rows)
        if asbool(row.accepted)
            excess = numeric(row.wind_curtail_excess_mw)
            isfinite(excess) && excess <= TEST_WIND_TOL_MW ||
                error("accepted row violates test-wind availability")
            !asbool(row.wind_curtail_event) || error("accepted wind-event flag")
        else
            isapprox(numeric(row.shed_mw), numeric(row.realized_full_load_mw);
                     atol = 1e-8, rtol = 1e-12) || error("nonaccepted full-load penalty mismatch")
        end
    end

    summaries = NamedTuple[]
    for model in MODELS
        g = rows[rows.model .== model, :]
        accepted = asbool.(g.accepted)
        shed = numeric.(g.shed_mw; default = Inf)
        all(isfinite, shed) || error("nonfinite conservative shedding for $model")
        lolp_event = shed .> 1.0e-3
        safety_event = [!accepted[i] || any(asbool(g[i, column]) for column in EVENT_COLUMNS)
                        for i in 1:nrow(g)]
        accepted_safety_event = [accepted[i] &&
                                 any(asbool(g[i, column]) for column in EVENT_COLUMNS)
                                 for i in 1:nrow(g)]
        excess = [accepted[i] ? numeric(g.max_safety_excess_mw[i]; default = Inf) : Inf
                  for i in 1:nrow(g)]
        accepted_excess = [numeric(g.max_safety_excess_mw[i]; default = Inf)
                           for i in 1:nrow(g) if accepted[i]]
        severe = excess .> 1.0
        accepted_severe = accepted_excess .> 1.0
        accepted_time = [numeric(g.T_on_s[i]) for i in 1:nrow(g)
                         if accepted[i] && isfinite(numeric(g.T_on_s[i]))]
        wind_excess = numeric.(g.wind_curtail_excess_mw; default = Inf)
        wind_event = [!accepted[i] || asbool(g.wind_curtail_event[i]) for i in 1:nrow(g)]
        accepted_wind_excess = [numeric(g.wind_curtail_excess_mw[i]; default = Inf)
                                for i in 1:nrow(g) if accepted[i]]
        accepted_wind_event = [accepted[i] && asbool(g.wind_curtail_event[i])
                               for i in 1:nrow(g)]
        curtail = max.(0.0, numeric.(g.curtail_mw; default = Inf))
        all(isfinite, curtail) || error("nonfinite curtailment for $model")
        all(isfinite, accepted_excess) || error("nonfinite accepted safety excess for $model")
        all(isfinite, accepted_wind_excess) ||
            error("nonfinite accepted wind-curtailment excess for $model")
        shed_sd = nrow(g) > 1 ? std(shed; corrected = true) : NaN
        shed_se = nrow(g) > 1 ? shed_sd / sqrt(nrow(g)) : NaN
        shed_cov = mean(shed) == 0.0 ? NaN : shed_se / mean(shed)
        lolp_ci = wilson_interval(count(lolp_event), nrow(g))
        safety_ci = wilson_interval(count(safety_event), nrow(g))
        accepted_safety_ci = wilson_interval(count(accepted_safety_event), count(accepted))
        eue = scope == "unconditional" ? mean(shed) * 8760.0 : missing
        push!(summaries, (
            system = system,
            scope = scope,
            model = model,
            n_states = nrow(g),
            accepted = count(accepted),
            nonaccepted = count(!, accepted),
            test_wind_rejected_count = count(asbool, g.test_wind_rejected),
            shed_metric = scope == "stress" ? "SCMS" : "EDNS",
            mean_shed_mw = mean(shed),
            shed_sd_mw = shed_sd,
            shed_se_mw = shed_se,
            shed_cov = shed_cov,
            lolp_metric = scope == "stress" ? "LOLP_stress" : "LOLP",
            lolp = mean(lolp_event),
            lolp_wilson95_low = lolp_ci[1],
            lolp_wilson95_high = lolp_ci[2],
            eue_mwh_per_year = eue,
            safety_event_count = count(safety_event),
            safety_event_rate = mean(safety_event),
            safety_event_wilson95_low = safety_ci[1],
            safety_event_wilson95_high = safety_ci[2],
            accepted_safety_event_count = count(accepted_safety_event),
            accepted_safety_event_rate = count(accepted) == 0 ? NaN :
                count(accepted_safety_event) / count(accepted),
            accepted_safety_event_wilson95_low = accepted_safety_ci[1],
            accepted_safety_event_wilson95_high = accepted_safety_ci[2],
            max_excess_mw = maximum(excess),
            p95_excess_mw = finite_quantile(excess, 0.95),
            p99_excess_mw = finite_quantile(excess, 0.99),
            severe_excess_count = count(severe),
            accepted_max_excess_mw = isempty(accepted_excess) ? NaN : maximum(accepted_excess),
            accepted_p95_excess_mw = finite_quantile(accepted_excess, 0.95),
            accepted_p99_excess_mw = finite_quantile(accepted_excess, 0.99),
            accepted_severe_excess_count = count(accepted_severe),
            median_online_s = isempty(accepted_time) ? NaN : median(accepted_time),
            mean_curtail_mw = mean(curtail),
            max_wind_curtail_excess_mw = maximum(wind_excess),
            wind_curtail_event_count = count(wind_event),
            accepted_max_wind_curtail_excess_mw = isempty(accepted_wind_excess) ? NaN :
                maximum(accepted_wind_excess),
            accepted_wind_curtail_event_count = count(accepted_wind_event),
        ))
    end

    mkpath(output_dir)
    prefix = "final_$(lowercase(system))_$(scope)"
    long_path = joinpath(output_dir, prefix * "_long.csv")
    summary_path = joinpath(output_dir, prefix * "_summary.csv")
    CSV.write(long_path, rows)
    CSV.write(summary_path, DataFrame(summaries))
    manifest = Dict{String,Any}(
        "status" => "complete",
        "system" => system,
        "scope" => scope,
        "first_state" => first_state,
        "last_state" => last_state,
        "state_count" => n_expected,
        "models" => MODELS,
        "all_rows_model_called" => true,
        "input_files" => abspath.(files),
        "input_sha256" => [bytes2hex(open(SHA.sha256, path)) for path in files],
        "test_wind_gate_version" => TEST_WIND_GATE_VERSION,
        "test_wind_audit_tol_mw" => TEST_WIND_TOL_MW,
        "new_solver_calls" => 0,
        "long_file" => abspath(long_path),
        "long_file_sha256" => bytes2hex(open(SHA.sha256, long_path)),
        "summary_file" => abspath(summary_path),
        "summary_file_sha256" => bytes2hex(open(SHA.sha256, summary_path)),
        "eue_rule" => scope == "unconditional" ?
            "EDNS multiplied by 8760 for uniformly sampled chronological test hours" :
            "not reported for stress-conditioned sampling",
    )
    open(joinpath(output_dir, prefix * "_manifest.toml"), "w") do io
        TOML.print(io, manifest; sorted = true)
    end
    println("Final summary: ", summary_path)
end

main(String(ARGS[1]), String(ARGS[2]), normpath(ARGS[3]), parse(Int, ARGS[4]),
     parse(Int, ARGS[5]), normpath(ARGS[6]))
