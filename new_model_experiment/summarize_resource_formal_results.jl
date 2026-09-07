#!/usr/bin/env julia

# Auditable aggregation of one locked resource-pressure formal prefix.  Failed
# rows are already conservatively charged full load and all six safety events
# by the replay runner, so no post-hoc imputation or model-specific filtering is
# performed here.

using CSV
using DataFrames
using Dates
using Statistics
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))

const SUMMARY_MODELS = ("M0", "M1", "M2", "M3", "M4", "M5",
                        "Proposed-Tradeoff", "Proposed-Safety")
const SUMMARY_EVENTS = ("RU_event", "RD_event", "Fplus_event", "Fminus_event",
                        "Splus_event", "Sminus_event")

asbool(value) = lowercase(string(value)) == "true"

function wilson_interval(events::Int, total::Int; z::Float64 = 1.959963984540054)
    total > 0 || return (NaN, NaN)
    p = events / total
    denominator = 1 + z^2 / total
    center = (p + z^2 / (2total)) / denominator
    half = z * sqrt(p * (1 - p) / total + z^2 / (4total^2)) / denominator
    return max(0.0, center - half), min(1.0, center + half)
end

function finite_column(rows, name::String)
    values = Float64[]
    for row in eachrow(rows)
        value = row[name]
        ismissing(value) && continue
        parsed = value isa Number ? Float64(value) : tryparse(Float64, string(value))
        isnothing(parsed) || isfinite(parsed) || continue
        push!(values, parsed)
    end
    return values
end

function main(result_dir::String, lock_path::String, draw_path::String,
              prefix::Int, output_dir::String)
    lock_hash = sha256_file(lock_path)
    draw_hash = sha256_file(draw_path)
    pattern = Regex("_N$(prefix)_shard[0-9]+of[0-9]+_[A-Za-z0-9_-]+\\.csv\$")
    files = sort(filter(path -> occursin(pattern, basename(path)),
                        readdir(result_dir; join = true)))
    isempty(files) && error("no formal replay CSVs found for prefix N=$prefix")
    table = vcat((CSV.read(file, DataFrame) for file in files)...; cols = :union)
    required = ["model", "state_id", "accepted", "shed_mw", "LOLP_event",
                "model_called", "T_on_s", "T_e2e_s", "formal_lock_sha256",
                "draw_sha256", collect(SUMMARY_EVENTS)...]
    all(name -> name in names(table), required) || error("formal row schema is incomplete")
    all(string(value) == lock_hash for value in table.formal_lock_sha256) ||
        error("formal rows do not share the requested lock hash")
    all(string(value) == draw_hash for value in table.draw_sha256) ||
        error("formal rows do not share the requested draw hash")
    nrow(table) == prefix * length(SUMMARY_MODELS) || error("formal row count mismatch")
    nrow(unique(table[:, [:state_id, :model]])) == nrow(table) ||
        error("duplicate model-state formal rows")
    Set(String.(table.model)) == Set(SUMMARY_MODELS) || error("formal model set mismatch")
    Set(Int.(table.state_id)) == Set(1:prefix) || error("formal state prefix mismatch")

    summaries = Dict{String,Any}[]
    for model in SUMMARY_MODELS
        rows = table[String.(table.model) .== model, :]
        accepted = asbool.(rows.accepted)
        called = asbool.(rows.model_called)
        lolp = asbool.(rows.LOLP_event)
        any_safety = [any(asbool(row[event]) for event in SUMMARY_EVENTS)
                      for row in eachrow(rows)]
        event_count = count(identity, any_safety)
        low, high = wilson_interval(event_count, nrow(rows))
        shed = Float64.(rows.shed_mw)
        conditional = shed[lolp]
        called_accepted = rows[called .& accepted, :]
        online = finite_column(called_accepted, "T_on_s")
        e2e = finite_column(called_accepted, "T_e2e_s")
        standard_error = length(shed) > 1 ? std(shed; corrected = true) / sqrt(length(shed)) : NaN
        summary = Dict{String,Any}(
            "model" => model, "states" => nrow(rows),
            "accepted" => count(identity, accepted),
            "acceptance_rate" => mean(accepted),
            "model_calls" => count(identity, called),
            "edns_mw" => mean(shed), "edns_standard_error_mw" => standard_error,
            "edns_ci95_low_mw" => mean(shed) - 1.959963984540054 * standard_error,
            "edns_ci95_high_mw" => mean(shed) + 1.959963984540054 * standard_error,
            "edns_cov" => mean(shed) > 0 ? standard_error / mean(shed) : 0.0,
            "lolp" => mean(lolp),
            "conditional_shed_mw" => isempty(conditional) ? 0.0 : mean(conditional),
            "any_safety_event_count" => event_count,
            "any_safety_event_frequency" => event_count / nrow(rows),
            "any_safety_wilson95_low" => low,
            "any_safety_wilson95_high" => high,
            "mean_online_called_accepted_s" => isempty(online) ? NaN : mean(online),
            "median_online_called_accepted_s" => isempty(online) ? NaN : median(online),
            "mean_e2e_called_accepted_s" => isempty(e2e) ? NaN : mean(e2e),
            "mean_online_all_states_s" => mean(coalesce.(rows.T_on_s, 0.0)),
        )
        for event in SUMMARY_EVENTS
            summary[replace(event, "_event" => "_frequency")] =
                mean(asbool.(rows[!, event]))
        end
        push!(summaries, summary)
    end
    summary_table = DataFrame(summaries)
    sort!(summary_table, :model)
    mkpath(output_dir)
    summary_path = joinpath(output_dir, "resource_formal_summary_N$(prefix).csv")
    write_csv_atomic(summary_path,
        [Dict{String,Any}(String(name) => row[name] for name in names(summary_table))
         for row in eachrow(summary_table)])
    manifest = Dict{String,Any}(
        "status" => "resource_formal_summary_complete",
        "created_utc" => string(now(UTC)), "prefix" => prefix,
        "failure_policy" => "conservative_full_shed",
        "formal_lock_path" => abspath(lock_path), "formal_lock_sha256" => lock_hash,
        "draw_path" => abspath(draw_path), "draw_sha256" => draw_hash,
        "input_files" => abspath.(files),
        "input_sha256" => Dict(abspath(file) => sha256_file(file) for file in files),
        "summary_path" => abspath(summary_path),
        "summary_sha256" => sha256_file(summary_path))
    manifest_path = joinpath(output_dir, "resource_formal_summary_N$(prefix)_manifest.toml")
    write_toml_atomic(manifest_path, manifest)
    println("Formal summary: ", summary_path)
    show(stdout, MIME("text/plain"), summary_table)
    println()
end

length(ARGS) == 5 || error(
    "usage: summarize_resource_formal_results.jl RESULT_DIR LOCK.toml DRAWS.csv PREFIX OUTPUT_DIR")
main(normpath(ARGS[1]), normpath(ARGS[2]), normpath(ARGS[3]), parse(Int, ARGS[4]),
     normpath(ARGS[5]))
