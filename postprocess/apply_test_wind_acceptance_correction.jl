#!/usr/bin/env julia
# Deterministic post-replay correction. Never calls a solver or overwrites inputs.
using CSV, DataFrames, Dates, SHA, TOML
include(joinpath(@__DIR__, "TestWindAcceptance.jl"))
using .TestWindAcceptance
hash_file(path) = bytes2hex(open(SHA.sha256, path))

function main(input, manifest_path, load_path, output_dir)
    ispath(output_dir) && error("use a new correction directory; existing outputs are not overwritten")
    manifest = TOML.parsefile(manifest_path)
    manifest["status"] == "complete" || error("incomplete source replay")
    # Frozen manifests retain the original host path. Relocation is allowed,
    # while filename, schema, state coverage, and model coverage remain checked.
    basename(normpath(manifest["long_file"])) == basename(normpath(input)) ||
        error("input/manifest filename mismatch")
    inputs = [abspath(input), abspath(manifest_path), abspath(load_path)]
    input_hashes = hash_file.(inputs)
    rows = CSV.read(input, DataFrame)
    expected = Int(manifest["state_count"])
    models = String.(manifest["models"])
    nrow(rows) == expected * length(models) || error("row count mismatch")
    length(models) == 9 && Set(String.(rows.model)) == Set(models) || error("model coverage mismatch")
    all(TestWindAcceptance.truth, rows.model_called) || error("substituted model call")
    allunique(zip(rows.state_id, rows.model)) || error("duplicate model-state rows")
    Set(Int.(rows.state_id)) == Set(Int(manifest["first_state"]):Int(manifest["last_state"])) ||
        error("state coverage mismatch")
    loads = CSV.read(load_path, DataFrame)
    load_values = Float64.(Matrix(loads[:, 2:end]))
    all(isfinite, load_values) && all(>=(0.0), load_values) || error("invalid load source")
    full_loads = vec(sum(load_values; dims = 2))
    updated = Dict{String,Any}[]
    changes = DataFrame(system = String[], scope = String[], model = String[],
        state_id = Int[], source_hour = Int[], wind_curtail_excess_mw = Float64[],
        old_shed_mw = Float64[], penalty_shed_mw = Float64[])
    prior_failure_checks = 0
    for source in eachrow(rows)
        row = Dict{String,Any}(string(key) => source[key] for key in propertynames(source))
        hour = Int(row["source_hour"])
        1 <= hour <= length(full_loads) || error("invalid chronological load index")
        full_load = full_loads[hour]
        if !TestWindAcceptance.truth(row["accepted"])
            isapprox(Float64(row["shed_mw"]), full_load; atol = 1e-8, rtol = 1e-12) ||
                error("existing full-load penalty disagrees with source load")
            prior_failure_checks += 1
        end
        apply_test_wind_gate!(row, full_load)
        if row["test_wind_rejected"]
            push!(changes, (manifest["system"], manifest["scope"], string(row["model"]),
                Int(row["state_id"]), hour, Float64(row["wind_curtail_excess_mw"]),
                Float64(row["pre_test_wind_gate_shed_mw"]), full_load))
        end
        push!(updated, row)
    end
    mkpath(joinpath(output_dir, "corrected_inputs"))
    corrected_path = joinpath(output_dir, "corrected_inputs", basename(input))
    changes_path = joinpath(output_dir, "acceptance_changes.csv")
    CSV.write(corrected_path, DataFrame(updated))
    CSV.write(changes_path, changes)
    hash_file.(inputs) == input_hashes || error("source input changed during correction")
    receipt = Dict("status" => "complete_post_replay_correction",
        "created_utc" => string(now(UTC)), "test_wind_gate_version" => TEST_WIND_GATE_VERSION,
        "test_wind_audit_tol_mw" => TEST_WIND_TOL_MW,
        "system" => manifest["system"], "scope" => manifest["scope"],
        "state_count" => expected, "row_count" => nrow(rows),
        "newly_nonaccepted" => nrow(changes), "prior_full_load_checks" => prior_failure_checks,
        "input_files" => inputs, "input_sha256" => input_hashes,
        "corrected_file" => abspath(corrected_path), "corrected_sha256" => hash_file(corrected_path),
        "change_log" => abspath(changes_path), "change_log_sha256" => hash_file(changes_path),
        "gate_script_sha256" => hash_file(joinpath(@__DIR__, "TestWindAcceptance.jl")),
        "correction_script_sha256" => hash_file(@__FILE__),
        "criterion_history" => "Existing 1 kW audit tolerance; rejection enforcement corrected after replay, not a pretest gate.",
        "failure_policy" => "Existing full-realized-load/all-six-event accounting; no state removed.",
        "new_solver_calls" => 0, "parameters_or_draws_changed" => false)
    open(joinpath(output_dir, "correction_manifest.toml"), "w") do io
        TOML.print(io, receipt; sorted = true)
    end
    println(manifest["system"], "/", manifest["scope"], ": ", nrow(changes),
            " newly nonaccepted; ", prior_failure_checks, " prior full-load penalties verified")
end

length(ARGS) == 4 || error("usage: INPUT_LONG INPUT_MANIFEST BUS_LOADS NEW_OUTPUT_DIR")
main(abspath.(ARGS)...)
