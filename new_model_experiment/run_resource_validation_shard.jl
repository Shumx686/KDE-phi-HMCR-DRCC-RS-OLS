#!/usr/bin/env julia

# Resumable validation-only common-state shard for a frozen resource candidate.
# State positions are interleaved across shards so all workers sample the same
# chronological validation prefix without overlapping model-state pairs.

using CSV
using DataFrames
using Dates
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))

const BASE_VALIDATION_MODELS = ("M0", "M1", "M2", "M4", "M5", "Proposed-RS",
                           "M3T005", "M3T010", "M3T020", "M3T050", "M3T100",
                           "Proposed-Tradeoff", "Proposed-Tradeoff75",
                           "Proposed-TradeoffW1", "Proposed-Tradeoff75W1",
                           "Proposed-Safety", "Proposed-Safety10",
                           "Proposed-Safety100", "Proposed-SafetyC0",
                           "Proposed-SafetyC25", "Proposed-SafetyC75",
                           "Proposed-SafetyC100", "Proposed-SafetyRU125",
                           "Proposed-SafetyRU150", "Proposed-SafetyRU200",
                           "Proposed-SafetyRU300", "Proposed-SafetyRU500",
                           "Proposed-SafetyRU1000", "Proposed-SafetyBW125",
                           "Proposed-SafetyBW150", "Proposed-SafetyBW200",
                           "Proposed-SafetyBW225", "Proposed-SafetyBW250",
                           "Proposed-SafetyBW255", "Proposed-SafetyBW260",
                           "Proposed-SafetyBW265", "Proposed-SafetyBW270",
                           "Proposed-SafetyBW272", "Proposed-SafetyBW274",
                           "Proposed-SafetyBW2745", "Proposed-SafetyBW2747",
                           "Proposed-SafetyBW2748", "Proposed-SafetyBW2749",
                           "Proposed-SafetyBW275", "Proposed-SafetyBW290",
                           "Proposed-SafetyBW300", "Proposed-TradeoffBW150",
                           "Proposed-TradeoffBW200", "Proposed-TradeoffBW225",
                           "Proposed-SafetyB275C000", "Proposed-SafetyB275C025",
                           "Proposed-SafetyB275C050", "Proposed-SafetyB275C075",
                           "Proposed-SafetyB275C080", "Proposed-SafetyB275C085",
                           "Proposed-SafetyB275C090", "Proposed-SafetyB275C095",
                           "Proposed-SafetyB275C098", "Proposed-SafetyB275C099")
const RHO_GRID_TAGS = ("000", "025", "050", "075", "100")
const RHO_GRID_MODELS = Tuple("Proposed-GC$(cost)S$(safety)"
                              for cost in RHO_GRID_TAGS for safety in RHO_GRID_TAGS)
const RHO_FINE_SAFETY_TAGS = ("055", "060", "065", "070", "080", "085", "090", "095")
const RHO_FINE_MODELS = Tuple("Proposed-GC000S$(safety)"
                              for safety in RHO_FINE_SAFETY_TAGS)
const VALIDATION_MODELS = (BASE_VALIDATION_MODELS..., RHO_GRID_MODELS...,
                           RHO_FINE_MODELS...)

function proposed_endpoint(model::String, ctx)
    grid_match = match(r"^Proposed-GC(\d{3})S(\d{3})$", model)
    if !isnothing(grid_match)
        rho_c = parse(Float64, grid_match.captures[1]) / 100.0
        rho_s = parse(Float64, grid_match.captures[2]) / 100.0
        return (rho_tau_c = rho_c, rho_tau_s = rho_s, omega = ones(6))
    elseif startswith(model, "Proposed-SafetyB275C")
        tag = replace(model, "Proposed-SafetyB275C" => "")
        rho_c = parse(Float64, tag) / 100.0
        return (rho_tau_c = rho_c, rho_tau_s = 1.0, omega = ones(6),
                bandwidth_multiplier = 2.75)
    elseif startswith(model, "Proposed-TradeoffBW")
        tag = replace(model, "Proposed-TradeoffBW" => "")
        bw = parse(Float64, tag) / 100.0
        return (rho_tau_c = 0.5, rho_tau_s = 0.5, omega = ones(6),
                bandwidth_multiplier = bw)
    elseif startswith(model, "Proposed-SafetyBW")
        tag = replace(model, "Proposed-SafetyBW" => "")
        bw = parse(Float64, tag) / (length(tag) == 4 ? 1000.0 : 100.0)
        return (rho_tau_c = 1.0, rho_tau_s = 1.0, omega = ones(6),
                bandwidth_multiplier = bw)
    elseif startswith(model, "Proposed-SafetyRU")
        tag = replace(model, "Proposed-SafetyRU" => "")
        ru_weight = parse(Float64, tag) / 100.0
        return (rho_tau_c = 0.5, rho_tau_s = 1.0,
                omega = [ru_weight, 1.0, 1.0, 1.0, 1.0, 1.0])
    elseif model == "Proposed-SafetyC0"
        return (rho_tau_c = 0.0, rho_tau_s = 1.0, omega = ones(6))
    elseif model == "Proposed-SafetyC25"
        return (rho_tau_c = 0.25, rho_tau_s = 1.0, omega = ones(6))
    elseif model == "Proposed-SafetyC75"
        return (rho_tau_c = 0.75, rho_tau_s = 1.0, omega = ones(6))
    elseif model == "Proposed-SafetyC100"
        return (rho_tau_c = 1.0, rho_tau_s = 1.0, omega = ones(6))
    elseif model == "Proposed-Safety100"
        return (rho_tau_c = 0.5, rho_tau_s = 1.0, omega = fill(100.0, 6))
    elseif model == "Proposed-Safety10"
        return (rho_tau_c = 0.5, rho_tau_s = 1.0, omega = fill(10.0, 6))
    elseif model == "Proposed-Safety"
        return (rho_tau_c = 0.5, rho_tau_s = 1.0, omega = ones(6))
    elseif model == "Proposed-Tradeoff75"
        return (rho_tau_c = 0.5, rho_tau_s = 0.75, omega = fill(0.3, 6))
    elseif model == "Proposed-TradeoffW1"
        return (rho_tau_c = 0.5, rho_tau_s = 0.5, omega = ones(6))
    elseif model == "Proposed-Tradeoff75W1"
        return (rho_tau_c = 0.5, rho_tau_s = 0.75, omega = ones(6))
    elseif model == "Proposed-Tradeoff"
        return (rho_tau_c = 0.5, rho_tau_s = 0.5, omega = fill(0.3, 6))
    end
    return (rho_tau_c = ctx.rho_tau_c, rho_tau_s = ctx.rho_tau_s,
            omega = ctx.omega)
end

function prior_rows(path::String)
    isfile(path) || return Dict{String,Any}[]
    table = CSV.read(path, DataFrame)
    return [Dict{String,Any}(String(name) => row[name] for name in names(table))
            for row in eachrow(table)]
end

row_accepted(row::AbstractDict) =
    haskey(row, "accepted") && lowercase(string(row["accepted"])) == "true"

function main(protocol_path::String, candidate_requested::String,
              total_states::Int, shard_index::Int, shard_count::Int,
              models::Vector{String}, output_dir::String)
    total_states > 0 || error("total_states must be positive")
    1 <= shard_index <= shard_count || error("shard index must lie in 1:shard_count")
    !isempty(models) || error("at least one model is required")
    all(model in VALIDATION_MODELS for model in models) ||
        error("validation shard contains an unsupported model")

    protocol, screen = protocol_inputs(protocol_path)
    declared = Dict(candidate_key(c) => c for c in declared_candidates(screen))
    haskey(declared, candidate_requested) || error("candidate is not declared")
    candidate = declared[candidate_requested]
    ctx = load_candidate(protocol_path, candidate, total_states)
    all_ids = frozen_indices(ctx.pipe.split.val, total_states)
    isempty(intersect(Set(all_ids), Set(ctx.pipe.split.test))) ||
        error("validation shard touched a test state")
    shard_ids = [snapshot for (position, snapshot) in enumerate(all_ids)
                 if mod(position - 1, shard_count) == shard_index - 1]

    Pbase = baseline_parameters(protocol, screen)
    selected_margin, support_audit = m4_support_margin_audit(ctx, all_ids, screen)
    Pbase["support_margin"] = selected_margin

    mkpath(output_dir)
    model_tag = join(replace.(models, "-" => "_"), "-")
    stem = "validation_$(candidate_requested)_$(model_tag)_shard$(shard_index)of$(shard_count)"
    long_path = joinpath(output_dir, stem * ".csv")
    rows = prior_rows(long_path)
    # Accepted rows are immutable resume checkpoints.  Failed rows remain
    # retryable and are replaced in place so a resumed shard cannot create
    # duplicate model-state pairs.
    processed = Set((Int(row["snapshot"]), String(row["model"])) for row in rows
                    if row_accepted(row))

    for snapshot in shard_ids
        snap = snapshot_at(ctx.panel, snapshot, ctx.cd)
        for model in models
            (snapshot, model) in processed && continue
            started = time()
            row = try
                if startswith(model, "Proposed-")
                    endpoint = proposed_endpoint(model, ctx)
                    run = solve_final_snapshot(ctx, snapshot;
                        rho_tau_c = endpoint.rho_tau_c,
                        rho_tau_s = endpoint.rho_tau_s,
                        omega = endpoint.omega,
                        bandwidth_multiplier = hasproperty(endpoint, :bandwidth_multiplier) ?
                            endpoint.bandwidth_multiplier :
                            Float64(ctx.protocol["model"]["bandwidth_multiplier"]))
                    proposed_row = solved_proposed_row(candidate, snapshot, ctx, snap,
                                                       run, time() - started)
                    proposed_row["model"] = model
                    proposed_row
                else
                    P = deepcopy(Pbase)
                    profile = nothing
                    dispatch_model = model
                    if startswith(model, "M3T")
                        tag = replace(model, "M3T" => "")
                        P["theta_m"] = parse(Float64, tag) / 1000.0
                        P["kde_mode"] = "fixed"
                        profile = kde_profile(ctx, snap, P)
                        dispatch_model = "M3"
                    elseif model == "M2"
                        P["kde_mode"] = "fixed"
                        profile = kde_profile(ctx, snap, P)
                    elseif model == "M4"
                        P["wasserstein_radius_multiplier"] =
                            Float64(screen["m4_radius_multiplier_anchor"])
                    end
                    u = Driver._solve_one(dispatch_model, ctx.cd, snap, ctx.scen, P, 0.0;
                                          bandwidth_profile = profile)
                    solved_baseline_row(candidate, model, snapshot, ctx, snap, u, P,
                                        time() - started)
                end
            catch err
                failed_row(candidate, model, snapshot, snap, sprint(showerror, err),
                           time() - started, ctx.wind_meta)
            end
            previous = findfirst(existing ->
                Int(existing["snapshot"]) == snapshot &&
                String(existing["model"]) == model, rows)
            if isnothing(previous)
                push!(rows, row)
            else
                rows[previous] = row
            end
            push!(processed, (snapshot, model))
            write_csv_atomic(long_path, rows)
            println(candidate_requested, ",", model, ",", snapshot, ",", row["status"])
            flush(stdout)
        end
        GC.gc()
    end

    summaries = [summarize_model(rows, candidate, model, length(shard_ids))
                 for model in models]
    summary_path = joinpath(output_dir, stem * "_summary.csv")
    write_csv_atomic(summary_path, summaries)
    receipt = Dict{String,Any}(
        "status" => "validation_shard_complete",
        "validation_only" => true,
        "formal_test_unlocked" => false,
        "test_access" => "forbidden",
        "protocol_path" => abspath(protocol_path),
        "protocol_sha256" => sha256_file(protocol_path),
        "candidate" => candidate,
        "candidate_key" => candidate_requested,
        "total_validation_prefix" => total_states,
        "shard_index" => shard_index,
        "shard_count" => shard_count,
        "validation_state_ids" => shard_ids,
        "models" => models,
        "m4_support_margin" => selected_margin,
        "m4_support_margin_audit" => support_audit,
        "long_file" => abspath(long_path),
        "long_file_sha256" => sha256_file(long_path),
        "summary_file" => abspath(summary_path),
        "summary_file_sha256" => sha256_file(summary_path),
    )
    receipt_path = joinpath(output_dir, stem * "_receipt.toml")
    write_toml_atomic(receipt_path, receipt)
    println("Validation shard receipt: $receipt_path")
    return receipt_path
end

if abspath(PROGRAM_FILE) == @__FILE__
    length(ARGS) == 7 || error(
        "usage: run_resource_validation_shard.jl protocol.toml CANDIDATE TOTAL_STATES SHARD_INDEX SHARD_COUNT MODELS_CSV OUTPUT_DIR")
    main(normpath(ARGS[1]), String(ARGS[2]), parse(Int, ARGS[3]),
         parse(Int, ARGS[4]), parse(Int, ARGS[5]), String.(split(ARGS[6], ',')),
         normpath(ARGS[7]))
end
