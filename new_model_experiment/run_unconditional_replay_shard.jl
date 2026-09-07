#!/usr/bin/env julia

# All-model replay on a uniformly sampled chronological test-hour stream.
# The locked stress-trained policy and representative scenarios are preserved;
# only the forecast/realization snapshot population changes.  Every row is an
# actual model call and nonaccepted solves receive the same conservative rule.

using CSV
using DataFrames
using SHA
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))
using .DRCCExp: Cases, CaseInterface, DataPipeline, Driver

const UNCONDITIONAL_MODELS = ("M0", "M1", "M2", "M3", "M3-ClassMax",
                              "M4", "M5", "Proposed-Tradeoff", "Proposed-Safety")

length(ARGS) in (9, 10) || error(
    "usage: run_unconditional_replay_shard.jl SYSTEM PROTOCOL LOCK DRAWS SHARD_INDEX SHARD_COUNT MODELS_CSV OUTPUT_DIR RUN_TAG [SEED_DIR|-]")

hash_file(path::String) = bytes2hex(open(SHA.sha256, path))
branch_online(rc::Cases.RawCase) = rc.branch[:, 11] .> 0

function parse_down_indices(value, n::Int)
    down = falses(n)
    raw = ismissing(value) ? "" : strip(string(value))
    isempty(raw) && return down
    for item in split(raw, ';')
        isempty(item) && continue
        index = parse(Int, item)
        1 <= index <= n || error("component index out of range")
        down[index] = true
    end
    return down
end

function connected_after_line_outages(rc::Cases.RawCase, down)
    buses = Int.(rc.bus[:, 1]); pos = Dict(buses[i] => i for i in eachindex(buses))
    adjacent = [Int[] for _ in buses]
    for line in findall(branch_online(rc) .& .!down)
        a, b = pos[Int(rc.branch[line, 1])], pos[Int(rc.branch[line, 2])]
        push!(adjacent[a], b); push!(adjacent[b], a)
    end
    seen = falses(length(buses)); seen[1] = true; stack = [1]
    while !isempty(stack)
        node = pop!(stack)
        for next in adjacent[node]
            seen[next] || (seen[next] = true; push!(stack, next))
        end
    end
    return all(seen)
end

function chronological_context(base_ctx)
    cfg = TOML.parsefile(joinpath(base_ctx.root, "config", "experiment.toml"))
    base = DataPipeline.build_panel(base_ctx.system, cfg,
                                    Driver._sysdir(base_ctx.root, base_ctx.system))
    base = DataPipeline.build_delta!(base, base_ctx.pipe.case)
    full = wind_scaled_panel(base, base_ctx.pipe.case,
        Float64(base_ctx.wind_meta.wind_scale);
        lambda_eps = Float64(base_ctx.wind_meta.lambda_eps))
    split = DataPipeline.time_split(full, cfg)
    selected = Int.(base_ctx.pipe.screen_meta.selected_indices)
    base_ctx.panel.timestamps == full.timestamps[selected] ||
        error("stress/full chronological alignment mismatch")
    ctx = FinalContext(base_ctx.root, base_ctx.protocol_path, base_ctx.protocol,
        base_ctx.system, base_ctx.pipe, full, base_ctx.cd, base_ctx.scen,
        base_ctx.train, base_ctx.raw_delta, base_ctx.raw_omega, base_ctx.raw_eL,
        base_ctx.P, base_ctx.Pfinal, base_ctx.rho_tau_c, base_ctx.rho_tau_s,
        base_ctx.omega, base_ctx.wind_meta)
    return ctx, Int.(collect(split.test))
end

function state_context(base_ctx, generator_down, line_down)
    connected_after_line_outages(base_ctx.pipe.case, line_down) ||
        error("disconnected frozen line state")
    rawcase = if any(line_down)
        branch = copy(base_ctx.pipe.case.branch); branch[line_down, 11] .= 0.0
        Cases.RawCase(base_ctx.pipe.case.name, base_ctx.pipe.case.baseMVA,
            base_ctx.pipe.case.bus, base_ctx.pipe.case.gen, branch,
            base_ctx.pipe.case.gencost)
    else
        base_ctx.pipe.case
    end
    cfg = TOML.parsefile(joinpath(base_ctx.root, "config", "experiment.toml"))
    gen_keep = hasproperty(base_ctx.pipe, :gen_keep) ? base_ctx.pipe.gen_keep : nothing
    cd = build_casedata(rawcase, base_ctx.panel.load_buses,
        base_ctx.panel.wind_buses, base_ctx.panel.wind_cap, cfg; gen_keep = gen_keep)
    cd = apply_dispatchable_profile(cd, base_ctx.cd)
    cd = CaseInterface.apply_generator_availability(cd, .!generator_down)
    cd = CaseInterface.scale_Fmax(cd, base_ctx.wind_meta.line_limit_scale)
    return FinalContext(base_ctx.root, base_ctx.protocol_path, base_ctx.protocol,
        base_ctx.system, base_ctx.pipe, base_ctx.panel, cd, base_ctx.scen,
        base_ctx.train, base_ctx.raw_delta, base_ctx.raw_omega, base_ctx.raw_eL,
        base_ctx.P, base_ctx.Pfinal, base_ctx.rho_tau_c, base_ctx.rho_tau_s,
        base_ctx.omega, base_ctx.wind_meta)
end

function attach_state!(row, lock, draw, precheck, lock_hash::String,
                       draw_hash::String)
    row["scenario_id"] = lock["scenario_id"]
    row["scope"] = "unconditional"
    row["formal_lock_sha256"] = lock_hash
    row["draw_sha256"] = draw_hash
    row["state_id"] = Int(draw.mc_draw)
    row["candidate_draw"] = Int(draw.candidate_draw)
    row["component_source_mc_draw"] = Int(draw.component_source_mc_draw)
    row["source_hour"] = Int(draw.source_hour)
    row["timestamp"] = string(draw.timestamp)
    row["generator_down_indices"] = string(draw.generator_down_indices)
    row["line_down_indices"] = string(draw.line_down_indices)
    row["generator_outages"] = Int(draw.generator_outages)
    row["line_outages"] = Int(draw.line_outages)
    row["precheck_ok"] = precheck.ok
    row["precheck_status"] = string(precheck.status)
    row["precheck_time_s"] = precheck.time
    row["model_called"] = true
    return row
end

row_accepted(row) = haskey(row, "accepted") &&
                    lowercase(string(row["accepted"])) == "true"
row_model_called(row) = haskey(row, "model_called") &&
                        lowercase(string(row["model_called"])) == "true"

function prior_rows(path::String, lock_hash::String, draw_hash::String)
    isfile(path) || return Dict{String,Any}[]
    table = CSV.read(path, DataFrame)
    all(name -> name in names(table), ["formal_lock_sha256", "draw_sha256"]) ||
        return Dict{String,Any}[]
    return [Dict{String,Any}(String(name) => row[name] for name in names(table))
            for row in eachrow(table)
            if string(row.formal_lock_sha256) == lock_hash &&
               string(row.draw_sha256) == draw_hash]
end

function main(system::String, protocol_path::String, lock_path::String,
              draw_path::String, shard_index::Int, shard_count::Int,
              models::Vector{String}, output_dir::String, run_tag::String,
              seed_dir::Union{Nothing,String})
    all(in(UNCONDITIONAL_MODELS), models) || error("unsupported model")
    1 <= shard_index <= shard_count || error("invalid shard index")
    occursin(r"^[A-Za-z0-9_-]+$", run_tag) || error("invalid run tag")

    lock = TOML.parsefile(lock_path)
    String(lock["system"]) == system || error("lock/system mismatch")
    Bool(get(lock, "formal_test_unlocked", false)) || error("formal lock is locked")
    lock["protocol_sha256"] == hash_file(protocol_path) || error("protocol hash mismatch")
    manifest_path = joinpath(dirname(draw_path), "unconditional_draw_manifest.toml")
    manifest = TOML.parsefile(manifest_path)
    String(manifest["status"]) == "frozen_uniform_chronological_draws" ||
        error("unfrozen unconditional draw stream")
    String(manifest["scope"]) == "unconditional" || error("scope mismatch")
    String(manifest["system"]) == system || error("manifest/system mismatch")
    Bool(manifest["equal_hour_weights"]) || error("nonuniform hour weights")
    String(manifest["model_specific_gate"]) == "none" || error("model gate detected")
    lock_hash = hash_file(lock_path); draw_hash = hash_file(draw_path)
    String(manifest["formal_lock_sha256"]) == lock_hash || error("draw/lock mismatch")
    String(manifest["draw_file_sha256"]) == draw_hash || error("draw hash mismatch")

    resource = lock["resource_candidate"]
    candidate = candidate_dict(resource["wind_penetration"], resource["lambda_eps"],
        resource["replacement_ratio"], resource["target_rnet"])
    stress_ctx = load_candidate(protocol_path, candidate, Int(resource["rnet_state_count"]))
    ctx, chronological_test = chronological_context(stress_ctx)
    draws = CSV.read(draw_path, DataFrame)
    nrow(draws) == Int(manifest["state_count"]) || error("draw row-count mismatch")
    Int.(draws.mc_draw) == collect(1:nrow(draws)) || error("noncontiguous states")
    all(in(Set(chronological_test)), Int.(draws.snapshot)) || error("non-test snapshot")
    shard = draws[mod.(Int.(draws.mc_draw) .- 1, shard_count) .== shard_index - 1, :]

    protocol, screen = protocol_inputs(protocol_path)
    Pbase = baseline_parameters(protocol, screen)
    Pbase["support_margin"] = Float64(lock["baselines"]["M4_support_margin"])
    mkpath(output_dir)
    stem = "unconditional_$(system)_N$(nrow(draws))_shard$(shard_index)of$(shard_count)_$(run_tag)"
    long_path = joinpath(output_dir, stem * ".csv")
    rows = prior_rows(long_path, lock_hash, draw_hash)
    if isempty(rows) && !isnothing(seed_dir)
        isdir(seed_dir) || error("seed directory does not exist")
        state_ids = Set(Int.(shard.mc_draw))
        for file in sort(filter(path -> endswith(lowercase(path), ".csv"),
                                readdir(seed_dir; join = true)))
            for row in prior_rows(file, lock_hash, draw_hash)
                haskey(row, "state_id") && haskey(row, "model") || continue
                Int(row["state_id"]) in state_ids || continue
                String(row["model"]) in models || continue
                row_accepted(row) && row_model_called(row) || continue
                push!(rows, row)
            end
        end
        unique!(row -> (Int(row["state_id"]), String(row["model"])), rows)
        !isempty(rows) && write_csv_atomic(long_path, rows)
    end
    processed = Set((Int(row["state_id"]), String(row["model"])) for row in rows
                    if row_accepted(row) && row_model_called(row))

    for draw in eachrow(shard)
        snapshot = Int(draw.snapshot)
        generator_down = parse_down_indices(draw.generator_down_indices, ctx.cd.nG)
        line_down = parse_down_indices(draw.line_down_indices,
                                       size(ctx.pipe.case.branch, 1))
        state_ctx = state_context(ctx, generator_down, line_down)
        snap = snapshot_at(state_ctx.panel, snapshot, state_ctx.cd)
        precheck = Driver.dc_state_precheck(state_ctx.cd, snap)
        profile = nothing; profile_ready = false
        for model in models
            key = (Int(draw.mc_draw), model); key in processed && continue
            started = time()
            row = try
                if startswith(model, "Proposed-")
                    endpoint_name = model == "Proposed-Tradeoff" ? "tradeoff" : "safety"
                    endpoint = lock["endpoints"][endpoint_name]
                    run = solve_final_snapshot(state_ctx, snapshot;
                        rho_tau_c = Float64(endpoint["rho_tau_c"]),
                        rho_tau_s = Float64(endpoint["rho_tau_s"]),
                        omega = Float64.(endpoint["omega"]),
                        bandwidth_multiplier = Float64(endpoint["bandwidth_multiplier"]))
                    solved = solved_proposed_row(candidate, snapshot, state_ctx,
                                                 snap, run, time() - started)
                    solved["model"] = model; solved
                else
                    P = deepcopy(Pbase)
                    dispatch_model = model == "M3-ClassMax" ? "M3G" : model
                    if model in ("M2", "M3", "M3-ClassMax")
                        P["kde_mode"] = "fixed"
                        if !profile_ready
                            profile = kde_profile(state_ctx, snap, P)
                            profile_ready = true
                        end
                    end
                    model in ("M3", "M3-ClassMax") &&
                        (P["theta_m"] = Float64(lock["baselines"]["M3_theta"]))
                    model == "M4" && (P["wasserstein_radius_multiplier"] =
                        Float64(lock["baselines"]["M4_radius_multiplier"]))
                    u = Driver._solve_one(dispatch_model, state_ctx.cd, snap,
                        state_ctx.scen, P, 0.0; bandwidth_profile = profile)
                    solved_baseline_row(candidate, model, snapshot, state_ctx,
                                        snap, u, P, time() - started)
                end
            catch err
                failed = failed_row(candidate, model, snapshot, snap,
                    sprint(showerror, err), time() - started, state_ctx.wind_meta)
                failed["model"] = model; failed
            end
            curtail_cap = CaseInterface.sample_safe_curtailment_cap(
                state_ctx.cd, snap, state_ctx.scen)
            row["sample_safe_curtailment_cap_mw"] = sum(curtail_cap)
            row["sample_safe_curtailment_margin_mw"] =
                sum(curtail_cap) - Float64(row["curtail_mw"])
            attach_state!(row, lock, draw, precheck, lock_hash, draw_hash)
            previous = findfirst(existing ->
                Int(existing["state_id"]) == Int(draw.mc_draw) &&
                String(existing["model"]) == model, rows)
            isnothing(previous) ? push!(rows, row) : (rows[previous] = row)
            push!(processed, key)
            write_csv_atomic(long_path, rows)
            println(model, " state ", draw.mc_draw, "/", nrow(draws), ": ",
                    row["status"])
            flush(stdout)
        end
        GC.gc()
    end

    receipt = Dict{String,Any}(
        "status" => "unconditional_replay_shard_complete",
        "scope" => "unconditional", "equal_hour_weights" => true,
        "system" => system, "formal_lock_path" => abspath(lock_path),
        "formal_lock_sha256" => lock_hash, "draw_path" => abspath(draw_path),
        "draw_sha256" => draw_hash, "state_count" => nrow(draws),
        "shard_index" => shard_index, "shard_count" => shard_count,
        "models" => models, "state_ids" => Int.(shard.mc_draw),
        "long_file" => abspath(long_path),
        "long_file_sha256" => hash_file(long_path),
        "runner_script" => abspath(@__FILE__),
        "runner_script_sha256" => hash_file(@__FILE__),
        "experiment_source_tree_sha256" => execution_source_provenance(ROOT).tree_sha256,
        "failure_policy" => "conservative_full_shed_all_events")
    receipt_path = joinpath(output_dir, stem * "_receipt.toml")
    write_toml_atomic(receipt_path, receipt)
    println("Unconditional shard receipt: ", receipt_path)
end

main(String(ARGS[1]), normpath(ARGS[2]), normpath(ARGS[3]), normpath(ARGS[4]),
     parse(Int, ARGS[5]), parse(Int, ARGS[6]), String.(split(ARGS[7], ',')),
     normpath(ARGS[8]), String(ARGS[9]),
     length(ARGS) == 10 && ARGS[10] != "-" ? normpath(ARGS[10]) : nothing)
