#!/usr/bin/env julia

# Generate an equal-hour chronological snapshot stream while reusing a frozen,
# model-independent component-availability stream.  Reusing component states
# makes the stress/unconditional scope contrast paired on outages; only the
# forecast-hour population changes.  No optimizer is called here.

using CSV
using DataFrames
using Dates
using SHA
using StableRNGs
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))
using .DRCCExp: DataPipeline, Driver

length(ARGS) == 9 || error(
    "usage: generate_unconditional_draws.jl SYSTEM PROTOCOL LOCK BASE_DRAWS FIRST_COMPONENT_STATE LAST_COMPONENT_STATE STATE_COUNT SNAPSHOT_SEED OUTPUT_DIR")

function hash_file(path::String)
    return bytes2hex(open(SHA.sha256, path))
end

function full_resource_panel(ctx)
    cfg = TOML.parsefile(joinpath(ctx.root, "config", "experiment.toml"))
    base = DataPipeline.build_panel(ctx.system, cfg, Driver._sysdir(ctx.root, ctx.system))
    base = DataPipeline.build_delta!(base, ctx.pipe.case)
    full = wind_scaled_panel(base, ctx.pipe.case, Float64(ctx.wind_meta.wind_scale);
                             lambda_eps = Float64(ctx.wind_meta.lambda_eps))
    split = DataPipeline.time_split(full, cfg)

    selected = Int.(ctx.pipe.screen_meta.selected_indices)
    ctx.panel.timestamps == full.timestamps[selected] ||
        error("stress/full timestamp alignment mismatch")
    maximum(abs.(ctx.panel.Lf .- full.Lf[selected, :])) <= 1.0e-10 ||
        error("stress/full load-forecast alignment mismatch")
    maximum(abs.(ctx.panel.Wact .- full.Wact[selected, :])) <= 1.0e-10 ||
        error("stress/full wind-actual alignment mismatch")
    return full, Int.(collect(split.test))
end

function main()
    system = String(ARGS[1])
    protocol_path = normpath(ARGS[2])
    lock_path = normpath(ARGS[3])
    base_draw_path = normpath(ARGS[4])
    first_component_state = parse(Int, ARGS[5])
    last_component_state = parse(Int, ARGS[6])
    state_count = parse(Int, ARGS[7])
    snapshot_seed = parse(Int, ARGS[8])
    output_dir = normpath(ARGS[9])
    state_count == last_component_state - first_component_state + 1 ||
        error("component-state range must equal STATE_COUNT")

    lock = TOML.parsefile(lock_path)
    String(lock["system"]) == system || error("lock/system mismatch")
    lock["protocol_sha256"] == hash_file(protocol_path) || error("protocol hash mismatch")
    resource = lock["resource_candidate"]
    candidate = candidate_dict(resource["wind_penetration"], resource["lambda_eps"],
        resource["replacement_ratio"], resource["target_rnet"])
    ctx = load_candidate(protocol_path, candidate, Int(resource["rnet_state_count"]))
    full_panel, chronological_test = full_resource_panel(ctx)

    base_rows = CSV.read(base_draw_path, DataFrame)
    selected = base_rows[Bool.(base_rows.accepted) .&
        (Int.(coalesce.(base_rows.mc_draw, -1)) .>= first_component_state) .&
        (Int.(coalesce.(base_rows.mc_draw, -1)) .<= last_component_state), :]
    sort!(selected, :mc_draw)
    nrow(selected) == state_count || error("base component-state coverage mismatch")
    Int.(selected.mc_draw) == collect(first_component_state:last_component_state) ||
        error("base component states are not contiguous")

    rng = StableRNG(snapshot_seed)
    snapshots = rand(rng, chronological_test, state_count)
    rows = DataFrame(
        candidate_draw = Int.(selected.candidate_draw),
        mc_draw = collect(1:state_count),
        accepted = trues(state_count),
        topology_valid = Bool.(selected.topology_valid),
        data_valid = Bool.(selected.data_valid),
        snapshot = snapshots,
        source_hour = snapshots,
        timestamp = string.(full_panel.timestamps[snapshots]),
        generator_down_indices = String.(coalesce.(selected.generator_down_indices, "")),
        line_down_indices = String.(coalesce.(selected.line_down_indices, "")),
        generator_outages = Int.(selected.generator_outages),
        line_outages = Int.(selected.line_outages),
        filter_reason = fill("", state_count),
        component_source_mc_draw = Int.(selected.mc_draw),
    )
    all(in(Set(chronological_test)), rows.snapshot) || error("non-test snapshot generated")

    mkpath(output_dir)
    draw_path = joinpath(output_dir, "unconditional_draws.csv")
    CSV.write(draw_path, rows)
    manifest = Dict{String,Any}(
        "status" => "frozen_uniform_chronological_draws",
        "scope" => "unconditional",
        "system" => system,
        "candidate" => lock["candidate"],
        "state_count" => state_count,
        "equal_hour_weights" => true,
        "chronological_test_first" => first(chronological_test),
        "chronological_test_last" => last(chronological_test),
        "chronological_test_hours" => length(chronological_test),
        "snapshot_sampling" => "uniform_with_replacement",
        "snapshot_seed" => snapshot_seed,
        "component_state_pairing" => "reuse frozen model-independent component states",
        "component_source_first" => first_component_state,
        "component_source_last" => last_component_state,
        "model_specific_gate" => "none",
        "formal_lock_path" => abspath(lock_path),
        "formal_lock_sha256" => hash_file(lock_path),
        "base_component_draw_path" => abspath(base_draw_path),
        "base_component_draw_sha256" => hash_file(base_draw_path),
        "draw_file" => abspath(draw_path),
        "draw_file_sha256" => hash_file(draw_path),
        "generator_script" => abspath(@__FILE__),
        "generator_script_sha256" => hash_file(@__FILE__),
        "created_at" => string(now()),
    )
    open(joinpath(output_dir, "unconditional_draw_manifest.toml"), "w") do io
        TOML.print(io, manifest; sorted = true)
    end
    println("Unconditional draws: ", abspath(draw_path))
end

main()
