#!/usr/bin/env julia

# Generate the post-lock, model-independent NSMC state stream for the frozen
# resource-pressure candidate.  Only topology and case constructibility may
# reject a candidate draw.

using CSV
using DataFrames
using Dates
using StableRNGs
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))
using .DRCCExp: Cases, ReliabilityStates

length(ARGS) == 2 || error(
    "usage: generate_resource_component_state_draws.jl protocol.toml resource_formal_lock.toml")

branch_online(rc::Cases.RawCase) = rc.branch[:, 11] .> 0
component_indices(down) = join(findall(down), ';')

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

function rawcase_with_line_state(rc::Cases.RawCase, down)
    branch = copy(rc.branch); branch[down, 11] .= 0.0
    return Cases.RawCase(rc.name, rc.baseMVA, rc.bus, rc.gen, branch, rc.gencost)
end

function main(protocol_path::String, lock_path::String)
    lock = TOML.parsefile(lock_path)
    Bool(get(lock, "formal_test_unlocked", false)) || error("formal lock is not unlocked")
    lock["protocol_sha256"] == sha256_file(protocol_path) || error("protocol hash mismatch")
    assert_execution_source_provenance(lock; root = ROOT)
    resource = lock["resource_candidate"]
    candidate = candidate_dict(resource["wind_penetration"], resource["lambda_eps"],
        resource["replacement_ratio"], resource["target_rnet"])
    candidate_key(candidate) == String(lock["candidate"]) || error("candidate lock mismatch")
    ctx = load_candidate(protocol_path, candidate, Int(resource["rnet_state_count"]))
    nsmc = ctx.protocol["nsmc"]
    max_valid = Int(nsmc["maximum_valid_states"])
    candidate_factor = Int(nsmc["candidate_factor"])
    candidate_count = max_valid * candidate_factor

    cfg = TOML.parsefile(joinpath(ROOT, "config", "experiment.toml"))
    outage_cfg = cfg["monte_carlo"]["generator_outages"]
    Bool(get(outage_cfg, "enabled", true)) || error("generator outages are disabled")
    seed = Int(nsmc["seed"])
    system_offset = sum(Int, codeunits(ctx.system))
    generator_seed = seed + Int(outage_cfg["seed_offset"]) + system_offset
    line_seed = seed + Int(outage_cfg["line_seed_offset"]) + system_offset
    candidate_cfg = deepcopy(cfg)
    candidate_cfg["monte_carlo"]["seed"] = seed
    candidate_cfg["monte_carlo"]["n_samples"] = candidate_count
    candidate_cfg["monte_carlo"]["n_samples_by_system"] = Dict{String,Any}()
    test_pool = collect(ctx.pipe.split.test)
    snapshots = Driver._mc_draws(ctx.system, test_pool, candidate_cfg)
    generators = ReliabilityStates.sample_generator_states(ctx.system, ctx.cd,
        candidate_count, generator_seed; profile = String(outage_cfg["profile"]))
    sample_lines = Bool(outage_cfg["sample_lines"])
    line_unavailability = Float64(outage_cfg["line_unavailability"])
    line_rng = StableRNG(line_seed)
    online_lines = branch_online(ctx.pipe.case)
    gen_keep = hasproperty(ctx.pipe, :gen_keep) ? ctx.pipe.gen_keep : nothing

    rows = DataFrame(candidate_draw = Int[], mc_draw = Union{Missing,Int}[],
        accepted = Bool[], topology_valid = Bool[], data_valid = Bool[], snapshot = Int[],
        source_hour = Int[], timestamp = String[], generator_down_indices = String[],
        line_down_indices = String[], generator_outages = Int[], line_outages = Int[],
        filter_reason = String[])
    accepted = 0; topology_ok = 0; data_ok = 0
    for candidate_draw in eachindex(snapshots)
        snapshot = Int(snapshots[candidate_draw])
        generator_down = vec(generators.down[candidate_draw, :])
        line_down = sample_lines ?
            [online_lines[i] && rand(line_rng) < line_unavailability for i in eachindex(online_lines)] :
            falses(length(online_lines))
        connected = connected_after_line_outages(ctx.pipe.case, line_down)
        topology_ok += connected
        constructible = false; reason = ""
        if !connected
            reason = "DISCONNECTED_LINE_STATE"
        else
            try
                rawcase = any(line_down) ? rawcase_with_line_state(ctx.pipe.case, line_down) : ctx.pipe.case
                build_casedata(rawcase, ctx.panel.load_buses, ctx.panel.wind_buses,
                    ctx.panel.wind_cap, cfg; gen_keep = gen_keep)
                constructible = true; data_ok += 1
            catch err
                reason = "CASE_BUILD_FAILED: " * replace(sprint(showerror, err), r"[\r\n]+" => " ")
            end
        end
        selected = constructible && accepted < max_valid
        selected && (accepted += 1)
        constructible && !selected && (reason = "NOT_SELECTED_AFTER_MAX_VALID_PREFIX")
        source_hour = ctx.pipe.screen_meta.selected_indices[snapshot]
        push!(rows, (candidate_draw, selected ? accepted : missing, selected, connected,
            constructible, snapshot, Int(source_hour), string(ctx.panel.timestamps[snapshot]),
            component_indices(generator_down), component_indices(line_down),
            count(generator_down), count(line_down), reason))
    end
    accepted == max_valid || error("only $accepted of $max_valid states were constructible")
    all(Int(row.snapshot) in Set(Int.(test_pool)) for row in eachrow(rows) if row.accepted) ||
        error("selected state lies outside the frozen test split")

    outdir = joinpath(@__DIR__, "results", "ResourceComponentDraws_$(ctx.system)_" *
                      Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    mkpath(outdir)
    draw_path = joinpath(outdir, "component_draws.csv"); CSV.write(draw_path, rows)
    manifest = Dict{String,Any}(
        "status" => "frozen_model_independent_component_draws",
        "scenario_id" => lock["scenario_id"], "system" => ctx.system,
        "candidate" => lock["candidate"], "test_split_only" => true,
        "model_specific_gate" => "none", "candidate_states" => candidate_count,
        "topology_valid_states" => topology_ok, "data_valid_states" => data_ok,
        "selected_valid_states" => accepted, "generator_seed" => generator_seed,
        "line_seed" => line_seed, "sample_lines" => sample_lines,
        "line_unavailability" => line_unavailability,
        "formal_lock_path" => abspath(lock_path),
        "formal_lock_sha256" => sha256_file(lock_path),
        "draw_file" => abspath(draw_path), "draw_file_sha256" => sha256_file(draw_path),
        "source_tree_sha256" => lock["source_tree_sha256"],
        "state_selection_rule" => "first constructible prefix in the frozen model-independent candidate stream")
    open(joinpath(outdir, "component_draw_manifest.toml"), "w") do io
        TOML.print(io, manifest; sorted = true)
    end
    println("Resource component draws: ", draw_path)
end

main(normpath(ARGS[1]), normpath(ARGS[2]))
