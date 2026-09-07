#!/usr/bin/env julia

# Create the single post-validation lock for the RTS79 resource-pressure
# experiment.  This program audits validation artifacts only and never reads
# a test snapshot or a formal component-state draw.

using CSV
using DataFrames
using Dates
using Statistics
using TOML

include(joinpath(@__DIR__, "run_resource_pressure_screen.jl"))

length(ARGS) == 5 || error(
    "usage: lock_resource_formal_experiment.jl protocol.toml CANDIDATE validation_dir safety_dir m3_dir")

function long_files(dir::String, pattern::Regex)
    files = filter(path -> occursin(pattern, basename(path)) &&
                           !occursin("_summary", basename(path)),
                   readdir(dir; join = true))
    isempty(files) && error("no validation files matched $pattern in $dir")
    return sort(files)
end

function model_rows(files, model::String)
    table = reduce(vcat, (CSV.read(path, DataFrame) for path in files))
    rows = table[String.(table.model) .== model, :]
    nrow(rows) > 0 || error("no rows found for model=$model")
    length(unique(Int.(rows.snapshot))) == nrow(rows) ||
        error("duplicate validation state for model=$model")
    return rows
end

function any_event(row)
    return any(Bool(row[field]) for field in
        (:RU_event, :RD_event, :Fplus_event, :Fminus_event,
         :Splus_event, :Sminus_event))
end

function metrics(rows)
    accepted = rows[Bool.(rows.accepted), :]
    return (n = nrow(rows), accepted = nrow(accepted),
            edns = mean(Float64.(accepted.shed_mw)),
            any_count = count(any_event, eachrow(accepted)),
            online = mean(Float64.(accepted.T_on_s)))
end

function main(protocol_path::String, requested::String, validation_dir::String,
              safety_dir::String, m3_dir::String)
    protocol, screen = protocol_inputs(protocol_path)
    declared = Dict(candidate_key(c) => c for c in declared_candidates(screen))
    haskey(declared, requested) || error("candidate is not declared")
    candidate = declared[requested]
    ctx = load_candidate(protocol_path, candidate, 60)

    base_file = only(long_files(validation_dir,
        r"M0-M4-M5_shard1of1\.csv$"))
    m2_files = long_files(validation_dir, r"_M2_shard[12]of2\.csv$")
    trade_file = only(long_files(validation_dir,
        r"_Proposed_Safety_shard1of1\.csv$"))
    safety_files = long_files(safety_dir,
        r"_Proposed_SafetyBW275_shard[12]of2\.csv$")
    m3_files = long_files(m3_dir, r"_M3T050_shard[1-4]of4\.csv$")

    rows = Dict(
        "M0" => model_rows([base_file], "M0"),
        "M2" => model_rows(m2_files, "M2"),
        "M4" => model_rows([base_file], "M4"),
        "M5" => model_rows([base_file], "M5"),
        "Proposed-Tradeoff" => model_rows([trade_file], "Proposed-Safety"),
        "Proposed-Safety" => model_rows(safety_files, "Proposed-SafetyBW275"),
        "M3" => model_rows(m3_files, "M3T050"),
    )
    ids = sort(Int.(rows["M0"].snapshot))
    length(ids) == 60 || error("formal lock requires exactly 60 validation states")
    for (model, table) in rows
        sort(Int.(table.snapshot)) == ids || error("$model used different validation states")
        all(String.(table.candidate) .== requested) || error("$model candidate mismatch")
    end
    validation_set = Set(Int.(collect(ctx.pipe.split.val)))
    test_set = Set(Int.(collect(ctx.pipe.split.test)))
    all(in(validation_set), ids) || error("validation evidence includes a non-validation state")
    isempty(intersect(Set(ids), test_set)) || error("validation evidence touched test data")

    stat = Dict(model => metrics(table) for (model, table) in rows)
    for model in ("M0", "M4", "M5", "M3", "Proposed-Tradeoff", "Proposed-Safety")
        stat[model].accepted == 60 || error("$model is not 60/60 accepted")
    end
    stat["M2"].accepted / stat["M2"].n >=
        Float64(screen["confirmation_minimum_acceptance_rate"]) ||
        error("M2 did not pass the predeclared validation acceptance gate")
    stat["Proposed-Safety"].any_count == 0 ||
        error("Safety endpoint has an observed validation safety event")
    stat["Proposed-Tradeoff"].any_count > 0 ||
        error("Tradeoff endpoint does not separate from Safety on validation risk")
    stat["Proposed-Tradeoff"].edns < stat["Proposed-Safety"].edns ||
        error("Tradeoff must shed less than Safety")
    for endpoint in ("Proposed-Tradeoff", "Proposed-Safety"), heavy in ("M4", "M5")
        stat[endpoint].edns < stat[heavy].edns ||
            error("$endpoint does not shed less than $heavy")
    end
    unique(Float64.(skipmissing(rows["M4"].m4_support_margin))) == [0.0] ||
        error("M4 support margin is not the validation-selected zero margin")

    artifacts = unique(vcat([base_file, trade_file], m2_files, safety_files, m3_files))
    config_path = joinpath(ROOT, "config", "experiment.toml")
    cache_root = joinpath(ROOT, String(protocol["case"]["pipeline_cache_dir"]), ctx.system)
    data_paths = [joinpath(cache_root, "panel.arrow"), joinpath(cache_root, "split.toml"),
                  joinpath(cache_root, "scenarios_K$(protocol["case"]["K"]).csv")]
    scenario_id = resource_scenario_code(ctx;
        seed = Int(protocol["nsmc"]["seed"]), outage = "NSMC", split = "SPLIT01")
    lock = merge(provenance_record(root = ROOT, protocol_path = protocol_path,
        config_path = config_path, data_paths = data_paths, scenario_id = scenario_id,
        seed = Int(protocol["nsmc"]["seed"])), Dict{String,Any}(
        "status" => "locked", "formal_test_unlocked" => true,
        "test_access_before_lock" => "forbidden",
        "system" => ctx.system, "candidate" => requested,
        "validation_state_ids" => ids, "validation_state_count" => length(ids),
        "selection_rule" => "Tradeoff: lower shedding with nonzero-but-low union risk; Safety: zero observed union events then minimum validation EDNS; no test result used",
        "validation_artifacts" => abspath.(artifacts),
        "validation_artifact_sha256" => Dict(abspath(path) => sha256_file(path) for path in artifacts),
        "resource_candidate" => Dict(
            "wind_penetration" => candidate["wind_penetration"],
            "lambda_eps" => candidate["lambda_eps"],
            "replacement_ratio" => candidate["replacement_ratio"],
            "target_rnet" => candidate["target_rnet"],
            "rnet_state_count" => 60),
        "endpoints" => Dict(
            "tradeoff" => Dict("rho_tau_c" => 0.5, "rho_tau_s" => 1.0,
                "omega" => ones(6), "bandwidth_multiplier" => 1.0,
                "validation_edns_mw" => stat["Proposed-Tradeoff"].edns,
                "validation_any_event_count" => stat["Proposed-Tradeoff"].any_count),
            "safety" => Dict("rho_tau_c" => 1.0, "rho_tau_s" => 1.0,
                "omega" => ones(6), "bandwidth_multiplier" => 2.75,
                "validation_edns_mw" => stat["Proposed-Safety"].edns,
                "validation_any_event_count" => stat["Proposed-Safety"].any_count)),
        "baselines" => Dict("M3_theta" => 0.05,
            "M4_radius_multiplier" => Float64(screen["m4_radius_multiplier_anchor"]),
            "M4_support_margin" => 0.0),
        "validation_metrics" => Dict(model => Dict(
            "states" => value.n, "accepted" => value.accepted,
            "edns_mw" => value.edns, "any_event_count" => value.any_count,
            "mean_online_s" => value.online) for (model, value) in stat),
    ))
    outdir = joinpath(@__DIR__, "results", "ResourceFormalLock_$(ctx.system)_" *
                      Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    mkpath(outdir)
    path = joinpath(outdir, "resource_formal_lock.toml")
    open(path, "w") do io
        TOML.print(io, lock; sorted = true)
    end
    println("Resource formal lock: ", path)
end

main(normpath(ARGS[1]), String(ARGS[2]), normpath(ARGS[3]),
     normpath(ARGS[4]), normpath(ARGS[5]))
