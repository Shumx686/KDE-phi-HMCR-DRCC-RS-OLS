#!/usr/bin/env julia

# Build the unconditional reliability reduction for the frozen two-regime
# operating policy.  A common deterministic no-shed dispatcher is used outside
# the training-defined emergency activation domain; M0--M6 are invoked inside
# that domain.  The full-domain all-model replay remains an execution audit and
# is never silently substituted for the operational policy.

using CSV
using DataFrames
using SHA
using Statistics
using TOML

include(joinpath(@__DIR__, "TestWindAcceptance.jl"))
using .TestWindAcceptance

const MODELS = ["M0", "M1", "M2", "M3", "M3-ClassMax", "M4", "M5",
                "Proposed-Tradeoff", "Proposed-Safety"]

length(ARGS) == 5 || error(
    "usage: build_hybrid_reliability_summary.jl SYSTEM RAW_LONG_CSV RAW_MANIFEST_TOML OUTPUT_DIR FROZEN_HYBRID_LONG")

asbool(value) = value isa Bool ? value : lowercase(strip(string(value))) == "true"
numeric(value; default = NaN) = try
    Float64(value)
catch
    default
end
hash_file(path::String) = bytes2hex(open(SHA.sha256, path))

function wilson_interval(successes::Int, trials::Int; z::Float64 = 1.959963984540054)
    trials == 0 && return (NaN, NaN)
    p = successes / trials
    denominator = 1.0 + z^2 / trials
    centre = (p + z^2 / (2trials)) / denominator
    radius = z * sqrt(p * (1.0 - p) / trials + z^2 / (4trials^2)) / denominator
    return (max(0.0, centre - radius), min(1.0, centre + radius))
end

function frozen_activation(rows, reference_path, system)
    reference = CSV.read(reference_path, DataFrame)
    reference_manifest = TOML.parsefile(replace(reference_path, "_long.csv" => "_manifest.toml"))
    reference_manifest["system"] == system || error("activation-reference system mismatch")
    reference_manifest["status"] == "complete" || error("incomplete activation reference")
    reference_manifest["policy_mode"] == "common_normal_dispatch_plus_emergency_activation" ||
        error("activation-reference policy mismatch")
    reference_manifest["long_file_sha256"] == hash_file(reference_path) ||
        error("activation-reference hash mismatch")
    nrow(reference) == nrow(rows) || error("activation-reference row count mismatch")
    allunique(zip(reference.state_id, reference.model)) || error("duplicate activation reference")
    Set(zip(reference.state_id, reference.model, reference.source_hour)) ==
        Set(zip(rows.state_id, rows.model, rows.source_hour)) || error("activation-reference state mismatch")
    hours = Set{Int}()
    for state in groupby(reference, :state_id)
        length(unique(state.in_emergency_activation_domain)) == 1 || error("model-dependent activation")
        asbool(state.in_emergency_activation_domain[1]) && push!(hours, Int(state.source_hour[1]))
    end
    all((Int(row.source_hour) in hours) == asbool(row.in_emergency_activation_domain)
        for row in eachrow(reference)) || error("inconsistent activation for repeated source hours")
    return hours
end

function verify_raw_input(rows::DataFrame, manifest::Dict{String,Any},
                          system::String, raw_long_path::String)
    String(manifest["status"]) == "complete" || error("raw reduction is incomplete")
    manifest["test_wind_gate_version"] == TEST_WIND_GATE_VERSION || error("uncorrected raw reduction")
    manifest["test_wind_audit_tol_mw"] == TEST_WIND_TOL_MW || error("wind tolerance mismatch")
    manifest["long_file_sha256"] == hash_file(raw_long_path) || error("raw long hash mismatch")
    String(manifest["system"]) == system || error("raw system mismatch")
    String(manifest["scope"]) == "unconditional" || error("raw scope mismatch")
    Bool(manifest["all_rows_model_called"]) || error("raw replay contains substitutions")
    String.(manifest["models"]) == MODELS || error("raw manifest model mismatch")
    Int(manifest["state_count"]) == 500 || error("expected N=500 unconditional states")
    normpath(String(manifest["long_file"])) == normpath(abspath(raw_long_path)) ||
        error("raw long file/manifest mismatch")

    required = [:model, :state_id, :source_hour, :accepted, :model_called,
                :precheck_ok, :shed_mw, :T_on_s]
    all(in(propertynames(rows)), required) || error("raw long table lacks required columns")
    nrow(rows) == 500 * length(MODELS) || error("raw row-count mismatch")
    Set(String.(rows.model)) == Set(MODELS) || error("raw model mismatch")
    all(asbool, rows.model_called) || error("not every raw row is an actual model call")
    nrow(unique(rows[:, [:state_id, :model]])) == nrow(rows) ||
        error("duplicate raw state/model row")
    sort(unique(Int.(rows.state_id))) == collect(1:500) || error("state coverage mismatch")
end

function main(system::String, raw_long_path::String,
              raw_manifest_path::String, output_dir::String, frozen_hybrid_path::String)
    raw_long_path = normpath(raw_long_path)
    raw_manifest_path = normpath(raw_manifest_path)
    output_dir = normpath(output_dir)
    isfile(raw_long_path) || error("missing raw long table")
    isfile(raw_manifest_path) || error("missing raw manifest")

    rows = CSV.read(raw_long_path, DataFrame)
    manifest = TOML.parsefile(raw_manifest_path)
    verify_raw_input(rows, manifest, system, raw_long_path)
    activation_hours = frozen_activation(rows, frozen_hybrid_path, system)

    hybrid = NamedTuple[]
    for state_id in 1:500
        state = rows[Int.(rows.state_id) .== state_id, :]
        nrow(state) == length(MODELS) || error("incomplete state $state_id")
        length(unique(Int.(state.source_hour))) == 1 || error("source-hour mismatch")
        length(unique(asbool.(state.precheck_ok))) == 1 || error("precheck mismatch")
        source_hour = Int(state.source_hour[1])
        active = source_hour in activation_hours
        m0 = state[String.(state.model) .== "M0", :][1, :]
        precheck_ok = asbool(state.precheck_ok[1])

        for model in MODELS
            original = state[String.(state.model) .== model, :][1, :]
            original_accepted = asbool(original.accepted)
            shed = numeric(original.shed_mw; default = Inf)
            accepted = original_accepted
            policy = "emergency_model"

            if !active
                if precheck_ok
                    shed = 0.0
                    accepted = true
                    policy = "common_normal_no_shed_dispatch"
                elseif asbool(m0.accepted)
                    shed = numeric(m0.shed_mw; default = Inf)
                    accepted = true
                    policy = "common_normal_m0_dispatch"
                else
                    shed = numeric(m0.shed_mw; default = Inf)
                    accepted = false
                    policy = "common_normal_conservative_failure"
                end
            end
            isfinite(shed) || error("nonfinite hybrid shedding at state $state_id/$model")
            push!(hybrid, (
                system = system,
                scope = "unconditional",
                model = model,
                state_id = state_id,
                source_hour = source_hour,
                in_emergency_activation_domain = active,
                operational_policy = policy,
                shed_mw = shed,
                accepted = accepted,
                raw_model_called = true,
                raw_model_accepted = original_accepted,
                raw_model_shed_mw = numeric(original.shed_mw; default = Inf),
                raw_test_wind_rejected = asbool(original.test_wind_rejected),
                operational_test_wind_rejected = active ? asbool(original.test_wind_rejected) :
                    (!precheck_ok && asbool(m0.test_wind_rejected)),
                test_wind_gate_version = TEST_WIND_GATE_VERSION,
                online_time_s = active && original_accepted ?
                    numeric(original.T_on_s; default = NaN) : NaN,
            ))
        end
    end
    hybrid_rows = DataFrame(hybrid)
    activation_states = length(unique(hybrid_rows.state_id[
        hybrid_rows.in_emergency_activation_domain]))
    outside_states = 500 - activation_states

    summaries = NamedTuple[]
    for model in MODELS
        group = hybrid_rows[String.(hybrid_rows.model) .== model, :]
        nrow(group) == 500 || error("hybrid model coverage mismatch")
        shed = Float64.(group.shed_mw)
        accepted = Bool.(group.accepted)
        lolp_event = shed .> 1.0e-3
        shed_sd = std(shed)
        shed_se = shed_sd / sqrt(length(shed))
        lolp_ci = wilson_interval(count(lolp_event), length(lolp_event))
        online = Float64.(group.online_time_s)
        online = online[isfinite.(online)]
        push!(summaries, (
            system = system,
            scope = "unconditional",
            model = model,
            n_states = nrow(group),
            accepted = count(accepted),
            nonaccepted = count(!, accepted),
            test_wind_rejected_count = count(group.operational_test_wind_rejected),
            shed_metric = "EDNS",
            mean_shed_mw = mean(shed),
            shed_sd_mw = shed_sd,
            shed_se_mw = shed_se,
            shed_cov = mean(shed) == 0.0 ? NaN : shed_se / mean(shed),
            lolp_metric = "LOLP",
            lolp = mean(lolp_event),
            lolp_wilson95_low = lolp_ci[1],
            lolp_wilson95_high = lolp_ci[2],
            eue_mwh_per_year = mean(shed) * 8760.0,
            safety_event_count = missing,
            safety_event_rate = missing,
            accepted_p95_excess_mw = missing,
            median_online_s = isempty(online) ? NaN : median(online),
            activation_state_count = activation_states,
            outside_activation_state_count = outside_states,
            normal_policy_state_count = outside_states,
            emergency_model_state_count = activation_states,
        ))
    end

    mkpath(output_dir)
    prefix = "final_$(lowercase(system))_unconditional"
    long_path = joinpath(output_dir, prefix * "_long.csv")
    summary_path = joinpath(output_dir, prefix * "_summary.csv")
    CSV.write(long_path, hybrid_rows)
    CSV.write(summary_path, DataFrame(summaries))

    output_manifest = Dict{String,Any}(
        "status" => "complete",
        "system" => system,
        "scope" => "unconditional",
        "state_count" => 500,
        "models" => MODELS,
        "policy_mode" => "common_normal_dispatch_plus_emergency_activation",
        "test_wind_gate_version" => TEST_WIND_GATE_VERSION,
        "test_wind_audit_tol_mw" => TEST_WIND_TOL_MW,
        "frozen_activation_reference" => abspath(frozen_hybrid_path),
        "frozen_activation_reference_sha256" => hash_file(frozen_hybrid_path),
        "new_solver_calls" => 0,
        "activation_rule" => "source hour belongs to the frozen training-defined stress test domain",
        "activation_state_count" => activation_states,
        "outside_activation_state_count" => outside_states,
        "outside_policy" => "common no-shed DC dispatch; if infeasible, common M0 load-shedding dispatch; conservative common failure otherwise",
        "inside_policy" => "actual frozen M0--M6 emergency-policy result with conservative nonacceptance",
        "raw_full_domain_all_models_called" => true,
        "raw_full_domain_replay_role" => "execution and applicability audit, not the operational reliability policy",
        "equal_hour_weights" => true,
        "eue_rule" => "EDNS multiplied by 8760 for uniformly sampled chronological test hours",
        "raw_long_file" => abspath(raw_long_path),
        "raw_long_sha256" => hash_file(raw_long_path),
        "raw_manifest_file" => abspath(raw_manifest_path),
        "raw_manifest_sha256" => hash_file(raw_manifest_path),
        "long_file" => abspath(long_path),
        "long_file_sha256" => hash_file(long_path),
        "summary_file" => abspath(summary_path),
        "summary_file_sha256" => hash_file(summary_path),
        "builder_script" => abspath(@__FILE__),
        "builder_script_sha256" => hash_file(@__FILE__),
    )
    manifest_path = joinpath(output_dir, prefix * "_manifest.toml")
    open(manifest_path, "w") do io
        TOML.print(io, output_manifest; sorted = true)
    end
    println("Hybrid reliability summary: ", abspath(summary_path))
    println("Activation/outside states: ", activation_states, "/", outside_states)
end

main(String(ARGS[1]), String(ARGS[2]), String(ARGS[3]), String(ARGS[4]), String(ARGS[5]))
