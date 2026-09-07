#!/usr/bin/env julia

# Two-stage, validation-only resource-pressure screen for the revised TPS plan.
# Phase `pilot` runs M0 on the full declared grid, then M2/M4/Proposed-RS on a
# predeclared short list. Phase `confirm` reruns the retained candidates on 60
# common validation states. No test index or component-outage test draw is read.

using CSV
using DataFrames
using Dates
using Statistics
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "DRCCExp.jl"))
using .DRCCExp
using .DRCCExp: Driver
include(joinpath(@__DIR__, "FinalSevenTargetModel.jl"))
using .FinalSevenTargetModel
include(joinpath(@__DIR__, "FinalExperimentSupport.jl"))
using .FinalExperimentSupport
include(joinpath(@__DIR__, "PlanExperimentSupport.jl"))
using .PlanExperimentSupport

const SCREEN_MODELS = ("M0", "M2", "M4", "Proposed-RS")
const EVENT_FIELDS = (("RU", :RU_event), ("RD", :RD_event),
                      ("Fplus", :Fp_event), ("Fminus", :Fm_event),
                      ("Sminus", :Sm_event), ("Splus", :Sp_event))

function baseline_parameters(protocol, screen)
    cfg = TOML.parsefile(joinpath(ROOT, "config", "experiment.toml"))
    P = Dict{String,Any}()
    for group in (cfg["model"], get(cfg, "baseline", Dict{String,Any}()))
        for (key, value) in group
            P[String(key)] = value
        end
    end
    mcfg = protocol["model"]
    P["p"] = Int(mcfg["p"])
    P["gamma_m"] = Float64(mcfg["gamma_m"])
    P["kde_quad_rule"] = "gauss_hermite"
    P["kde_quad_nodes"] = Int(mcfg["quadrature_order"])
    P["kde_quad_weight_floor"] = 1.0e-8
    P["wasserstein_radius_mode"] = "compression_multiple"
    P["wasserstein_radius_multiplier"] = Float64(screen["m4_radius_multiplier_anchor"])
    P["support_margin"] = Float64(screen["m4_support_margin"])
    P["wasserstein_support_source"] = "raw_training"
    P["moment_source"] = "raw_training"
    return P
end

function m4_support_margin_audit(ctx, validation_ids, screen)
    margins = sort(unique(Float64.(screen["m4_support_margin_grid"])))
    target = Float64(screen["m4_support_validation_coverage"])
    0.0 <= target <= 1.0 || error("M4 support coverage target must lie in [0,1]")
    lo = ctx.scen.raw_lower
    hi = ctx.scen.raw_upper
    span = max.(hi .- lo, 1.0e-6)
    delta = ctx.panel.δ[Int.(validation_ids), :]
    rows = Dict{String,Any}[]
    for margin in margins
        lower = lo .- margin .* span
        upper = hi .+ margin .* span
        covered = [all(delta[i, :] .>= lower .- 1e-12) &&
                   all(delta[i, :] .<= upper .+ 1e-12)
                   for i in axes(delta, 1)]
        push!(rows, Dict{String,Any}(
            "support_margin" => margin,
            "validation_state_coverage" => mean(covered),
            "covered_states" => count(covered),
            "validation_states" => length(covered),
            "coverage_target" => target,
        ))
    end
    eligible = [row for row in rows if
                Float64(row["validation_state_coverage"]) >= target]
    isempty(eligible) && error("no declared M4 support margin reaches validation coverage=$target")
    return Float64(first(eligible)["support_margin"]), rows
end

function candidate_dict(wind, lambda_eps, replacement_ratio, target_rnet)
    return Dict{String,Any}(
        "wind_penetration" => Float64(wind),
        "lambda_eps" => Float64(lambda_eps),
        "replacement_ratio" => Float64(replacement_ratio),
        "target_rnet" => Float64(target_rnet),
    )
end

function candidate_key(candidate)
    tag(x, scale, width) = lpad(string(round(Int, scale * Float64(x))), width, '0')
    return "W$(tag(candidate["wind_penetration"], 100, 2))_" *
           "UQ$(tag(candidate["lambda_eps"], 100, 3))_" *
           "REP$(tag(candidate["replacement_ratio"], 100, 3))_" *
           "RNET$(tag(candidate["target_rnet"], 100, 2))"
end

function declared_candidates(screen)
    return [candidate_dict(w, uq, rep, rnet)
            for w in Float64.(screen["wind_penetrations"])
            for uq in Float64.(screen["lambda_eps_grid"])
            for rep in Float64.(screen["replacement_ratios"])
            for rnet in Float64.(screen["target_rnet_grid"])]
end

function load_candidate(protocol_path, candidate, nstates)
    return load_final_context(protocol_path;
        target_wind_penetration = Float64(candidate["wind_penetration"]),
        lambda_eps = Float64(candidate["lambda_eps"]),
        replacement_ratio = Float64(candidate["replacement_ratio"]),
        target_rnet = Float64(candidate["target_rnet"]),
        rnet_state_count = nstates)
end

function kde_profile(ctx, snap, P)
    pilotP = deepcopy(P)
    pilotP["kde_mode"] = "off"
    pilot = build_ols(ctx.cd, snap, ctx.scen, pilotP; safety = :cvar, cost = :expected)
    status_ok(pilot.status) || error("M2 bandwidth pilot failed: $(pilot.status)")
    return build_fixed_bandwidth_profile(ctx.cd, snap, ctx.raw_delta, ctx.raw_omega,
                                         ctx.raw_eL, pilot.x; multiplier = 1.0)
end

function failed_row(candidate, model, snapshot, snap, reason, elapsed, meta)
    row = Dict{String,Any}(
        "candidate" => candidate_key(candidate), "model" => model,
        "snapshot" => snapshot, "status" => "NONACCEPTED", "accepted" => false,
        "failure_reason" => replace(reason, r"[\r\n]+" => " "),
        "shed_mw" => sum(snap.lact), "curtail_mw" => 0.0,
        "wind_curtail_excess_mw" => Inf, "wind_curtail_event" => true,
        "LOLP_event" => true, "T_on_s" => NaN, "T_e2e_s" => elapsed,
        "secondary_objective_mw" => NaN,
        "lexicographic_activated" => missing,
        "lexicographic_stage2_status" => "",
        "wind_level" => candidate["wind_penetration"],
        "lambda_eps" => candidate["lambda_eps"],
        "replacement_ratio" => candidate["replacement_ratio"],
        "target_rnet" => candidate["target_rnet"],
        "rnet_median" => meta.rnet_median, "rnet_forecast" => NaN,
    )
    for (label, _) in EVENT_FIELDS
        row["$(label)_event"] = true
    end
    return row
end

function solved_baseline_row(candidate, model, snapshot, ctx, snap, u, P, elapsed)
    accepted = Driver._solution_accepted(u, P)
    if !accepted
        details = "status=$(u.status),primal=$(u.primal_residual),dual=$(u.dual_residual)," *
                  "gap_abs=$(u.gap_abs),gap_rel=$(u.gap_rel)," *
                  "physical=$(u.physical_violation),physical_label=$(u.physical_violation_label)"
        return failed_row(candidate, model, snapshot, snap,
            "solver_or_physical_quality_gate:$details", elapsed, ctx.wind_meta)
    end
    x = u.x
    audit = directional_audit(ctx.cd, x, snap;
        beta = getproperty(x, Symbol("β")), rhoG = getproperty(x, Symbol("ρG")),
        alphaS = getproperty(x, Symbol("αS")))
    forecast_netload = sum(snap.lf) - sum(snap.wf)
    row = Dict{String,Any}(
        "candidate" => candidate_key(candidate), "model" => model,
        "snapshot" => snapshot, "status" => string(u.status), "accepted" => true,
        "failure_reason" => "", "shed_mw" => audit.realized_shed_mw,
        "curtail_mw" => audit.curtail,
        "wind_curtail_excess_mw" => audit.curtail_audit,
        "wind_curtail_event" => audit.curtail_audit > Float64(P["audit_tol_mw"]),
        "LOLP_event" => audit.realized_shed_mw > Float64(P["audit_tol_mw"]),
        "T_on_s" => u.solve_time, "T_e2e_s" => elapsed,
        "secondary_objective_mw" => NaN,
        "lexicographic_activated" => missing,
        "lexicographic_stage2_status" => "",
        "wind_level" => candidate["wind_penetration"],
        "lambda_eps" => candidate["lambda_eps"],
        "replacement_ratio" => candidate["replacement_ratio"],
        "target_rnet" => candidate["target_rnet"],
        "rnet_median" => ctx.wind_meta.rnet_median,
        "m4_support_margin" => model == "M4" ? Float64(P["support_margin"]) : NaN,
        "rnet_forecast" => forecast_netload > 0.0 ?
            (ctx.wind_meta.dispatchable_pmax_mw - forecast_netload) / forecast_netload : NaN,
    )
    for (label, field) in EVENT_FIELDS
        row["$(label)_event"] = getproperty(audit, field)
        excess_field = Symbol(replace(String(field), "_event" => "_excess_mw"))
        row["$(label)_excess_mw"] = getproperty(audit, excess_field)
    end
    row["max_safety_excess_mw"] = maximum(
        Float64(row["$(label)_excess_mw"]) for (label, _) in EVENT_FIELDS)
    return row
end

function solved_proposed_row(candidate, snapshot, ctx, snap, run, elapsed)
    acceptance = ctx.protocol["acceptance"]
    accepted = accepted_result(run.final, acceptance) &&
               reference_quality_accepted(run.calibration, acceptance)
    if !accepted
        return failed_row(candidate, "Proposed-RS", snapshot, snap,
            "solver_reference_or_physical_quality_gate:$(run.final.status)", elapsed,
            ctx.wind_meta)
    end
    audit_tol = Float64(get(ctx.P, "audit_tol_mw", 1.0e-3))
    audit = final_directional_audit(ctx.cd, run.final, snap;
        tol_mw = audit_tol)
    forecast_netload = sum(snap.lf) - sum(snap.wf)
    row = Dict{String,Any}(
        "candidate" => candidate_key(candidate), "model" => "Proposed-RS",
        "snapshot" => snapshot, "status" => string(run.final.status), "accepted" => true,
        "failure_reason" => "", "shed_mw" => audit.realized_shed_mw,
        "curtail_mw" => audit.curtail,
        "wind_curtail_excess_mw" => audit.curtail_audit,
        "wind_curtail_event" => audit.curtail_audit > audit_tol,
        "LOLP_event" => audit.realized_shed_mw > audit_tol,
        "T_on_s" => run.final.solve_time, "T_e2e_s" => elapsed,
        "secondary_objective_mw" => run.final.secondary_objective,
        "lexicographic_activated" => run.final.lexicographic_activated,
        "lexicographic_stage2_status" => run.final.lexicographic_stage2_status,
        "lexicographic_primary_star" => run.final.lexicographic_primary_star,
        "lexicographic_primary_tolerance" => run.final.lexicographic_primary_tolerance,
        "lexicographic_primary_final" => run.final.lexicographic_primary_final,
        "stage1_solve_time_s" => run.final.stage1_solve_time,
        "stage2_solve_time_s" => run.final.stage2_solve_time,
        "pilot_solver_mode" => run.pilot_solver_mode,
        "calibration_solver_mode" => run.calibration_mode,
        "reference_solver_mode" => run.reference_solver_mode,
        "final_solver_mode" => run.final_solver_mode,
        "kappa_c" => run.final.kappa_c,
        "kappa_safety_sum" => sum(run.final.kappa_m),
        "bandwidth_multiplier" => run.bandwidth.multiplier,
        "wind_level" => candidate["wind_penetration"],
        "lambda_eps" => candidate["lambda_eps"],
        "replacement_ratio" => candidate["replacement_ratio"],
        "target_rnet" => candidate["target_rnet"],
        "rnet_median" => ctx.wind_meta.rnet_median,
        "rnet_forecast" => forecast_netload > 0.0 ?
            (ctx.wind_meta.dispatchable_pmax_mw - forecast_netload) / forecast_netload : NaN,
    )
    for (label, field) in EVENT_FIELDS
        row["$(label)_event"] = getproperty(audit, field)
        excess_field = Symbol(replace(String(field), "_event" => "_excess_mw"))
        row["$(label)_excess_mw"] = getproperty(audit, excess_field)
    end
    row["max_safety_excess_mw"] = maximum(
        Float64(row["$(label)_excess_mw"]) for (label, _) in EVENT_FIELDS)
    for m in 1:6
        row["kappa_$(m)"] = run.final.kappa_m[m]
    end
    return row
end

function solve_candidate_models(protocol_path, protocol, screen, candidate, nstates, models)
    ctx = load_candidate(protocol_path, candidate, nstates)
    validation_ids = frozen_indices(ctx.pipe.split.val, nstates)
    isempty(intersect(Set(validation_ids), Set(ctx.pipe.split.test))) ||
        error("resource-pressure screen touched a test state")
    Pbase = baseline_parameters(protocol, screen)
    selected_margin, support_audit = m4_support_margin_audit(ctx, validation_ids, screen)
    Pbase["support_margin"] = selected_margin
    rows = Dict{String,Any}[]
    for snapshot in validation_ids
        snap = snapshot_at(ctx.panel, snapshot, ctx.cd)
        for model in models
            started = time()
            row = try
                if model == "Proposed-RS"
                    run = solve_final_snapshot(ctx, snapshot)
                    solved_proposed_row(candidate, snapshot, ctx, snap, run, time() - started)
                else
                    P = deepcopy(Pbase)
                    profile = nothing
                    if model == "M2"
                        P["kde_mode"] = "fixed"
                        profile = kde_profile(ctx, snap, P)
                    elseif model == "M4"
                        P["wasserstein_radius_multiplier"] =
                            Float64(screen["m4_radius_multiplier_anchor"])
                    elseif model != "M0" && model != "M5"
                        error("undeclared resource-pressure screen model $model")
                    end
                    u = Driver._solve_one(model, ctx.cd, snap, ctx.scen, P, 0.0;
                                          bandwidth_profile = profile)
                    solved_baseline_row(candidate, model, snapshot, ctx, snap, u, P,
                                        time() - started)
                end
            catch err
                failed_row(candidate, model, snapshot, snap, sprint(showerror, err),
                           time() - started, ctx.wind_meta)
            end
            push!(rows, row)
            println(candidate_key(candidate), ",", model, ",", snapshot, ",", row["status"])
        end
        GC.gc()
    end
    return ctx, rows
end

function summarize_model(rows, candidate, model, expected_n)
    key = candidate_key(candidate)
    part = [row for row in rows if row["candidate"] == key && row["model"] == model]
    n = length(part)
    acceptance_rate = n == 0 ? 0.0 : mean(Bool(row["accepted"]) for row in part)
    accepted = [row for row in part if Bool(row["accepted"])]
    lolp = isempty(accepted) ? NaN : mean(Bool(row["LOLP_event"]) for row in accepted)
    event_rates = Dict(label => isempty(accepted) ? NaN :
        mean(Bool(row["$(label)_event"]) for row in accepted)
        for (label, _) in EVENT_FIELDS)
    max_safety = isempty(accepted) ? NaN : maximum(values(event_rates))
    any_safety = isempty(accepted) ? NaN : mean(any(
        Bool(row["$(label)_event"]) for (label, _) in EVENT_FIELDS)
        for row in accepted)
    reserve_union = isempty(accepted) ? NaN : mean(
        Bool(row["RU_event"]) || Bool(row["RD_event"]) for row in accepted)
    line_union = isempty(accepted) ? NaN : mean(
        Bool(row["Fplus_event"]) || Bool(row["Fminus_event"]) for row in accepted)
    shedbound_union = isempty(accepted) ? NaN : mean(
        Bool(row["Sminus_event"]) || Bool(row["Splus_event"]) for row in accepted)
    sheds = Float64[row["shed_mw"] for row in accepted]
    result = Dict{String,Any}(
        "candidate" => key, "model" => model, "n_states" => n,
        "expected_states" => expected_n, "acceptance_rate" => acceptance_rate,
        "LOLP" => lolp, "EDNS" => isempty(sheds) ? NaN : mean(sheds),
        "conditional_shed_mw" => conditional_mean(sheds),
        "max_safety_event_rate" => max_safety,
        "any_safety_event_rate" => any_safety,
        "reserve_union_event_rate" => reserve_union,
        "line_union_event_rate" => line_union,
        "shedbound_union_event_rate" => shedbound_union,
        "mean_T_on_s" => isempty(accepted) ? NaN : mean(Float64(row["T_on_s"]) for row in accepted),
        "mean_T_e2e_s" => isempty(accepted) ? NaN : mean(Float64(row["T_e2e_s"]) for row in accepted),
    )
    for (label, _) in EVENT_FIELDS
        result["$(label)_event_rate"] = event_rates[label]
    end
    return result
end

function candidate_summary(candidate, model_summaries, ctx, screen, minimum_acceptance_rate)
    bymodel = Dict(String(row["model"]) => row for row in model_summaries)
    m0 = bymodel["M0"]
    all_present = all(model -> haskey(bymodel, model), SCREEN_MODELS)
    min_acceptance = all_present ? minimum(Float64(bymodel[m]["acceptance_rate"]) for m in SCREEN_MODELS) : 0.0
    m4_conditional = all_present ? Float64(bymodel["M4"]["conditional_shed_mw"]) : NaN
    m0_conditional = Float64(m0["conditional_shed_mw"])
    robust_explainable = !all_present || !isfinite(m4_conditional) ? false :
        m4_conditional <= Float64(screen["strong_baseline_conditional_shed_multiplier_max"]) *
                          max(m0_conditional, 1e-6)
    m0_lolp = Float64(m0["LOLP"])
    saturation_threshold = Float64(screen["risk_model_saturation_lolp"])
    risk_models = ("M2", "M4", "Proposed-RS")
    saturated_risk_models = all_present ?
        count(model -> Float64(bymodel[model]["LOLP"]) >= saturation_threshold,
              risk_models) : length(risk_models)
    strong_baseline_nonsaturated = all_present &&
        Float64(bymodel["M4"]["LOLP"]) < saturation_threshold
    require_strong_baseline_nonsaturated = Bool(get(
        screen, "require_strong_baseline_nonsaturated", true))
    proposed_nonsaturated = all_present &&
        Float64(bymodel["Proposed-RS"]["LOLP"]) < saturation_threshold
    risk_models_discriminating = saturated_risk_models <=
        Int(screen["maximum_saturated_risk_models"])
    proposed = all_present ? bymodel["Proposed-RS"] : Dict{String,Any}()
    baselines = all_present ? [bymodel[m] for m in ("M0", "M2", "M4")] : Dict{String,Any}[]
    p_edns = all_present ? Float64(proposed["EDNS"]) : NaN
    p_safety = all_present ? Float64(proposed["max_safety_event_rate"]) : NaN
    dominates(a, b) = Float64(a["EDNS"]) <= Float64(b["EDNS"]) &&
                      Float64(a["max_safety_event_rate"]) <= Float64(b["max_safety_event_rate"]) &&
                      (Float64(a["EDNS"]) < Float64(b["EDNS"]) ||
                       Float64(a["max_safety_event_rate"]) < Float64(b["max_safety_event_rate"]))
    proposed_dominated_by = all_present ?
        [String(b["model"]) for b in baselines if dominates(b, proposed)] : String[]
    # This is a diagnostic only, not a pressure-selection gate: a desired
    # middle tradeoff is bracketed by one cheaper/riskier and one
    # safer/more-curtailed comparator on the common validation states.
    cheaper_riskier = all_present ? [String(b["model"]) for b in baselines if
        Float64(b["EDNS"]) < p_edns && Float64(b["max_safety_event_rate"]) > p_safety] : String[]
    safer_costlier = all_present ? [String(b["model"]) for b in baselines if
        Float64(b["EDNS"]) > p_edns && Float64(b["max_safety_event_rate"]) < p_safety] : String[]
    gate = all_present && min_acceptance >= minimum_acceptance_rate &&
           ctx.wind_meta.forecast_adequacy_rate >= Float64(screen["minimum_forecast_adequacy_rate"]) &&
           Float64(screen["m0_lolp_min"]) <= m0_lolp <= Float64(screen["m0_lolp_max"]) &&
           robust_explainable &&
           (!require_strong_baseline_nonsaturated || strong_baseline_nonsaturated) &&
           proposed_nonsaturated &&
           risk_models_discriminating
    row = Dict{String,Any}(
        "candidate" => candidate_key(candidate),
        "wind_penetration" => candidate["wind_penetration"],
        "lambda_eps" => candidate["lambda_eps"],
        "replacement_ratio" => candidate["replacement_ratio"],
        "target_rnet" => candidate["target_rnet"],
        "wind_scale" => ctx.wind_meta.wind_scale,
        "dispatchable_pmax_mw" => ctx.wind_meta.dispatchable_pmax_mw,
        "rnet_min" => ctx.wind_meta.rnet_min, "rnet_median" => ctx.wind_meta.rnet_median,
        "rnet_max" => ctx.wind_meta.rnet_max,
        "forecast_adequacy_rate" => ctx.wind_meta.forecast_adequacy_rate,
        "realization_inadequacy_rate" => ctx.wind_meta.realization_inadequacy_rate,
        "minimum_acceptance_rate" => min_acceptance,
        "M0_LOLP" => m0_lolp, "M0_EDNS" => m0["EDNS"],
        "M4_conditional_shed_mw" => m4_conditional,
        "robust_baseline_explainable" => robust_explainable,
        "risk_model_saturation_lolp" => saturation_threshold,
        "saturated_risk_model_count" => saturated_risk_models,
        "strong_baseline_nonsaturated" => strong_baseline_nonsaturated,
        "strong_baseline_nonsaturation_required" =>
            require_strong_baseline_nonsaturated,
        "proposed_nonsaturated" => proposed_nonsaturated,
        "risk_models_discriminating" => risk_models_discriminating,
        "Proposed_max_safety_event_rate" => p_safety,
        "Proposed_pareto_nondominated" => all_present && isempty(proposed_dominated_by),
        "Proposed_dominated_by" => join(proposed_dominated_by, ";"),
        "cheaper_riskier_baselines" => join(cheaper_riskier, ";"),
        "safer_costlier_baselines" => join(safer_costlier, ";"),
        "desired_tradeoff_bracketing_observed" => !isempty(cheaper_riskier) &&
                                                   !isempty(safer_costlier),
        "gate_pass" => gate,
    )
    for model in SCREEN_MODELS
        if haskey(bymodel, model)
            safe = replace(model, "-" => "_")
            row["$(safe)_LOLP"] = bymodel[model]["LOLP"]
            row["$(safe)_EDNS"] = bymodel[model]["EDNS"]
            row["$(safe)_max_safety_event_rate"] =
                bymodel[model]["max_safety_event_rate"]
            row["$(safe)_acceptance_rate"] = bymodel[model]["acceptance_rate"]
        end
    end
    return row
end

function pressure_rank(candidate, m0_lolp, screen)
    preferred = Float64(screen["m0_lolp_preferred_min"]) <= m0_lolp <=
                Float64(screen["m0_lolp_preferred_max"])
    return (!preferred,
            abs(m0_lolp - Float64(screen["m0_lolp_target"])),
            abs(Float64(candidate["target_rnet"]) - Float64(screen["preferred_target_rnet"])),
            abs(Float64(candidate["wind_penetration"]) -
                Float64(screen["preferred_wind_penetration"])),
            abs(Float64(candidate["lambda_eps"]) - 1.50),
            candidate_key(candidate))
end

function write_csv_atomic(path, rows)
    tmp = path * ".tmp.$(getpid())"
    isempty(rows) && error("refusing to write an empty table: $path")
    field_names = sort!(collect(union((Set(String.(keys(row))) for row in rows)...)))
    normalized = [Dict{String,Any}(name => get(row, name, missing)
                                   for name in field_names) for row in rows]
    CSV.write(tmp, DataFrame(normalized))
    mv(tmp, path; force = true)
    return path
end

function write_toml_atomic(path, record)
    tmp = path * ".tmp.$(getpid())"
    open(tmp, "w") do io
        TOML.print(io, record; sorted = true)
    end
    mv(tmp, path; force = true)
    return path
end

function protocol_inputs(protocol_path)
    protocol = TOML.parsefile(protocol_path)
    screen = protocol["resource_pressure_screen"]
    Bool(screen["validation_only"]) || error("resource-pressure screen must be validation-only")
    String(screen["test_access"]) == "forbidden_until_resource_and_parameter_locks" ||
        error("resource-pressure screen does not forbid test access")
    String.(screen["models"]) == collect(SCREEN_MODELS) ||
        error("resource-pressure screen model order differs from the plan")
    return protocol, screen
end

function pilot(protocol_path)
    protocol, screen = protocol_inputs(protocol_path)
    nstates = Int(screen["pilot_state_count"])
    outdir = joinpath(@__DIR__, "results", "ResourcePressurePilot_$(protocol["case"]["system"])_" *
                      Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    mkpath(outdir)
    all_candidates = declared_candidates(screen)
    all_rows = Dict{String,Any}[]
    contexts = Dict{String,Any}()
    valid_candidates = Dict{String,Any}[]
    invalid = Dict{String,Any}[]

    for candidate in all_candidates
        key = candidate_key(candidate)
        try
            ctx, rows = solve_candidate_models(protocol_path, protocol, screen, candidate,
                                                nstates, ("M0",))
            contexts[key] = ctx
            append!(all_rows, rows)
            push!(valid_candidates, candidate)
        catch err
            push!(invalid, merge(copy(candidate), Dict("candidate" => key,
                "reason" => replace(sprint(showerror, err), r"[\r\n]+" => " "))))
            println(key, ",INVALID,", sprint(showerror, err))
        end
    end
    !isempty(all_rows) && write_csv_atomic(joinpath(outdir, "pilot_m0_partial.csv"), all_rows)

    m0_summaries = Dict{String,Any}[]
    for candidate in valid_candidates
        push!(m0_summaries, summarize_model(all_rows, candidate, "M0", nstates))
    end
    m0_by_key = Dict(row["candidate"] => row for row in m0_summaries)
    eligible = [candidate for candidate in valid_candidates if begin
        ctx = contexts[candidate_key(candidate)]
        m0 = m0_by_key[candidate_key(candidate)]
        Float64(m0["acceptance_rate"]) >= Float64(screen["pilot_minimum_acceptance_rate"]) &&
        ctx.wind_meta.forecast_adequacy_rate >= Float64(screen["minimum_forecast_adequacy_rate"]) &&
        Float64(screen["m0_lolp_min"]) <= Float64(m0["LOLP"]) <= Float64(screen["m0_lolp_max"])
    end]
    sort!(eligible; by = candidate -> pressure_rank(candidate,
        Float64(m0_by_key[candidate_key(candidate)]["LOLP"]), screen))
    shortlist = eligible[1:min(length(eligible), Int(screen["shortlist_count"]))]

    for candidate in shortlist
        _, rows = solve_candidate_models(protocol_path, protocol, screen, candidate,
                                          nstates, ("M2", "M4", "Proposed-RS"))
        append!(all_rows, rows)
        write_csv_atomic(joinpath(outdir, "pilot_long_partial.csv"), all_rows)
    end
    model_summaries = Dict{String,Any}[]
    candidate_summaries = Dict{String,Any}[]
    promising = Dict{String,Any}[]
    for candidate in shortlist
        parts = [summarize_model(all_rows, candidate, model, nstates) for model in SCREEN_MODELS]
        append!(model_summaries, parts)
        row = candidate_summary(candidate, parts, contexts[candidate_key(candidate)], screen,
                                Float64(screen["pilot_minimum_acceptance_rate"]))
        push!(candidate_summaries, row)
        Bool(row["gate_pass"]) && push!(promising, candidate)
    end
    sort!(promising; by = candidate -> pressure_rank(candidate,
        Float64(m0_by_key[candidate_key(candidate)]["LOLP"]), screen))

    write_csv_atomic(joinpath(outdir, "pilot_long.csv"), all_rows)
    shortlist_keys = Set(candidate_key(candidate) for candidate in shortlist)
    unshortlisted_m0 = [row for row in m0_summaries if !(row["candidate"] in shortlist_keys)]
    write_csv_atomic(joinpath(outdir, "pilot_model_summary.csv"),
                     vcat(unshortlisted_m0, model_summaries))
    !isempty(candidate_summaries) &&
        write_csv_atomic(joinpath(outdir, "pilot_candidate_summary.csv"), candidate_summaries)
    !isempty(invalid) && write_csv_atomic(joinpath(outdir, "pilot_invalid_candidates.csv"), invalid)
    provenance = provenance_record(root = ROOT, protocol_path = protocol_path,
        config_path = joinpath(ROOT, "config", "experiment.toml"),
        data_paths = [joinpath(ROOT, String(protocol["case"]["pipeline_cache_dir"]),
                               String(protocol["case"]["system"]), name)
                      for name in ("panel.arrow", "split.toml",
                                   "scenarios_K$(protocol["case"]["K"]).csv")],
        scenario_id = "$(protocol["case"]["system"])--RESOURCE-PILOT--VALIDATION",
        seed = Int(screen["bootstrap_seed"]))
    lock = merge(provenance, Dict{String,Any}(
        "phase" => "pilot", "validation_only" => true, "formal_test_unlocked" => false,
        "pilot_state_count" => nstates, "declared_candidate_count" => length(all_candidates),
        "valid_candidate_count" => length(valid_candidates), "m0_eligible_count" => length(eligible),
        "shortlist" => shortlist, "promising_candidates" => promising,
        "status" => isempty(promising) ? "no_promising_candidate" : "promising_candidates_retained",
        "selection_rule" => String(screen["selection_priority"]),
        "validation_amendment_reason" => String(screen["validation_amendment_reason"]),
        "long_file" => abspath(joinpath(outdir, "pilot_long.csv")),
    ))
    lock_path = write_toml_atomic(joinpath(outdir, "resource_pressure_pilot_lock.toml"), lock)
    println("Pilot lock: ", lock_path)
    isempty(promising) && error("resource-pressure pilot found no candidate that passed the declared gate")
    return lock_path
end

function probe(protocol_path, requested_keys)
    protocol, screen = protocol_inputs(protocol_path)
    isempty(requested_keys) && error("probe requires at least one declared candidate code")
    nstates = Int(screen["pilot_state_count"])
    declared = Dict(candidate_key(candidate) => candidate
                    for candidate in declared_candidates(screen))
    unknown = [key for key in requested_keys if !haskey(declared, key)]
    isempty(unknown) || error("probe contains undeclared candidates: $(join(unknown, ", "))")
    candidates = [declared[key] for key in requested_keys]
    outdir = joinpath(@__DIR__, "results", "ResourcePressureProbe_$(protocol["case"]["system"])_" *
                      Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    mkpath(outdir)
    all_rows = Dict{String,Any}[]
    model_summaries = Dict{String,Any}[]
    candidate_summaries = Dict{String,Any}[]
    promising = Dict{String,Any}[]
    invalid = Dict{String,Any}[]

    for candidate in candidates
        key = candidate_key(candidate)
        try
            ctx, rows = solve_candidate_models(protocol_path, protocol, screen, candidate,
                                                nstates, SCREEN_MODELS)
            append!(all_rows, rows)
            parts = [summarize_model(rows, candidate, model, nstates) for model in SCREEN_MODELS]
            append!(model_summaries, parts)
            summary = candidate_summary(candidate, parts, ctx, screen,
                Float64(screen["pilot_minimum_acceptance_rate"]))
            push!(candidate_summaries, summary)
            Bool(summary["gate_pass"]) && push!(promising, candidate)
            write_csv_atomic(joinpath(outdir, "probe_long_partial.csv"), all_rows)
        catch err
            push!(invalid, merge(copy(candidate), Dict("candidate" => key,
                "reason" => replace(sprint(showerror, err), r"[\r\n]+" => " "))))
            println(key, ",INVALID,", sprint(showerror, err))
        end
    end
    !isempty(all_rows) && write_csv_atomic(joinpath(outdir, "probe_long.csv"), all_rows)
    !isempty(model_summaries) &&
        write_csv_atomic(joinpath(outdir, "probe_model_summary.csv"), model_summaries)
    !isempty(candidate_summaries) &&
        write_csv_atomic(joinpath(outdir, "probe_candidate_summary.csv"), candidate_summaries)
    !isempty(invalid) && write_csv_atomic(joinpath(outdir, "probe_invalid_candidates.csv"), invalid)
    provenance = provenance_record(root = ROOT, protocol_path = protocol_path,
        config_path = joinpath(ROOT, "config", "experiment.toml"),
        data_paths = [joinpath(ROOT, String(protocol["case"]["pipeline_cache_dir"]),
                               String(protocol["case"]["system"]), name)
                      for name in ("panel.arrow", "split.toml",
                                   "scenarios_K$(protocol["case"]["K"]).csv")],
        scenario_id = "$(protocol["case"]["system"])--RESOURCE-PROBE--VALIDATION",
        seed = Int(screen["bootstrap_seed"]))
    lock = merge(provenance, Dict{String,Any}(
        "phase" => "targeted_probe", "validation_only" => true,
        "formal_test_unlocked" => false, "pilot_state_count" => nstates,
        "requested_candidates" => candidates, "promising_candidates" => promising,
        "status" => isempty(promising) ? "no_promising_candidate" :
                    "promising_candidates_retained",
        "selection_rule" => String(screen["selection_priority"]),
        "validation_amendment_reason" => String(screen["validation_amendment_reason"]),
        "long_file" => isempty(all_rows) ? "" : abspath(joinpath(outdir, "probe_long.csv")),
    ))
    lock_path = write_toml_atomic(joinpath(outdir, "resource_pressure_probe_lock.toml"), lock)
    println("Probe lock: ", lock_path)
    return lock_path
end

function write_selected_protocol(protocol_path, selected, confirmation_count, lock_path, outdir)
    protocol = TOML.parsefile(protocol_path)
    protocol["resource_pressure"] = Dict{String,Any}(
        "mode" => "forecast_adequate_realization_fragile",
        "target_wind_penetration" => Float64(selected["wind_penetration"]),
        "lambda_eps" => Float64(selected["lambda_eps"]),
        "replacement_ratio" => Float64(selected["replacement_ratio"]),
        "target_rnet" => Float64(selected["target_rnet"]),
        "rnet_reference_statistic" => "validation_median",
        "rnet_state_count" => confirmation_count,
        "validation_only_pilot" => false,
        "test_access" => "forbidden_until_parameter_and_composite_locks",
        "selection_lock_path" => abspath(lock_path),
    )
    path = joinpath(outdir, "selected_resource_pressure_protocol.toml")
    return write_toml_atomic(path, protocol)
end

function confirm(protocol_path, pilot_lock_path)
    protocol, screen = protocol_inputs(protocol_path)
    pilot_lock = TOML.parsefile(pilot_lock_path)
    String(pilot_lock["protocol_sha256"]) == sha256_file(protocol_path) ||
        error("pilot lock does not belong to this resource-pressure protocol")
    Bool(pilot_lock["validation_only"]) || error("pilot lock is not validation-only")
    candidates = [Dict{String,Any}(String(k) => v for (k, v) in record)
                  for record in pilot_lock["promising_candidates"]]
    isempty(candidates) && error("pilot lock contains no promising candidates")
    nstates = Int(screen["confirmation_state_count"])
    all_rows = Dict{String,Any}[]
    summaries = Dict{String,Any}[]
    contexts = Dict{String,Any}()
    passing = Tuple{Dict{String,Any},Dict{String,Any}}[]
    for candidate in candidates
        ctx, rows = try
            solve_candidate_models(protocol_path, protocol, screen, candidate,
                                   nstates, SCREEN_MODELS)
        catch err
            println(candidate_key(candidate), ",INVALID_CONFIRMATION,", sprint(showerror, err))
            continue
        end
        contexts[candidate_key(candidate)] = ctx
        append!(all_rows, rows)
        parts = [summarize_model(rows, candidate, model, nstates) for model in SCREEN_MODELS]
        row = candidate_summary(candidate, parts, ctx, screen,
                                Float64(screen["confirmation_minimum_acceptance_rate"]))
        push!(summaries, row)
        Bool(row["gate_pass"]) && push!(passing, (candidate, row))
    end
    sort!(passing; by = pair -> pressure_rank(pair[1], Float64(pair[2]["M0_LOLP"]), screen))
    selected = isempty(passing) ? nothing : first(passing)[1]
    outdir = joinpath(@__DIR__, "results", "ResourcePressureConfirm_$(protocol["case"]["system"])_" *
                      Dates.format(now(), dateformat"yyyymmdd_HHMMSS"))
    mkpath(outdir)
    long_path = write_csv_atomic(joinpath(outdir, "confirmation_long.csv"), all_rows)
    summary_path = write_csv_atomic(joinpath(outdir, "confirmation_summary.csv"), summaries)
    lock = Dict{String,Any}(
        "phase" => "confirmation", "validation_only" => true,
        "formal_test_unlocked" => false, "confirmation_state_count" => nstates,
        "pilot_lock_path" => abspath(pilot_lock_path),
        "pilot_lock_sha256" => sha256_file(pilot_lock_path),
        "protocol_path" => abspath(protocol_path), "protocol_sha256" => sha256_file(protocol_path),
        "status" => selected === nothing ? "no_candidate_passed_confirmation" : "resource_pressure_selected",
        "selection_rule" => String(screen["selection_priority"]),
        "long_file" => abspath(long_path), "long_file_sha256" => sha256_file(long_path),
        "summary_file" => abspath(summary_path), "summary_file_sha256" => sha256_file(summary_path),
    )
    selected !== nothing && (lock["selected"] = selected)
    lock_path = joinpath(outdir, "resource_pressure_selection_lock.toml")
    write_toml_atomic(lock_path, lock)
    selected === nothing && error("no resource-pressure candidate passed the 60-state confirmation gate")
    selected_protocol = write_selected_protocol(protocol_path, selected, nstates, lock_path, outdir)
    lock["selected_protocol_path"] = abspath(selected_protocol)
    lock["selected_protocol_sha256"] = sha256_file(selected_protocol)
    write_toml_atomic(lock_path, lock)
    println("Resource-pressure selection lock: ", lock_path)
    println("Selected protocol: ", selected_protocol)
    return selected_protocol
end

function main(args)
    length(args) >= 2 || error(
        "usage: run_resource_pressure_screen.jl protocol.toml pilot | protocol.toml probe CANDIDATE... | protocol.toml confirm pilot_lock.toml")
    protocol_path = normpath(args[1])
    phase = lowercase(args[2])
    phase == "pilot" && length(args) == 2 && return pilot(protocol_path)
    phase == "probe" && length(args) >= 3 && return probe(protocol_path, args[3:end])
    phase == "confirm" && length(args) == 3 && return confirm(protocol_path, normpath(args[3]))
    error("invalid resource-pressure screen invocation")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(copy(ARGS))
end
