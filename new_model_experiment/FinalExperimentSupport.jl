"""Shared, model-faithful support for the final seven-target experiments.

This file contains only experiment setup, frozen-split loading, pressure/wind
scenario construction, acceptance checks, and reporting helpers.  It never
adds a variable, constraint, loss, or objective term to `FinalSevenTargetModel`.
"""
module FinalExperimentSupport

using TOML
using Statistics

using Main.DRCCExp
using Main.DRCCExp: Driver, CaseInterface, DataPipeline
using Main.FinalSevenTargetModel

export FinalContext, load_final_context, frozen_indices, solve_final_snapshot,
       select_two_target_relaxation, select_two_target_weight_scale, audit_solution, accepted_result, status_ok,
       record_snapshot, wind_scaled_panel, replacement_casedata,
       target_wind_scale, apply_dispatchable_profile,
       resource_scenario_code, resource_tag

struct FinalContext
    root::String
    protocol_path::String
    protocol::Dict{String,Any}
    system::String
    pipe
    panel
    cd
    scen
    train::Vector{Int}
    raw_delta::Matrix{Float64}
    raw_omega::Vector{Float64}
    raw_eL::Matrix{Float64}
    P::Dict{String,Any}
    Pfinal::Dict{String,Any}
    rho_tau_c::Float64
    rho_tau_s::Float64
    omega::Vector{Float64}
    wind_meta
end

status_ok(status) = string(status) in ("OPTIMAL", "ALMOST_OPTIMAL")

function numerical_backend_error(err)
    text = sprint(showerror, err)
    return occursin("ZeroPivotException", text) ||
           occursin("zero pivot", lowercase(text)) ||
           occursin("SingularException", text)
end

const RETRYABLE_NUMERICAL_STATUSES =
    ("NUMERICAL_ERROR", "SLOW_PROGRESS", "ITERATION_LIMIT")

function numerical_stage_error(err)
    text = sprint(showerror, err)
    return numerical_backend_error(err) ||
           any(status -> occursin(status, text), RETRYABLE_NUMERICAL_STATUSES)
end

function numerical_stage_status(result)
    return hasproperty(result, :status) &&
           string(getproperty(result, :status)) in RETRYABLE_NUMERICAL_STATUSES
end

"Run one unchanged optimization stage through a deterministic numerical-only
recovery ladder.  The first retry adds the smallest tested static KKT
regularization to QDLDL; CHOLMOD is the final algebraically equivalent backup."
function solve_with_numerical_recovery(operation, base_parameters::Dict)
    profiles = (
        ("base_solver", nothing, nothing),
        ("qdldl_static_1e-7_retry", "qdldl", 1.0e-7),
        ("cholmod_static_1e-7_retry", "cholmod", 1.0e-7),
    )
    last_result = nothing
    last_error = nothing
    for (mode, backend, static_regularization) in profiles
        parameters = copy(base_parameters)
        if !isnothing(backend)
            parameters["max_iter"] = max(Int(get(parameters, "max_iter", 3000)), 10_000)
            parameters["direct_solve_method"] = backend
            parameters["static_regularization_constant"] = static_regularization
        end
        try
            result = operation(parameters)
            last_result = result
            numerical_stage_status(result) || return result, mode
        catch err
            numerical_stage_error(err) || rethrow()
            last_error = err
        end
    end
    !isnothing(last_result) && return last_result, "numerical_retries_exhausted"
    throw(last_error)
end

function accepted_result(result, acceptance::AbstractDict)
    status_ok(result.status) || return false
    residual = Float64(get(acceptance, "residual", 1e-5))
    gap_abs = Float64(get(acceptance, "gap_abs", 5e-5))
    gap_rel = Float64(get(acceptance, "gap_rel", 1e-5))
    all(isfinite, (result.primal_residual, result.dual_residual,
                   result.gap_abs, result.gap_rel)) || return false
    result.primal_residual <= residual || return false
    result.dual_residual <= residual || return false
    return result.gap_abs <= gap_abs || result.gap_rel <= gap_rel
end

function _model_parameters(protocol::Dict{String,Any})
    mcfg = protocol["model"]
    solvercfg = protocol["solver"]
    P = Dict{String,Any}(
        "p" => Int(mcfg["p"]),
        "gamma_c" => Float64(mcfg["gamma_c"]),
        "gamma_m" => Float64(mcfg["gamma_m"]),
        "quad_order" => Int(mcfg["quadrature_order"]),
        "eps_y" => Float64(mcfg["eps_y"]),
    )
    for key in ("lexicographic_secondary_shed", "lexicographic_abs_tol",
                "lexicographic_rel_tol", "lexicographic_gap_multiplier",
                "lexicographic_activation_tol", "eps_curtail",
                "eps_gen_tiebreak", "voll")
        haskey(mcfg, key) && (P[key] = mcfg[key])
    end
    for (key, value) in solvercfg
        P[String(key)] = value
    end
    Pfinal = copy(P)
    for (key, value) in get(protocol, "final_solver", Dict{String,Any}())
        Pfinal[String(key)] = value
    end
    return P, Pfinal
end

"Derive the wind-capacity multiplier for a declared post-replacement share."
function target_wind_scale(cd, target_wind_penetration::Float64,
                           replacement_ratio::Float64 = 1.0)
    0.0 < target_wind_penetration < 1.0 ||
        error("target_wind_penetration must lie strictly between zero and one")
    0.0 <= replacement_ratio <= 1.0 || error("replacement_ratio must lie in [0,1]")
    base_wind = sum(cd.wcap)
    base_dispatchable = sum(cd.pmax)
    base_wind > 0.0 || error("target wind penetration requires positive base wind capacity")
    denominator = base_wind * (1.0 - target_wind_penetration * (1.0 - replacement_ratio))
    denominator > 0.0 || error("target wind penetration has a nonpositive scaling denominator")
    return target_wind_penetration * (base_dispatchable + replacement_ratio * base_wind) /
           denominator
end

"""Return the plan-compliant wind panel while keeping the split frozen.

`wind_scale` controls installed/forecast wind. `lambda_eps` independently
multiplies the original wind forecast error before componentwise physical
clipping. With `lambda_eps == 1`, this is exactly the historical replacement-
wind transformation.
"""
function wind_scaled_panel(panel, rawcase, wind_scale::Float64;
                           lambda_eps::Float64 = 1.0)
    wind_scale >= 1.0 || error("wind_scale must be at least one")
    lambda_eps >= 1.0 || error("lambda_eps must be at least one")
    new_cap = panel.wind_cap .* wind_scale
    caprow = reshape(new_cap, 1, :)
    Wf = clamp.(panel.Wf .* wind_scale, 0.0, caprow)
    original_error = panel.Wact .- panel.Wf
    Wact_unclipped = Wf .+ wind_scale .* lambda_eps .* original_error
    Wact = clamp.(Wact_unclipped, 0.0, caprow)
    transformed = DataPipeline.Panel(panel.system, panel.timestamps,
        panel.load_buses, panel.wind_buses, new_cap, panel.Lf, panel.Lact,
        Wf, Wact, panel.δ, panel.Ω)
    return DataPipeline.build_delta!(transformed, rawcase)
end

"""Apply replacement and optional uniform adequacy derating to the case.

The physical generator vector is retained. `pmax`, the corresponding `pmin`,
and AGC participation are scaled together. An adequacy target may only derate
the post-replacement fleet; an input that would require thermal uprating fails
closed.
"""
function replacement_casedata(cd, wind_scale::Float64;
                              replacement_ratio::Float64 = 1.0,
                              target_dispatchable_pmax_mw::Union{Nothing,Float64} = nothing)
    wind_scale >= 1.0 || error("wind_scale must be at least one")
    0.0 <= replacement_ratio <= 1.0 || error("replacement_ratio must lie in [0,1]")
    new_wcap = cd.wcap .* wind_scale
    incremental_wind_mw = max(sum(new_wcap) - sum(cd.wcap), 0.0)
    replacement_mw = replacement_ratio * incremental_wind_mw
    base_pmax = sum(cd.pmax)
    base_total = sum(cd.wcap) + base_pmax
    post_replacement_pmax = max(base_pmax - replacement_mw, 0.0)
    final_dispatchable_pmax = isnothing(target_dispatchable_pmax_mw) ?
        post_replacement_pmax : Float64(target_dispatchable_pmax_mw)
    final_dispatchable_pmax >= 0.0 || error("target dispatchable capacity must be nonnegative")
    final_dispatchable_pmax <= post_replacement_pmax + 1e-9 * max(1.0, base_pmax) ||
        error("target R_net would require dispatchable-capacity uprating after replacement")
    dispatchable_scale = base_pmax <= eps(Float64) ? 1.0 :
        clamp(final_dispatchable_pmax / base_pmax, 0.0, 1.0)
    pmax = cd.pmax .* dispatchable_scale
    pmin = min.(cd.pmin .* dispatchable_scale, pmax)
    agc_mask = cd.agc_mask .& (pmax .> 1e-9)
    weights = [agc_mask[j] ? max(pmax[j], 0.0) : 0.0 for j in 1:cd.nG]
    βbar = sum(weights) > 0 ? weights ./ sum(weights) : zeros(cd.nG)
    transformed = CaseInterface.CaseData(cd.name, cd.nbus, cd.busidx, cd.nG,
        cd.gen_bus, pmax, pmin, cd.cgen, agc_mask, βbar, cd.nD, cd.load_buses,
        cd.cshed, cd.nW, cd.wind_buses, new_wcap, cd.nE, cd.F_max, cd.PTDF,
        cd.MG, cd.MD, cd.MW)
    total_capacity = sum(new_wcap) + sum(pmax)
    meta = (wind_scale = wind_scale,
        base_wind_cap_mw = sum(cd.wcap), wind_cap_mw = sum(new_wcap),
        base_wind_penetration = sum(cd.wcap) / base_total,
        wind_penetration = sum(new_wcap) / total_capacity,
        replacement_wind_penetration = sum(new_wcap) /
            (sum(new_wcap) + post_replacement_pmax),
        replacement_ratio = replacement_ratio,
        incremental_wind_mw = incremental_wind_mw,
        replacement_mw = replacement_mw,
        base_dispatchable_pmax_mw = base_pmax,
        post_replacement_dispatchable_pmax_mw = post_replacement_pmax,
        dispatchable_pmax_mw = sum(pmax),
        dispatchable_scale = dispatchable_scale,
        adequacy_derating_mw = max(post_replacement_pmax - sum(pmax), 0.0))
    return transformed, meta
end

"Copy a frozen resource-capacity profile onto a case with rebuilt network matrices."
function apply_dispatchable_profile(network_cd, profile_cd)
    network_cd.nG == profile_cd.nG || error("generator dimension changed while rebuilding outage state")
    network_cd.gen_bus == profile_cd.gen_bus || error("generator ordering changed while rebuilding outage state")
    return CaseInterface.CaseData(network_cd.name, network_cd.nbus, network_cd.busidx,
        network_cd.nG, network_cd.gen_bus, copy(profile_cd.pmax), copy(profile_cd.pmin),
        network_cd.cgen, copy(profile_cd.agc_mask), copy(profile_cd.βbar),
        network_cd.nD, network_cd.load_buses, network_cd.cshed, network_cd.nW,
        network_cd.wind_buses, network_cd.wcap, network_cd.nE, network_cd.F_max,
        network_cd.PTDF, network_cd.MG, network_cd.MD, network_cd.MW)
end

_pct_tag(value::Real) = lpad(string(round(Int, 100 * Float64(value))), 2, '0')

"Compact filesystem tag for a frozen resource-pressure scenario."
function resource_tag(meta)
    wind = _pct_tag(get(meta, :target_wind_penetration, meta.wind_penetration))
    uq = lpad(string(round(Int, 100 * get(meta, :lambda_eps, 1.0))), 3, '0')
    rep = lpad(string(round(Int, 100 * get(meta, :replacement_ratio, 1.0))), 3, '0')
    target = Float64(get(meta, :target_rnet, NaN))
    rnet = isfinite(target) ? lpad(string(round(Int, 100 * target)), 2, '0') : "NA"
    fscale = lpad(string(round(Int, 100 * get(meta, :line_limit_scale, 1.0))), 3, '0')
    return "RNET$(rnet)_W$(wind)_UQ$(uq)_REP$(rep)_F$(fscale)"
end

"Extended plan scenario code carried by locks and long-format outputs."
function resource_scenario_code(ctx; seed::Integer, outage::AbstractString = "VAL",
                                split::AbstractString = "SPLIT01")
    return "$(ctx.system)--V2--$(resource_tag(ctx.wind_meta))--$(outage)--$(split)--SEED$(seed)"
end

function load_final_context(protocol_path::AbstractString;
                            wind_scale::Union{Nothing,Float64} = nothing,
                            target_wind_penetration::Union{Nothing,Float64} = nothing,
                            lambda_eps::Union{Nothing,Float64} = nothing,
                            replacement_ratio::Union{Nothing,Float64} = nothing,
                            target_rnet::Union{Nothing,Float64} = nothing,
                            rnet_state_count::Union{Nothing,Int} = nothing,
                            line_limit_scale::Union{Nothing,Float64} = nothing,
                            scenario_cache_dir::Union{Nothing,AbstractString} = nothing,
                            scenario_count::Union{Nothing,Int} = nothing,
                            hmcr_order::Union{Nothing,Int} = nothing,
                            gamma::Union{Nothing,Float64} = nothing,
                            rho_tau_c::Union{Nothing,Float64} = nothing,
                            rho_tau_s::Union{Nothing,Float64} = nothing,
                            omega::Union{Nothing,AbstractVector} = nothing)
    root = normpath(joinpath(@__DIR__, ".."))
    resolved = normpath(protocol_path)
    protocol = TOML.parsefile(resolved)
    casecfg, mcfg = protocol["case"], protocol["model"]
    # Resource pressure is an immutable data/case transformation, not a change
    # to proposed or baseline optimization mathematics. The legacy
    # [wind_replacement] section remains readable for old evidence packages.
    resourcecfg = get(protocol, "resource_pressure", nothing)
    windcfg = get(protocol, "wind_replacement", nothing)
    resourcecfg !== nothing && windcfg !== nothing &&
        error("protocol cannot declare both resource_pressure and legacy wind_replacement")
    if resourcecfg !== nothing
        String(get(resourcecfg, "mode", "")) == "forecast_adequate_realization_fragile" ||
            error("resource_pressure.mode must be forecast_adequate_realization_fragile")
    elseif windcfg !== nothing
        String(get(windcfg, "mode", "")) == "equal_capacity_replacement" ||
            error("wind_replacement.mode must be equal_capacity_replacement")
    end
    system = String(casecfg["system"])
    pipeline_cfg = TOML.parsefile(joinpath(root, "config", "experiment.toml"))
    pipeline_cfg["run"]["cache"] = true
    pipeline_cfg["run"]["cache_dir"] = isnothing(scenario_cache_dir) ?
        String(casecfg["pipeline_cache_dir"]) : String(scenario_cache_dir)
    pipeline_cfg["data"]["compress"]["K"] = isnothing(scenario_count) ?
        Int(casecfg["K"]) : scenario_count
    pipe = Driver.load_or_run_pipeline(system, pipeline_cfg, root)
    base_panel = pipe.panel
    gen_keep = hasproperty(pipe, :gen_keep) ? pipe.gen_keep : nothing
    base_cd = build_casedata(pipe.case, base_panel.load_buses, base_panel.wind_buses,
        base_panel.wind_cap, pipeline_cfg; gen_keep = gen_keep)
    train = Int.(collect(pipe.split.train))

    declared_cfg = resourcecfg === nothing ? windcfg : resourcecfg
    declared_scale = declared_cfg === nothing ? nothing :
        (haskey(declared_cfg, "wind_scale") ? Float64(declared_cfg["wind_scale"]) : nothing)
    declared_penetration = resourcecfg === nothing ? nothing :
        (haskey(resourcecfg, "target_wind_penetration") ?
         Float64(resourcecfg["target_wind_penetration"]) : nothing)
    declared_lambda_eps = resourcecfg === nothing ? 1.0 : Float64(get(resourcecfg, "lambda_eps", 1.0))
    declared_replacement_ratio = resourcecfg === nothing ?
        (windcfg === nothing ? 1.0 : 1.0) : Float64(get(resourcecfg, "replacement_ratio", 1.0))
    declared_target_rnet = resourcecfg === nothing || !haskey(resourcecfg, "target_rnet") ?
        nothing : Float64(resourcecfg["target_rnet"])
    declared_rnet_count = resourcecfg === nothing ? nothing :
        (haskey(resourcecfg, "rnet_state_count") ? Int(resourcecfg["rnet_state_count"]) : nothing)

    function resolve_declared(explicit, declared, label)
        if explicit !== nothing && declared !== nothing &&
           !isapprox(Float64(explicit), Float64(declared); atol = 0.0, rtol = 0.0)
            error("explicit $label conflicts with the protocol-declared resource scenario")
        end
        return explicit === nothing ? declared : explicit
    end
    effective_replacement_ratio = Float64(something(
        resolve_declared(replacement_ratio, resourcecfg === nothing ? nothing : declared_replacement_ratio,
                         "replacement_ratio"), declared_replacement_ratio))
    effective_penetration = resolve_declared(target_wind_penetration, declared_penetration,
                                              "target_wind_penetration")
    derived_scale = effective_penetration === nothing ? nothing :
        target_wind_scale(base_cd, Float64(effective_penetration), effective_replacement_ratio)
    effective_wind_scale = Float64(something(resolve_declared(wind_scale, declared_scale, "wind_scale"),
                                              derived_scale, 1.0))
    if derived_scale !== nothing && !isapprox(effective_wind_scale, derived_scale; atol = 1e-10, rtol = 1e-10)
        error("wind_scale is inconsistent with target_wind_penetration and replacement_ratio")
    end
    effective_lambda_eps = Float64(something(resolve_declared(lambda_eps,
        resourcecfg === nothing ? nothing : declared_lambda_eps, "lambda_eps"), declared_lambda_eps))
    effective_target_rnet = resolve_declared(target_rnet, declared_target_rnet, "target_rnet")
    effective_rnet_count = Int(something(rnet_state_count, declared_rnet_count,
        get(get(protocol, "calibration", Dict{String,Any}()), "validation_state_count", 60)))
    effective_wind_scale >= 1.0 || error("wind_scale must be at least one")
    effective_lambda_eps >= 1.0 || error("lambda_eps must be at least one")
    0.0 <= effective_replacement_ratio <= 1.0 || error("replacement_ratio must lie in [0,1]")

    transformed_resource = effective_wind_scale != 1.0 || effective_lambda_eps != 1.0 ||
        effective_replacement_ratio != 1.0 || effective_target_rnet !== nothing
    resource_panel = transformed_resource ?
        wind_scaled_panel(base_panel, pipe.case, effective_wind_scale;
                          lambda_eps = effective_lambda_eps) : base_panel
    rnet_ids = frozen_indices(pipe.split.val, effective_rnet_count)
    forecast_netload = vec(sum(resource_panel.Lf[rnet_ids, :], dims = 2)) .-
                       vec(sum(resource_panel.Wf[rnet_ids, :], dims = 2))
    positive_netload = filter(>(0.0), forecast_netload)
    isempty(positive_netload) && error("resource-pressure validation states have no positive forecast net load")
    rnet_reference_statistic = resourcecfg === nothing ? "validation_median" :
        String(get(resourcecfg, "rnet_reference_statistic", "validation_median"))
    rnet_reference_statistic == "validation_median" ||
        error("only validation_median is supported for resource_pressure.rnet_reference_statistic")
    rnet_reference_mw = median(positive_netload)
    target_dispatchable = effective_target_rnet === nothing ? nothing :
        (1.0 + Float64(effective_target_rnet)) * rnet_reference_mw

    panel, cd, scen, wind_meta = if !transformed_resource
        base_panel, base_cd, build_scenariodata(pipe.scenarios),
        (wind_scale = 1.0, base_wind_cap_mw = sum(base_cd.wcap),
         wind_cap_mw = sum(base_cd.wcap),
         base_wind_penetration = sum(base_cd.wcap) / (sum(base_cd.wcap) + sum(base_cd.pmax)),
         wind_penetration = sum(base_cd.wcap) / (sum(base_cd.wcap) + sum(base_cd.pmax)),
         replacement_wind_penetration = sum(base_cd.wcap) / (sum(base_cd.wcap) + sum(base_cd.pmax)),
         target_wind_penetration = sum(base_cd.wcap) / (sum(base_cd.wcap) + sum(base_cd.pmax)),
         lambda_eps = 1.0, replacement_ratio = 1.0, incremental_wind_mw = 0.0,
         replacement_mw = 0.0, base_dispatchable_pmax_mw = sum(base_cd.pmax),
         post_replacement_dispatchable_pmax_mw = sum(base_cd.pmax),
         dispatchable_pmax_mw = sum(base_cd.pmax), dispatchable_scale = 1.0,
         adequacy_derating_mw = 0.0, target_rnet = NaN,
         rnet_reference_mw = rnet_reference_mw, rnet_state_count = effective_rnet_count)
    else
        p = resource_panel
        c, meta = replacement_casedata(base_cd, effective_wind_scale;
            replacement_ratio = effective_replacement_ratio,
            target_dispatchable_pmax_mw = target_dispatchable)
        compressed = DataPipeline.compress_scenarios(p, pipe.case, train, pipeline_cfg)
        p, c, build_scenariodata(compressed), merge(meta, (
            target_wind_penetration = effective_penetration === nothing ?
                meta.replacement_wind_penetration : Float64(effective_penetration),
            lambda_eps = effective_lambda_eps,
            target_rnet = effective_target_rnet === nothing ? NaN : Float64(effective_target_rnet),
            rnet_reference_mw = rnet_reference_mw,
            rnet_state_count = effective_rnet_count))
    end
    rnet_denominator = vec(sum(panel.Lf[rnet_ids, :], dims = 2)) .-
                       vec(sum(panel.Wf[rnet_ids, :], dims = 2))
    valid_rnet = rnet_denominator .> 0.0
    rnet_values = (wind_meta.dispatchable_pmax_mw .- rnet_denominator[valid_rnet]) ./
                  rnet_denominator[valid_rnet]
    realized_netload = vec(sum(panel.Lact[rnet_ids, :], dims = 2)) .-
                       vec(sum(panel.Wact[rnet_ids, :], dims = 2))
    forecast_adequate = vec(sum(panel.Lf[rnet_ids, :], dims = 2)) .<=
        wind_meta.dispatchable_pmax_mw .+ vec(sum(panel.Wf[rnet_ids, :], dims = 2)) .+ 1e-9
    wind_meta = merge(wind_meta, (
        rnet_min = minimum(rnet_values), rnet_q25 = quantile(rnet_values, 0.25),
        rnet_median = median(rnet_values), rnet_q75 = quantile(rnet_values, 0.75),
        rnet_max = maximum(rnet_values),
        forecast_adequacy_rate = mean(forecast_adequate),
        realization_inadequacy_rate = mean(realized_netload .> wind_meta.dispatchable_pmax_mw + 1e-9),
        wind_actual_clip_count = count((panel.Wact .<= 1e-12) .|
            (panel.Wact .>= reshape(panel.wind_cap, 1, :) .- 1e-12))))
    fs = isnothing(line_limit_scale) ? Float64(casecfg["line_limit_scale"]) : line_limit_scale
    cd = CaseInterface.scale_Fmax(cd, fs)
    P, Pfinal = _model_parameters(protocol)
    !isnothing(hmcr_order) && (P["p"] = hmcr_order; Pfinal["p"] = hmcr_order)
    if !isnothing(gamma)
        P["gamma_c"] = gamma; P["gamma_m"] = gamma
        Pfinal["gamma_c"] = gamma; Pfinal["gamma_m"] = gamma
    end
    haskey(mcfg, "rho_tau_c") && haskey(mcfg, "rho_tau_s") ||
        error("two-target-relaxation protocol requires model.rho_tau_c and model.rho_tau_s; legacy rho_tau is invalid")
    rho_c = isnothing(rho_tau_c) ? Float64(mcfg["rho_tau_c"]) : rho_tau_c
    rho_s = isnothing(rho_tau_s) ? Float64(mcfg["rho_tau_s"]) : rho_tau_s
    0.0 <= rho_c <= 1.0 || error("rho_tau_c must lie in [0,1]")
    0.0 <= rho_s <= 1.0 || error("rho_tau_s must lie in [0,1]")
    ω = isnothing(omega) ? Float64.(mcfg["omega"]) : Float64.(omega)
    length(ω) == 6 && all(>(0.0), ω) || error("six strictly positive omega values are required")
    raw_delta = Matrix{Float64}(getproperty(panel, Symbol(Char(0x03b4)))[train, :])
    raw_omega = Vector{Float64}(getproperty(panel, Symbol(Char(0x03a9)))[train])
    raw_eL = Matrix{Float64}(panel.Lact[train, :] .- panel.Lf[train, :])
    return FinalContext(root, resolved, protocol, system, pipe, panel, cd, scen, train,
        raw_delta, raw_omega, raw_eL, P, Pfinal, rho_c, rho_s, ω,
        merge(wind_meta, (line_limit_scale = fs,
                          wind_replacement_mode = windcfg === nothing ? "legacy_or_none" : String(windcfg["mode"]),
                          wind_replacement_protocol_declared = windcfg !== nothing,
                          resource_pressure_mode = resourcecfg === nothing ? "legacy_or_none" : String(resourcecfg["mode"]),
                          resource_pressure_protocol_declared = resourcecfg !== nothing)))
end

function frozen_indices(indices, count::Int)
    raw = Int.(collect(indices))
    1 <= count <= length(raw) || error("requested count must lie in 1:$(length(raw))")
    count == 1 && return [first(raw)]
    positions = round.(Int, range(1, length(raw), length = count))
    selected = raw[positions]
    length(unique(selected)) == count || error("frozen index selection duplicated an observation")
    return selected
end

"""Select the two target-relaxation parameters strictly on validation data.
The physical and solver-quality audits are hard gates.  Among admissible pairs,
the predeclared rule minimizes worst flow ratio, mean flow ratio, mean shedding,
then the two parameters lexicographically.  It never accesses test snapshots."""
function select_two_target_relaxation(ctx::FinalContext, validation_snapshots::AbstractVector{<:Integer};
                                      omega::AbstractVector = ctx.omega,
                                      rho_tau_c_grid = get(get(ctx.protocol, "experiment", Dict{String,Any}()),
                                                           "rho_tau_c_grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
                                      rho_tau_s_grid = get(get(ctx.protocol, "experiment", Dict{String,Any}()),
                                                           "rho_tau_s_grid", [0.0, 0.25, 0.5, 0.75, 1.0]),
                                      acceptance::AbstractDict = get(ctx.protocol, "acceptance", Dict{String,Any}()))
    rho_c_values = sort(unique(Float64.(rho_tau_c_grid)))
    rho_s_values = sort(unique(Float64.(rho_tau_s_grid)))
    all(0.0 .<= rho_c_values .<= 1.0) || error("rho_tau_c grid must lie in [0,1]")
    all(0.0 .<= rho_s_values .<= 1.0) || error("rho_tau_s grid must lie in [0,1]")
    length(omega) == 6 && all(>(0.0), omega) || error("six strictly positive omega values are required")

    rows = Dict{String,Any}[]
    mcfg = ctx.protocol["model"]
    for snapshot in Int.(validation_snapshots)
        snap = snapshot_at(ctx.panel, snapshot, ctx.cd)
        pilot = solve_no_smoothing_pilot(ctx.cd, snap, ctx.scen, ctx.P)
        status_ok(pilot.status) || error("validation snapshot=$snapshot pilot status=$(pilot.status)")
        bandwidth = estimate_bandwidths(ctx.cd, snap, ctx.raw_delta, ctx.raw_omega, ctx.raw_eL, pilot.x;
            multiplier = Float64(mcfg["bandwidth_multiplier"]),
            cost_floor = Float64(mcfg["cost_bandwidth_floor_mw"]),
            safety_floor = Float64(mcfg["safety_bandwidth_floor"]))
        base = calibrate_reference_targets(ctx.cd, snap, ctx.scen, ctx.P, bandwidth;
            rho_tau_c = 0.0, rho_tau_s = 0.0)
        for rho_tau_c in rho_c_values, rho_tau_s in rho_s_values
            calibration = retarget(base; rho_tau_c = rho_tau_c, rho_tau_s = rho_tau_s)
            final = solve_final_model(ctx.cd, snap, ctx.scen, ctx.Pfinal, calibration; omega = omega)
            accepted = accepted_result(final, acceptance)
            audit = accepted ? audit_solution(ctx.cd, final, snap) : nothing
            push!(rows, Dict{String,Any}(
                "snapshot" => snapshot,
                "rho_tau_c" => rho_tau_c, "rho_tau_s" => rho_tau_s,
                "status" => string(final.status), "accepted" => accepted,
                "objective" => final.objective, "kappa_c" => final.kappa_c,
                "kappa_m" => final.kappa_m, "Z0c_mw" => base.Z0c,
                "Z0m" => base.Z0m, "tau_c_mw" => calibration.tau_c,
                "tau_m" => calibration.tau_m,
                "primal_residual" => final.primal_residual,
                "dual_residual" => final.dual_residual, "gap_abs" => final.gap_abs,
                "gap_rel" => final.gap_rel, "solve_time_s" => final.solve_time,
                "secondary_objective_mw" => final.secondary_objective,
                "lexicographic_enabled" => final.lexicographic_enabled,
                "lexicographic_activated" => final.lexicographic_activated,
                "lexicographic_stage1_status" => final.lexicographic_stage1_status,
                "lexicographic_stage2_status" => final.lexicographic_stage2_status,
                "lexicographic_primary_star" => final.lexicographic_primary_star,
                "lexicographic_primary_tolerance" => final.lexicographic_primary_tolerance,
                "lexicographic_primary_final" => final.lexicographic_primary_final,
                "stage1_solve_time_s" => final.stage1_solve_time,
                "stage2_solve_time_s" => final.stage2_solve_time,
                "audit_line_violation" => accepted ? audit.line_viol : true,
                "audit_reserve_violation" => accepted ? audit.reserve_viol : true,
                "audit_shedbound_violation" => accepted ? audit.shedbound_viol : true,
                "audit_max_flow_ratio" => accepted ? audit.max_ratio : NaN,
                "audit_shed_mw" => accepted ? audit.shed_mw : NaN,
            ))
        end
    end

    eligible = Dict{String,Any}[]
    expected_count = length(validation_snapshots)
    for rho_tau_c in rho_c_values, rho_tau_s in rho_s_values
        subset = [row for row in rows if row["rho_tau_c"] == rho_tau_c &&
                  row["rho_tau_s"] == rho_tau_s]
        length(subset) == expected_count || continue
        all(row["accepted"] && !row["audit_line_violation"] &&
            !row["audit_reserve_violation"] && !row["audit_shedbound_violation"]
            for row in subset) || continue
        push!(eligible, Dict{String,Any}(
            "rho_tau_c" => rho_tau_c, "rho_tau_s" => rho_tau_s,
            "worst_flow_ratio" => maximum(row["audit_max_flow_ratio"] for row in subset),
            "mean_flow_ratio" => mean(row["audit_max_flow_ratio"] for row in subset),
            "mean_shed_mw" => mean(row["audit_shed_mw"] for row in subset),
        ))
    end
    rule = "Among pairs passing every predeclared solver-quality and physical audit, lexicographically minimize worst flow ratio, mean flow ratio, mean shedding, rho_tau_c, then rho_tau_s."
    if isempty(eligible)
        return (status = "no_candidate_passed", rho_tau_c = NaN, rho_tau_s = NaN,
                validation_rows = rows, eligible_pairs = eligible, selection_rule = rule)
    end
    best = sort(eligible; by = row -> (row["worst_flow_ratio"], row["mean_flow_ratio"],
                                       row["mean_shed_mw"], row["rho_tau_c"], row["rho_tau_s"]))[1]
    return (status = "selected", rho_tau_c = best["rho_tau_c"], rho_tau_s = best["rho_tau_s"],
            validation_rows = rows, eligible_pairs = eligible, selection_rule = rule)
end

"""Jointly select one common six-class safety scale and the two target
relaxations using validation snapshots only.  The ranking first retains only
weight scales with an admissible two-relaxation pair, then applies the same
predeclared flow/shedding lexicographic rule across scales."""
function select_two_target_weight_scale(ctx::FinalContext,
                                        validation_snapshots::AbstractVector{<:Integer};
                                        weight_scales = get(get(ctx.protocol, "experiment", Dict{String,Any}()),
                                                            "weight_scales",
                                                            [1.0, 10.0, 50.0, 100.0, 500.0, 1000.0]),
                                        acceptance::AbstractDict = get(ctx.protocol, "acceptance", Dict{String,Any}()))
    scales = sort(unique(Float64.(weight_scales)))
    !isempty(scales) && all(>(0.0), scales) ||
        error("joint target/weight selection requires positive common safety scales")
    scale_rows = Dict{String,Any}[]
    for scale in scales
        omega = fill(scale, 6)
        selection = select_two_target_relaxation(ctx, validation_snapshots;
            omega = omega, acceptance = acceptance)
        chosen = selection.status == "selected" ?
            [row for row in selection.validation_rows if
             row["rho_tau_c"] == selection.rho_tau_c && row["rho_tau_s"] == selection.rho_tau_s] :
            Dict{String,Any}[]
        push!(scale_rows, Dict{String,Any}(
            "weight_scale" => scale, "omega" => omega,
            "selection_status" => selection.status,
            "selected_rho_tau_c" => selection.rho_tau_c,
            "selected_rho_tau_s" => selection.rho_tau_s,
            "selection_rule" => selection.selection_rule,
            "eligible_pairs" => selection.eligible_pairs,
            "validation_rows" => selection.validation_rows,
            "worst_flow_ratio" => isempty(chosen) ? NaN : maximum(row["audit_max_flow_ratio"] for row in chosen),
            "mean_flow_ratio" => isempty(chosen) ? NaN : mean(row["audit_max_flow_ratio"] for row in chosen),
            "mean_shed_mw" => isempty(chosen) ? NaN : mean(row["audit_shed_mw"] for row in chosen),
        ))
    end
    eligible = [row for row in scale_rows if row["selection_status"] == "selected"]
    rule = "Among common positive safety scales with a two-relaxation pair passing every predeclared solver-quality and physical audit, lexicographically minimize worst flow ratio, mean flow ratio, mean shedding, then safety scale."
    isempty(eligible) && return (status = "no_scale_passed", weight_scale = NaN,
        omega = fill(NaN, 6), rho_tau_c = NaN, rho_tau_s = NaN,
        scale_rows = scale_rows, selection_rule = rule)
    best = sort(eligible; by = row -> (row["worst_flow_ratio"], row["mean_flow_ratio"], row["mean_shed_mw"], row["weight_scale"]))[1]
    return (status = "selected", weight_scale = best["weight_scale"], omega = best["omega"],
        rho_tau_c = best["selected_rho_tau_c"], rho_tau_s = best["selected_rho_tau_s"],
        scale_rows = scale_rows, selection_rule = rule)
end
function audit_solution(cd, result, snap)
    x = (g = result.x.g, rU = result.x.rU, rD = result.x.rD,
         snom = result.x.snom, cW = result.x.cW, β = result.x.beta,
         ρG = 1.0, αS = result.x.alphaS)
    return audit_snapshot(cd, x, snap; βbar = result.x.beta)
end

"Solve the exact calibration, cost-reference comparator, and final model at
one frozen operating snapshot.  No test result is used to alter any setting."
function solve_final_snapshot(ctx::FinalContext, snapshot::Int;
                              rho_tau_c::Float64 = ctx.rho_tau_c,
                              rho_tau_s::Float64 = ctx.rho_tau_s,
                              omega::AbstractVector = ctx.omega,
                              bandwidth_multiplier::Float64 = Float64(
                                  ctx.protocol["model"]["bandwidth_multiplier"]))
    mcfg = ctx.protocol["model"]
    snap = snapshot_at(ctx.panel, snapshot, ctx.cd)
    pilot, pilot_solver_mode = solve_with_numerical_recovery(ctx.P) do parameters
        solve_no_smoothing_pilot(ctx.cd, snap, ctx.scen, parameters)
    end
    status_ok(pilot.status) || error("pilot status=$(pilot.status)")
    bw = estimate_bandwidths(ctx.cd, snap, ctx.raw_delta, ctx.raw_omega, ctx.raw_eL, pilot.x;
        multiplier = bandwidth_multiplier,
        cost_floor = Float64(mcfg["cost_bandwidth_floor_mw"]),
        safety_floor = Float64(mcfg["safety_bandwidth_floor"]))
    cal, calibration_mode = solve_with_numerical_recovery(ctx.P) do parameters
        calibrate_reference_targets(ctx.cd, snap, ctx.scen, parameters, bw;
            rho_tau_c = rho_tau_c, rho_tau_s = rho_tau_s)
    end
    reference, reference_solver_mode = solve_with_numerical_recovery(ctx.P) do parameters
        solve_cost_reference(ctx.cd, snap, ctx.scen, parameters, bw)
    end
    final, final_solver_mode = solve_with_numerical_recovery(ctx.Pfinal) do parameters
        solve_final_model(ctx.cd, snap, ctx.scen, parameters, cal; omega = omega)
    end
    status_ok(reference.status) || error("cost-reference status=$(reference.status)")
    status_ok(final.status) || error("final-model status=$(final.status)")
    return (snapshot = snapshot, snap = snap, pilot = pilot,
            pilot_solver_mode = pilot_solver_mode, bandwidth = bw,
            calibration = cal, calibration_mode = calibration_mode,
            reference_solver_mode = reference_solver_mode,
            final_solver_mode = final_solver_mode,
            reference = reference, final = final,
            reference_audit = audit_solution(ctx.cd, reference, snap),
            final_audit = audit_solution(ctx.cd, final, snap))
end

function record_snapshot(run; acceptance = Dict{String,Any}())
    ref, final, ar, af, cal, bw = run.reference, run.final, run.reference_audit,
        run.final_audit, run.calibration, run.bandwidth
    solver_quality = accepted_result(final, acceptance) && status_ok(ref.status)
    final_physical_audit = !(af.line_viol || af.reserve_viol || af.shedbound_viol)
    reference_physical_audit = !(ar.line_viol || ar.reserve_viol || ar.shedbound_viol)
    reportable = solver_quality && final_physical_audit
    return Dict{String,Any}(
        "snapshot" => run.snapshot,
        "status" => reportable ? "ok" : (solver_quality ? "physical_audit_failed" : "quality_failed"),
        "solver_quality_accepted" => solver_quality,
        "physical_audit_passed" => final_physical_audit,
        "reference_physical_audit_passed" => reference_physical_audit,
        "reportable" => reportable,
        "calibration_solver_mode" => run.calibration_mode,
        "pilot_solver_mode" => run.pilot_solver_mode,
        "reference_solver_mode" => run.reference_solver_mode,
        "final_solver_mode" => run.final_solver_mode,
        "Z0c_mw" => cal.Z0c, "Z0m" => cal.Z0m,
        "rho_tau_c" => cal.rho_tau_c, "rho_tau_s" => cal.rho_tau_s,
        "tau_c_mw" => cal.tau_c, "tau_m" => cal.tau_m,
        "cost_bandwidth_mw" => bw.cost_h, "safety_bandwidth" => bw.safety_h,
        "reference_status" => string(ref.status), "reference_objective" => ref.objective,
        "reference_shed_mw" => ar.shed_mw, "reference_line_violation" => ar.line_viol,
        "reference_reserve_violation" => ar.reserve_viol,
        "reference_shedbound_violation" => ar.shedbound_viol,
        "reference_max_flow_ratio" => ar.max_ratio,
        "final_status" => string(final.status), "final_objective" => final.objective,
        "final_secondary_objective_mw" => final.secondary_objective,
        "kappa_c" => final.kappa_c, "kappa_m" => final.kappa_m,
        "lexicographic_enabled" => final.lexicographic_enabled,
        "lexicographic_activated" => final.lexicographic_activated,
        "lexicographic_stage1_status" => final.lexicographic_stage1_status,
        "lexicographic_stage2_status" => final.lexicographic_stage2_status,
        "lexicographic_primary_star" => final.lexicographic_primary_star,
        "lexicographic_primary_tolerance" => final.lexicographic_primary_tolerance,
        "lexicographic_primary_final" => final.lexicographic_primary_final,
        "final_shed_mw" => af.shed_mw, "final_line_violation" => af.line_viol,
        "final_reserve_violation" => af.reserve_viol,
        "final_shedbound_violation" => af.shedbound_viol,
        "final_max_flow_ratio" => af.max_ratio,
        "reference_solve_time_s" => ref.solve_time, "final_solve_time_s" => final.solve_time,
        "final_stage1_solve_time_s" => final.stage1_solve_time,
        "final_stage2_solve_time_s" => final.stage2_solve_time,
        "final_primal_residual" => final.primal_residual,
        "final_dual_residual" => final.dual_residual,
        "final_gap_abs" => final.gap_abs, "final_gap_rel" => final.gap_rel,
    )
end

end # module
