# =====================================================================
#  Driver.jl  —  模块 6: 实验驱动 (固定样本外主实验)
#
#  遍历 系统 × 模型 × 测试快照: 逐快照求解调度 -> 用该快照实际 δ 审计 ->
#  跨快照聚合 -> 主结果表。所有口径只读 config/experiment.toml。
# =====================================================================
module Driver

using TOML, DataFrames, CSV, Serialization, Statistics, Printf, Random, SHA, StableRNGs
using JuMP, Clarabel
using ..DataPipeline, ..CaseInterface, ..ModelCore, ..Bandwidth, ..Baselines, ..Audit
using ..Cases

export run_fixed_experiment, run_nsmc_experiment, load_or_run_pipeline,
       prepare_target_rs_context, dc_state_precheck,
       eval_model, eval_model_mc, eval_model_mc_screened

_sysdir(root, sys) = joinpath(root, lowercase(sys) == "rts79" ? "RTS_79" :
                                    uppercase(sys) == "RTS_GMLC" ? "RTS_GMLC" : sys)

function _canonical_value(x)
    if x isa AbstractDict
        keys_sorted = sort!(collect(keys(x)); by = string)
        return "{" * join((repr(string(k)) * ":" * _canonical_value(x[k])
                           for k in keys_sorted), ",") * "}"
    elseif x isa AbstractVector
        return "[" * join((_canonical_value(v) for v in x), ",") * "]"
    end
    return repr(x)
end

function _pipeline_source_files(system::String, root::String)
    dir = _sysdir(root, system)
    sysl = lowercase(system)
    if sysl in ("rbts", "rts79")
        case_file = joinpath(dir, sysl == "rbts" ? "RBTS.txt" : "RTS79_case.txt")
        wind_files = sort(filter(f -> occursin("Wind farm site", f), readdir(dir)))
        return vcat([case_file, joinpath(dir, "BUS_LOADS.csv")],
                    [joinpath(dir, f) for f in wind_files])
    elseif sysl in ("rts_gmlc", "gmlc")
        rts = joinpath(dir, "RTS_Data")
        return [
            joinpath(rts, "FormattedData", "MATPOWER", "RTS_GMLC.m"),
            joinpath(rts, "SourceData", "bus.csv"),
            joinpath(rts, "SourceData", "gen.csv"),
            joinpath(rts, "timeseries_data_files", "Load", "DAY_AHEAD_regional_Load.csv"),
            joinpath(rts, "timeseries_data_files", "Load", "REAL_TIME_regional_Load.csv"),
            joinpath(rts, "timeseries_data_files", "WIND", "DAY_AHEAD_wind.csv"),
            joinpath(rts, "timeseries_data_files", "WIND", "REAL_TIME_wind.csv"),
        ]
    end
    return String[]
end

function _pipeline_signature(system::String, cfg::Dict, root::String)
    wind_cfg = get(cfg["network"]["wind_bus"], uppercase(system), Dict{String,Any}())
    relevant = Dict(
        "pipeline_version" => DataPipeline.PIPELINE_VERSION,
        "system" => uppercase(system),
        "run_seed" => cfg["run"]["seed"],
        "data" => cfg["data"],
        "wind_bus" => wind_cfg,
    )
    file_hashes = String[]
    for path in _pipeline_source_files(system, root)
        push!(file_hashes,
              isfile(path) ? "$(normpath(path))=$(bytes2hex(sha256(read(path))))" :
                             "$(normpath(path))=MISSING")
    end
    payload = _canonical_value(relevant) * "\n" * join(file_hashes, "\n")
    return bytes2hex(sha256(payload))
end

function _target_log_fields(tm)
    tm === nothing && return (
        Z0c = NaN, Z0m1 = NaN, Z0m2 = NaN, Z0m3 = NaN,
        Z0m4 = NaN, Z0m5 = NaN, Z0m6 = NaN,
        tau_c = NaN, tau_m1 = NaN, tau_m2 = NaN, tau_m3 = NaN,
        tau_m4 = NaN, tau_m5 = NaN, tau_m6 = NaN,
        k_m1 = NaN, k_m2 = NaN, k_m3 = NaN,
        k_m4 = NaN, k_m5 = NaN, k_m6 = NaN,
        target_k_rule = "",
        target_payoff_anchor_count = -1,
        target_payoff_scale_source = "",
        target_payoff_zero_range_policy = "",
        target_payoff_evaluation_method = "",
        target_payoff_anchor_statuses = "",
        target_payoff_evaluation_statuses = "",
        target_payoff_anchor_build_time = NaN,
        target_payoff_anchor_solve_time = NaN,
        target_payoff_evaluation_build_time = NaN,
        target_payoff_evaluation_solve_time = NaN,
        target_payoff_evaluation_primal_residual_max = NaN,
        target_payoff_evaluation_dual_residual_max = NaN,
        target_payoff_evaluation_gap_min_max = NaN,
        target_payoff_ideal1 = NaN, target_payoff_ideal2 = NaN,
        target_payoff_ideal3 = NaN, target_payoff_ideal4 = NaN,
        target_payoff_ideal5 = NaN, target_payoff_ideal6 = NaN,
        target_payoff_ideal7 = NaN,
        target_payoff_anchor_maximum1 = NaN, target_payoff_anchor_maximum2 = NaN,
        target_payoff_anchor_maximum3 = NaN, target_payoff_anchor_maximum4 = NaN,
        target_payoff_anchor_maximum5 = NaN, target_payoff_anchor_maximum6 = NaN,
        target_payoff_anchor_maximum7 = NaN,
        target_payoff_range1 = NaN, target_payoff_range2 = NaN,
        target_payoff_range3 = NaN, target_payoff_range4 = NaN,
        target_payoff_range5 = NaN, target_payoff_range6 = NaN,
        target_payoff_range7 = NaN,
        target_k_priority1 = NaN, target_k_priority2 = NaN,
        target_k_priority3 = NaN, target_k_priority4 = NaN,
        target_k_priority5 = NaN, target_k_priority6 = NaN,
        target_k_cost_scale_reference = NaN,
        target_k_safety_scale1 = NaN, target_k_safety_scale2 = NaN,
        target_k_safety_scale3 = NaN, target_k_safety_scale4 = NaN,
        target_k_safety_scale5 = NaN, target_k_safety_scale6 = NaN,
        target_k_cost_sigma_proxy = NaN,
        target_k_safety_sigma1 = NaN, target_k_safety_sigma2 = NaN,
        target_k_safety_sigma3 = NaN, target_k_safety_sigma4 = NaN,
        target_k_safety_sigma5 = NaN, target_k_safety_sigma6 = NaN,
        target_k_cost_resolution = NaN,
        target_k_safety_resolution1 = NaN, target_k_safety_resolution2 = NaN,
        target_k_safety_resolution3 = NaN, target_k_safety_resolution4 = NaN,
        target_k_safety_resolution5 = NaN, target_k_safety_resolution6 = NaN,
        rho_tau = NaN, omega_d = NaN,
        target_rule_mode = "", target_tau_rule = "",
        target_cost_scale_rule = "", target_cost_scale = NaN,
        target_cost_coordinate = "", target_kappa_coordinate = "",
        target_gamma = NaN, target_p = -1,
        target_cost_loss = "", target_bandwidth_pilot = "",
        target_bandwidth_raw_source = "",
        target_cost_h = NaN, target_cost_h_normalized = NaN,
        target_h1 = NaN, target_h2 = NaN, target_h3 = NaN,
        target_h4 = NaN, target_h5 = NaN, target_h6 = NaN,
        target_h_min = NaN, target_h_max = NaN,
        target_bandwidth_raw_n = -1,
        target_bandwidth_multiplier = NaN,
        target_bandwidth_relative_floor = NaN,
        target_bandwidth_floor_active = missing,
        target_quad_rule = "", target_quad_nodes = -1,
        target_anchor_safety_max = NaN,
        target_anchor_statuses = "", target_pilot_status = "",
        target_pilot_attempt_statuses = "", target_pilot_retry_used = false,
        target_calibration_attempt_statuses = "",
        target_calibration_retry_used = false,
        target_pilot_build_time = NaN, target_pilot_solve_time = NaN,
        target_context_build_time = NaN, target_context_solve_time = NaN,
        target_lex_enabled = false,
        target_lex_activation_tol = NaN,
        target_lex_activated = false,
        target_lex_stage1_status = "", target_lex_stage2_status = "",
        target_lex_primary_star = NaN,
        target_lex_primary_tolerance = NaN,
        target_lex_primary_final = NaN,
        target_lex_secondary_final = NaN,
        target_lex_stage1_gap_abs = NaN, target_lex_stage1_gap_rel = NaN,
        target_lex_stage2_gap_abs = NaN, target_lex_stage2_gap_rel = NaN,
        target_primal_residual_max = NaN,
        target_dual_residual_max = NaN,
        target_gap_min_max = NaN,
    )
    getmeta(name::Symbol, default) = hasproperty(tm, name) ? getproperty(tm, name) : default
    target_h = getmeta(:target_h, fill(NaN, 6))
    length(target_h) == 6 || error("target metadata must contain six safety bandwidths")
    k_priority = Float64.(collect(getmeta(:k_priority, fill(NaN, 6))))
    k_safety_scale = Float64.(collect(getmeta(
        :k_safety_scale_reference, fill(NaN, 6))))
    k_safety_sigma = Float64.(collect(getmeta(
        :k_safety_sigma_proxy, fill(NaN, 6))))
    k_safety_resolution = Float64.(collect(getmeta(
        :k_safety_resolution, fill(NaN, 6))))
    all(length(v) == 6 for v in
        (k_priority, k_safety_scale, k_safety_sigma, k_safety_resolution)) ||
        error("target k metadata must contain six values per safety field")
    cost_scale = Float64(getmeta(:cost_scale, NaN))
    cost_h = Float64(getmeta(:cost_h, NaN))
    cost_h_normalized = Float64(getmeta(
        :cost_h_normalized,
        isfinite(cost_h) && isfinite(cost_scale) && cost_scale > 0 ?
            cost_h / cost_scale : NaN))
    statuses = getmeta(:statuses, String[])
    payoff_ideal = Float64.(collect(getmeta(:payoff_ideal, fill(NaN, 7))))
    payoff_anchor_maximum = Float64.(collect(getmeta(
        :payoff_anchor_maximum, fill(NaN, 7))))
    payoff_ranges = Float64.(collect(getmeta(:payoff_ranges, fill(NaN, 7))))
    payoff_anchor_build_times = Float64.(collect(getmeta(
        :payoff_anchor_build_times, fill(NaN, 7))))
    payoff_anchor_solve_times = Float64.(collect(getmeta(
        :payoff_anchor_solve_times, fill(NaN, 7))))
    payoff_evaluation_build_times = Float64.(collect(getmeta(
        :payoff_evaluation_build_times, fill(NaN, 7))))
    payoff_evaluation_solve_times = Float64.(collect(getmeta(
        :payoff_evaluation_solve_times, fill(NaN, 7))))
    payoff_eval_primal = Float64.(collect(getmeta(
        :payoff_evaluation_primal_residuals, fill(NaN, 7))))
    payoff_eval_dual = Float64.(collect(getmeta(
        :payoff_evaluation_dual_residuals, fill(NaN, 7))))
    payoff_eval_gap_abs = Float64.(collect(getmeta(
        :payoff_evaluation_gap_abs_values, fill(NaN, 7))))
    payoff_eval_gap_rel = Float64.(collect(getmeta(
        :payoff_evaluation_gap_rel_values, fill(NaN, 7))))
    all(length(v) == 7 for v in (
        payoff_ideal, payoff_anchor_maximum, payoff_ranges,
        payoff_anchor_build_times, payoff_anchor_solve_times,
        payoff_evaluation_build_times, payoff_evaluation_solve_times,
        payoff_eval_primal, payoff_eval_dual, payoff_eval_gap_abs,
        payoff_eval_gap_rel)) || error(
        "payoff target metadata must contain seven values per payoff field")
    payoff_evaluation_statuses = getmeta(:payoff_evaluation_statuses, String[])
    return (
        Z0c = tm.Z0c,
        Z0m1 = tm.Z0m[1], Z0m2 = tm.Z0m[2], Z0m3 = tm.Z0m[3],
        Z0m4 = tm.Z0m[4], Z0m5 = tm.Z0m[5], Z0m6 = tm.Z0m[6],
        tau_c = tm.τc,
        tau_m1 = tm.τm[1], tau_m2 = tm.τm[2], tau_m3 = tm.τm[3],
        tau_m4 = tm.τm[4], tau_m5 = tm.τm[5], tau_m6 = tm.τm[6],
        k_m1 = tm.k[1], k_m2 = tm.k[2], k_m3 = tm.k[3],
        k_m4 = tm.k[4], k_m5 = tm.k[5], k_m6 = tm.k[6],
        target_k_rule = String(getmeta(:k_rule, "")),
        target_payoff_anchor_count = Int(getmeta(:payoff_anchor_count, -1)),
        target_payoff_scale_source = String(getmeta(:payoff_scale_source, "")),
        target_payoff_zero_range_policy = String(getmeta(
            :payoff_zero_range_policy, "")),
        target_payoff_evaluation_method = String(getmeta(
            :payoff_evaluation_method, "")),
        target_payoff_anchor_statuses = join(string.(statuses), ";"),
        target_payoff_evaluation_statuses = join(
            string.(payoff_evaluation_statuses), ";"),
        target_payoff_anchor_build_time = sum(payoff_anchor_build_times),
        target_payoff_anchor_solve_time = sum(payoff_anchor_solve_times),
        target_payoff_evaluation_build_time = sum(payoff_evaluation_build_times),
        target_payoff_evaluation_solve_time = sum(payoff_evaluation_solve_times),
        target_payoff_evaluation_primal_residual_max = maximum(payoff_eval_primal),
        target_payoff_evaluation_dual_residual_max = maximum(payoff_eval_dual),
        target_payoff_evaluation_gap_min_max = maximum(min.(
            abs.(payoff_eval_gap_abs), abs.(payoff_eval_gap_rel))),
        target_payoff_ideal1 = payoff_ideal[1], target_payoff_ideal2 = payoff_ideal[2],
        target_payoff_ideal3 = payoff_ideal[3], target_payoff_ideal4 = payoff_ideal[4],
        target_payoff_ideal5 = payoff_ideal[5], target_payoff_ideal6 = payoff_ideal[6],
        target_payoff_ideal7 = payoff_ideal[7],
        target_payoff_anchor_maximum1 = payoff_anchor_maximum[1],
        target_payoff_anchor_maximum2 = payoff_anchor_maximum[2],
        target_payoff_anchor_maximum3 = payoff_anchor_maximum[3],
        target_payoff_anchor_maximum4 = payoff_anchor_maximum[4],
        target_payoff_anchor_maximum5 = payoff_anchor_maximum[5],
        target_payoff_anchor_maximum6 = payoff_anchor_maximum[6],
        target_payoff_anchor_maximum7 = payoff_anchor_maximum[7],
        target_payoff_range1 = payoff_ranges[1], target_payoff_range2 = payoff_ranges[2],
        target_payoff_range3 = payoff_ranges[3], target_payoff_range4 = payoff_ranges[4],
        target_payoff_range5 = payoff_ranges[5], target_payoff_range6 = payoff_ranges[6],
        target_payoff_range7 = payoff_ranges[7],
        target_k_priority1 = k_priority[1], target_k_priority2 = k_priority[2],
        target_k_priority3 = k_priority[3], target_k_priority4 = k_priority[4],
        target_k_priority5 = k_priority[5], target_k_priority6 = k_priority[6],
        target_k_cost_scale_reference = Float64(getmeta(
            :k_cost_scale_reference, NaN)),
        target_k_safety_scale1 = k_safety_scale[1],
        target_k_safety_scale2 = k_safety_scale[2],
        target_k_safety_scale3 = k_safety_scale[3],
        target_k_safety_scale4 = k_safety_scale[4],
        target_k_safety_scale5 = k_safety_scale[5],
        target_k_safety_scale6 = k_safety_scale[6],
        target_k_cost_sigma_proxy = Float64(getmeta(:k_cost_sigma_proxy, NaN)),
        target_k_safety_sigma1 = k_safety_sigma[1],
        target_k_safety_sigma2 = k_safety_sigma[2],
        target_k_safety_sigma3 = k_safety_sigma[3],
        target_k_safety_sigma4 = k_safety_sigma[4],
        target_k_safety_sigma5 = k_safety_sigma[5],
        target_k_safety_sigma6 = k_safety_sigma[6],
        target_k_cost_resolution = Float64(getmeta(:k_cost_resolution, NaN)),
        target_k_safety_resolution1 = k_safety_resolution[1],
        target_k_safety_resolution2 = k_safety_resolution[2],
        target_k_safety_resolution3 = k_safety_resolution[3],
        target_k_safety_resolution4 = k_safety_resolution[4],
        target_k_safety_resolution5 = k_safety_resolution[5],
        target_k_safety_resolution6 = k_safety_resolution[6],
        rho_tau = tm.rho_tau, omega_d = tm.omega_d,
        target_rule_mode = String(getmeta(:rule_mode, "legacy")),
        target_tau_rule = String(getmeta(:tau_rule, "")),
        target_cost_scale_rule = String(getmeta(:cost_scale_rule, "")),
        target_cost_scale = cost_scale,
        target_cost_coordinate = String(getmeta(:cost_coordinate, "")),
        target_kappa_coordinate = String(getmeta(:kappa_coordinate, "")),
        target_gamma = Float64(getmeta(:target_gamma, NaN)),
        target_p = Int(getmeta(:target_p, -1)),
        target_cost_loss = String(getmeta(:cost_loss, "")),
        target_bandwidth_pilot = String(getmeta(:bandwidth_pilot, "")),
        target_bandwidth_raw_source = String(getmeta(:bandwidth_raw_source, "")),
        target_cost_h = cost_h,
        target_cost_h_normalized = cost_h_normalized,
        target_h1 = target_h[1], target_h2 = target_h[2],
        target_h3 = target_h[3], target_h4 = target_h[4],
        target_h5 = target_h[5], target_h6 = target_h[6],
        target_h_min = all(isfinite, target_h) ? minimum(target_h) : NaN,
        target_h_max = all(isfinite, target_h) ? maximum(target_h) : NaN,
        target_bandwidth_raw_n = Int(getmeta(:bandwidth_raw_n, -1)),
        target_bandwidth_multiplier = Float64(getmeta(:bandwidth_multiplier, NaN)),
        target_bandwidth_relative_floor = Float64(getmeta(:bandwidth_relative_floor, NaN)),
        target_bandwidth_floor_active = getmeta(:bandwidth_floor_active, missing),
        target_quad_rule = String(getmeta(:quad_rule, "")),
        target_quad_nodes = Int(getmeta(:quad_nodes, -1)),
        target_anchor_safety_max = Float64(getmeta(:anchor_safety_max, NaN)),
        target_anchor_statuses = join(string.(statuses), ";"),
        target_pilot_status = String(getmeta(:pilot_status, "")),
        target_pilot_attempt_statuses = join(string.(getmeta(
            :pilot_attempt_statuses, String[])), "=>"),
        target_pilot_retry_used = Bool(getmeta(:pilot_retry_used, false)),
        target_calibration_attempt_statuses = join(string.(getmeta(
            :calibration_attempt_statuses, String[])), "=>"),
        target_calibration_retry_used = Bool(getmeta(
            :calibration_retry_used, false)),
        target_pilot_build_time = Float64(getmeta(:pilot_build_time, NaN)),
        target_pilot_solve_time = Float64(getmeta(:pilot_solve_time, NaN)),
        target_context_build_time = Float64(getmeta(:context_build_time, NaN)),
        target_context_solve_time = Float64(getmeta(:context_solve_time, NaN)),
        target_lex_enabled = Bool(getmeta(:lexicographic_enabled, false)),
        target_lex_activation_tol = Float64(getmeta(
            :lexicographic_activation_tolerance, NaN)),
        target_lex_activated = Bool(getmeta(:lexicographic_activated, false)),
        target_lex_stage1_status = String(getmeta(:lexicographic_stage1_status, "")),
        target_lex_stage2_status = String(getmeta(:lexicographic_stage2_status, "")),
        target_lex_primary_star = Float64(getmeta(:lexicographic_primary_star, NaN)),
        target_lex_primary_tolerance = Float64(getmeta(
            :lexicographic_primary_tolerance, NaN)),
        target_lex_primary_final = Float64(getmeta(:lexicographic_primary_final, NaN)),
        target_lex_secondary_final = Float64(getmeta(
            :lexicographic_secondary_final, NaN)),
        target_lex_stage1_gap_abs = Float64(getmeta(:lexicographic_stage1_gap_abs, NaN)),
        target_lex_stage1_gap_rel = Float64(getmeta(:lexicographic_stage1_gap_rel, NaN)),
        target_lex_stage2_gap_abs = Float64(getmeta(:lexicographic_stage2_gap_abs, NaN)),
        target_lex_stage2_gap_rel = Float64(getmeta(:lexicographic_stage2_gap_rel, NaN)),
        target_primal_residual_max = maximum(tm.primal_residuals),
        target_dual_residual_max = maximum(tm.dual_residuals),
        target_gap_min_max = maximum(min.(abs.(tm.gap_abs_values),
                                            abs.(tm.gap_rel_values))),
    )
end

function _validate_target_pilot(pilot_x)
    required = (:g, :rU, :rD, :snom, :cW, Symbol("β"), Symbol("αS"))
    for field in required
        hasproperty(pilot_x, field) || error("target bandwidth pilot is missing field $field")
        all(isfinite, getproperty(pilot_x, field)) ||
            error("target bandwidth pilot contains a non-finite value in $field")
    end
    return pilot_x
end

function _target_reference_accepted(status, primal, dual, gap_abs, gap_rel, P::Dict)
    string(status) in ("OPTIMAL", "ALMOST_OPTIMAL") || return false
    all(isfinite, (primal, dual, gap_abs, gap_rel)) || return false
    residual_tol = Float64(get(P, "solver_accept_scaled_residual", 1e-5))
    gap_abs_tol = Float64(get(P, "solver_accept_gap_abs",
                              get(P, "solver_accept_gap", 5e-5)))
    gap_rel_tol = Float64(get(P, "solver_accept_gap_rel",
                              get(P, "solver_accept_gap", 1e-5)))
    return max(primal, dual) <= residual_tol &&
           (abs(gap_abs) <= gap_abs_tol || abs(gap_rel) <= gap_rel_tol)
end

function _target_reference_refined_parameters(P::Dict)
    refined = deepcopy(P)
    refined["solver_direct_solve_method"] = String(get(
        P, "target_reference_retry_direct_solve_method", "qdldl"))
    refined["solver_equilibrate_max_iter"] = Int(get(
        P, "target_reference_retry_equilibrate_max_iter",
        get(P, "solver_retry_equilibrate_max_iter", 20)))
    refined["solver_iterative_refinement_max_iter"] = Int(get(
        P, "target_reference_retry_iterative_refinement_max_iter",
        get(P, "solver_retry_iterative_refinement_max_iter", 20)))
    return refined
end

_target_reference_retry_enabled(P::Dict) = Bool(get(
    P, "target_reference_retry_refined", true))

"""
Validate the auditable metadata specific to the opt-in anchor-range rule and
return its recomputed scales.  The range is deliberately derived from the
stored table every time; a saved `k` alone is not treated as evidence.
"""
function _validated_payoff_range_metadata(meta)
    required = (:payoff_matrix, :payoff_ideal, :payoff_anchor_maximum,
                :payoff_ranges, :payoff_anchor_count, :payoff_scale_source,
                :payoff_zero_range_policy, :payoff_evaluation_method)
    all(name -> hasproperty(meta, name), required) || error(
        "payoff-range target metadata is missing provenance fields")
    Int(meta.payoff_anchor_count) == 7 || error(
        "payoff-range target metadata must contain seven predeclared anchors")
    String(meta.payoff_scale_source) == "predeclared_anchor_payoff_range" || error(
        "payoff-range target metadata has an unknown scale source")
    String(meta.payoff_zero_range_policy) == "reject_no_epsilon" || error(
        "payoff-range target metadata must reject zero ranges without epsilon")
    String(meta.payoff_evaluation_method) == "joint_fixed_x_sum_epigraph" || error(
        "payoff-range target metadata has an unknown row-evaluation method")
    scale = ModelCore._payoff_range_scale_coefficients(meta.payoff_matrix)
    all(isapprox.(Float64.(meta.payoff_ideal), scale.payoff_ideal;
                  rtol = 1e-12, atol = 1e-12)) || error(
        "payoff-range target metadata ideal values do not match its matrix")
    all(isapprox.(Float64.(meta.payoff_anchor_maximum),
                  scale.payoff_anchor_maximum;
                  rtol = 1e-12, atol = 1e-12)) || error(
        "payoff-range target metadata anchor maxima do not match its matrix")
    all(isapprox.(Float64.(meta.payoff_ranges), scale.payoff_ranges;
                  rtol = 1e-12, atol = 1e-12)) || error(
        "payoff-range target metadata ranges do not match its matrix")
    hasproperty(meta, :k) && length(meta.k) == 6 &&
        all(isapprox.(Float64.(meta.k), scale.k;
                      rtol = 1e-12, atol = 1e-12)) || error(
        "payoff-range target k coefficients do not match its ranges")
    values = Float64.(Matrix(meta.payoff_matrix))
    row_one = vec(values[1, :])
    isapprox(Float64(meta.Z0c), row_one[1]; rtol = 1e-8, atol = 1e-10) || error(
        "payoff-range cost target does not match the cost-anchor payoff row")
    all(isapprox.(Float64.(meta.Z0m), row_one[2:end];
                  rtol = 1e-8, atol = 1e-10)) || error(
        "payoff-range safety targets do not match the cost-anchor payoff row")
    return scale
end

function _payoff_evaluation_quality_accepted(meta, P::Dict)
    names = (:payoff_evaluation_statuses,
             :payoff_evaluation_primal_residuals,
             :payoff_evaluation_dual_residuals,
             :payoff_evaluation_gap_abs_values,
             :payoff_evaluation_gap_rel_values)
    all(name -> hasproperty(meta, name), names) || return false
    nrows = length(meta.payoff_evaluation_statuses)
    nrows == 7 || return false
    all(length(getproperty(meta, name)) == nrows for name in names[2:end]) ||
        return false
    return all(i -> _target_reference_accepted(
        meta.payoff_evaluation_statuses[i],
        meta.payoff_evaluation_primal_residuals[i],
        meta.payoff_evaluation_dual_residuals[i],
        meta.payoff_evaluation_gap_abs_values[i],
        meta.payoff_evaluation_gap_rel_values[i], P), 1:nrows)
end

function _target_calibration_accepted(meta, P::Dict)
    anchors_ok = all(i -> _target_reference_accepted(
        meta.statuses[i], meta.primal_residuals[i], meta.dual_residuals[i],
        meta.gap_abs_values[i], meta.gap_rel_values[i], P),
        eachindex(meta.statuses)) && all(isfinite, [meta.Z0c; meta.Z0m])
    anchors_ok || return false
    rule = hasproperty(meta, :k_rule) ? String(meta.k_rule) : ""
    if rule == "payoff_range_no_eps"
        _payoff_evaluation_quality_accepted(meta, P) || return false
        try
            _validated_payoff_range_metadata(meta)
        catch
            return false
        end
    end
    return true
end

"""
    prepare_target_rs_context(cd, snap, scen, P, delta_raw, omega_raw, eL_raw;
                              pilot_x=nothing, calibrate=true,
                              state_index=nothing, draw_index=nothing)

Prepare the frozen bandwidth profile and calibrated target metadata required by
the `common_dimensionless` seven-target M6 mode.  All seven bandwidths are
estimated from one common pilot and the complete, uncompressed training sample;
the cost sample is the same per-load positive-part loss used by `ModelCore`.

For the default `legacy` mode this function is a no-op, so existing experiment
paths retain their historical calibration behavior.  In common mode it fails
closed on missing/mismatched raw samples or a failed pilot instead of silently
falling back to representative-scenario bandwidths.  Set `calibrate=false` to
return the validated common profile without running the seven M6 reference
problems; this lets M3 proceed even when M6 calibration is unavailable.  Public
evaluation calls require `state_index`, and additionally `draw_index` whenever
`case_by_draw` is used, so contexts cannot be silently paired with the wrong
operating state.
"""
function prepare_target_rs_context(cd::CaseData, snap::Snapshot,
                                   scen::ScenarioData, P::Dict,
                                   delta_raw::AbstractMatrix,
                                   omega_raw::AbstractVector,
                                   eL_raw::AbstractMatrix;
                                   pilot_x = nothing,
                                   calibrate::Bool = true,
                                   state_index = nothing,
                                   draw_index = nothing)
    mode = ModelCore._target_rs_mode(P)
    mode == "common_dimensionless" || return (
        rule_mode = mode, bandwidth_profile = nothing, target_meta = nothing,
        state_index = state_index, draw_index = draw_index,
        pilot_x = nothing, pilot_status = "NOT_REQUIRED",
        pilot_build_time = 0.0, pilot_solve_time = 0.0,
        profile_build_time = 0.0,
        calibration_build_time = 0.0, calibration_solve_time = 0.0,
        context_build_time = 0.0, context_solve_time = 0.0)

    ModelCore._validate_common_settings(P)
    nraw = size(delta_raw, 1)
    nraw == scen.raw_n || error(
        "common target bandwidths require all raw training samples: " *
        "received $nraw rows, expected scen.raw_n=$(scen.raw_n)")
    length(omega_raw) == nraw || error("delta_raw/omega_raw row mismatch")
    size(eL_raw, 1) == nraw || error("delta_raw/eL_raw row mismatch")
    size(delta_raw, 2) == cd.nbus || error("raw delta bus dimension mismatch")
    size(eL_raw, 2) == cd.nD || error("raw load-error dimension mismatch")
    all(isfinite, delta_raw) || error("delta_raw contains non-finite values")
    all(isfinite, omega_raw) || error("omega_raw contains non-finite values")
    all(isfinite, eL_raw) || error("eL_raw contains non-finite values")
    omega_check = vec(sum(delta_raw, dims = 2))
    all(isapprox.(omega_raw, omega_check; rtol = 1e-10, atol = 1e-8)) ||
        error("omega_raw must be the row sum of the same delta_raw sample")

    gamma = ModelCore._common_target_gamma(P)
    pilot_status = "SUPPLIED"
    pilot_attempt_statuses = String["SUPPLIED"]
    pilot_retry_used = false
    pilot_build_time = 0.0
    pilot_solve_time = 0.0
    if pilot_x === nothing
        # Break the pilot/bandwidth circularity with one common no-smoothing
        # cost-feasible anchor.  This temporary copy cannot activate the new
        # common branch and therefore does not require a provisional profile.
        pilot_P = deepcopy(P)
        pilot_P["target_rs_mode"] = "legacy"
        pilot_P["gamma_c"] = gamma
        pilot_P["gamma_m"] = gamma
        pilot_P["eps_m"] = 1.0 - gamma
        pilot_P["cost_kde_mode"] = "off"
        pilot_P["kde_mode"] = "off"
        pilot_P["rho_tau"] = 0.0
        pilot_P["omega_d"] = 1.0
        _, pilot = ModelCore._solve_reference_target(cd, snap, scen, pilot_P, :cost)
        pilot_status = string(pilot.status)
        pilot_attempt_statuses = String[pilot_status]
        pilot_build_time = Float64(pilot.build_time)
        pilot_solve_time = Float64(pilot.solve_time)
        pilot_ok = _target_reference_accepted(
            pilot.status, pilot.primal_residual, pilot.dual_residual,
            pilot.gap_abs, pilot.gap_rel, P) && pilot.x !== nothing
        if !pilot_ok && _target_reference_retry_enabled(P)
            retry_P = _target_reference_refined_parameters(pilot_P)
            _, retry = ModelCore._solve_reference_target(cd, snap, scen,
                                                          retry_P, :cost)
            pilot_retry_used = true
            push!(pilot_attempt_statuses, string(retry.status))
            pilot_build_time += Float64(retry.build_time)
            pilot_solve_time += Float64(retry.solve_time)
            pilot = retry
            pilot_status = string(retry.status)
            pilot_ok = _target_reference_accepted(
                retry.status, retry.primal_residual, retry.dual_residual,
                retry.gap_abs, retry.gap_rel, P) && retry.x !== nothing
        end
        pilot_ok || error(
            "common target bandwidth pilot failed after attempts: " *
            join(pilot_attempt_statuses, "=>"))
        pilot_x = pilot.x
    end
    _validate_target_pilot(pilot_x)

    cost_scale = ModelCore._target_cost_scale(cd, snap, P)
    multiplier = Float64(get(P, "target_kde_multiplier", 1.0))
    relative_floor = Float64(get(P, "target_kde_h_min_rel", 1e-6))
    profile_t0 = time()
    profile = build_fixed_bandwidth_profile(
        cd, snap, delta_raw, omega_raw, eL_raw, pilot_x;
        multiplier = multiplier,
        relative_floor = relative_floor,
        cost_scale = cost_scale,
        cost_loss = :positive_part)
    profile_build_time = time() - profile_t0

    if !calibrate
        context_build_time = pilot_build_time + profile_build_time
        return (
            rule_mode = mode, bandwidth_profile = profile, target_meta = nothing,
            state_index = state_index, draw_index = draw_index,
            pilot_x = pilot_x, pilot_status = pilot_status,
            pilot_attempt_statuses = copy(pilot_attempt_statuses),
            pilot_retry_used = pilot_retry_used,
            pilot_build_time = pilot_build_time,
            pilot_solve_time = pilot_solve_time,
            profile_build_time = profile_build_time,
            calibration_build_time = 0.0,
            calibration_solve_time = 0.0,
            context_build_time = context_build_time,
            context_solve_time = pilot_solve_time,
            bandwidth_relative_floor = relative_floor,
            cost_scale = cost_scale,
            calibration_status = "NOT_REQUESTED",
        )
    end

    meta0 = calibrate_targets(cd, snap, scen, P;
                              bandwidth_profile = profile)
    calibration_attempt_statuses = [join(string.(meta0.statuses), ";")]
    calibration_retry_used = false
    calibration_build_time = Float64(meta0.build_time)
    calibration_solve_time = Float64(meta0.solve_time)
    calibration_ok = _target_calibration_accepted(meta0, P)
    if !calibration_ok && _target_reference_retry_enabled(P)
        retry_P = _target_reference_refined_parameters(P)
        retry_meta = calibrate_targets(cd, snap, scen, retry_P;
                                       bandwidth_profile = profile)
        calibration_retry_used = true
        push!(calibration_attempt_statuses,
              join(string.(retry_meta.statuses), ";"))
        calibration_build_time += Float64(retry_meta.build_time)
        calibration_solve_time += Float64(retry_meta.solve_time)
        meta0 = retry_meta
        calibration_ok = _target_calibration_accepted(meta0, P)
    end
    calibration_ok || error(
        "common target calibration failed after attempts: " *
        join(calibration_attempt_statuses, "=>"))
    context_build_time = pilot_build_time + profile_build_time +
                         calibration_build_time
    context_solve_time = pilot_solve_time + calibration_solve_time
    meta = merge(meta0, (
        cost_scale_rule = String(get(P, "target_cost_scale_rule",
                                     "forecast_full_shed")),
        cost_coordinate = "dimensionless_positive_part",
        kappa_coordinate = "dimensionless",
        target_p = Int(P["p"]),
        cost_loss = "positive_part_per_load",
        bandwidth_pilot = "common_no_smoothing_cost_feasible_anchor",
        bandwidth_raw_source = "uncompressed_training",
        bandwidth_relative_floor = relative_floor,
        quad_rule = String(get(P, "kde_quad_rule", "gauss_hermite")),
        quad_nodes = Int(get(P, "kde_quad_nodes", 0)),
        pilot_status = pilot_status,
        pilot_attempt_statuses = copy(pilot_attempt_statuses),
        pilot_retry_used = pilot_retry_used,
        pilot_build_time = pilot_build_time,
        pilot_solve_time = pilot_solve_time,
        profile_build_time = profile_build_time,
        context_build_time = context_build_time,
        context_solve_time = context_solve_time,
        calibration_attempt_statuses = copy(calibration_attempt_statuses),
        calibration_retry_used = calibration_retry_used,
    ))
    return (rule_mode = mode, bandwidth_profile = profile,
            target_meta = meta, state_index = state_index,
            draw_index = draw_index, pilot_x = pilot_x,
            pilot_status = pilot_status,
            pilot_attempt_statuses = copy(pilot_attempt_statuses),
            pilot_retry_used = pilot_retry_used,
            pilot_build_time = pilot_build_time,
            pilot_solve_time = pilot_solve_time,
            profile_build_time = profile_build_time,
            calibration_build_time = calibration_build_time,
            calibration_solve_time = calibration_solve_time,
            context_build_time = context_build_time,
            context_solve_time = context_solve_time,
            bandwidth_relative_floor = relative_floor,
            cost_scale = cost_scale,
            calibration_attempt_statuses = copy(calibration_attempt_statuses),
            calibration_retry_used = calibration_retry_used,
            calibration_status = "ACCEPTED")
end

"""Convenience overload extracting the complete raw sample from a panel split."""
function prepare_target_rs_context(cd::CaseData, snap::Snapshot,
                                   scen::ScenarioData, P::Dict,
                                   panel, train_indices::AbstractVector{<:Integer};
                                   pilot_x = nothing,
                                   calibrate::Bool = true,
                                   state_index = nothing,
                                   draw_index = nothing)
    train = collect(Int, train_indices)
    delta_raw = panel.δ[train, :]
    omega_raw = panel.Ω[train]
    eL_raw = panel.Lact[train, :] .- panel.Lf[train, :]
    return prepare_target_rs_context(cd, snap, scen, P,
                                     delta_raw, omega_raw, eL_raw;
                                     pilot_x = pilot_x,
                                     calibrate = calibrate,
                                     state_index = state_index,
                                     draw_index = draw_index)
end

"运行(或从缓存载入)数据管线。缓存为 Serialization 的整段结果。"
function load_or_run_pipeline(system::String, cfg::Dict, root::String)
    cdir = joinpath(root, cfg["run"]["cache_dir"], system)
    jls  = joinpath(cdir, "pipeline.jls")
    signature = _pipeline_signature(system, cfg, root)
    if get(cfg["run"], "cache", true) && isfile(jls)
        res = deserialize(jls)
        ver = hasproperty(res, :pipeline_version) ? res.pipeline_version : 0
        cached_signature = hasproperty(res, :pipeline_signature) ?
                           res.pipeline_signature : ""
        if ver == DataPipeline.PIPELINE_VERSION && cached_signature == signature
            @info "载入管线缓存 $system"
            return res
        end
        reason = ver != DataPipeline.PIPELINE_VERSION ?
                 "version $ver -> $(DataPipeline.PIPELINE_VERSION)" :
                 "configuration/source signature changed"
        @info "管线缓存失效 $system: $reason, 重新生成"
    end
    @info "运行数据管线 $system ..."
    res = run_pipeline(system, cfg; case_dir = _sysdir(root, system), out_dir = cdir)
    res = merge(res, (pipeline_signature = signature,))
    serialize(jls, res)
    return res
end

# Unified model dispatch. M0--M3 and M6 use ModelCore; M4 and M5 use
# literature-baseline builders.
# Public table labels used by new experiments:
#   M4 = Wasserstein-CVaR-DRCC baseline, M5 = Moment-DRCC baseline,
#   M6 = proposed KDE robust-satisficing model.
function _max_primal_violation(m)
    has_values(m) || return Inf
    try
        report = primal_feasibility_report(m; atol = 0.0)
        return isempty(report) ? 0.0 : maximum(values(report))
    catch
        return NaN
    end
end

function _clarabel_quality(m)
    try
        info = unsafe_backend(m).solver_info
        return (primal = Float64(info.res_primal),
                dual = Float64(info.res_dual),
                gap_abs = Float64(info.gap_abs),
                gap_rel = Float64(info.gap_rel))
    catch
        return (primal = NaN, dual = NaN, gap_abs = NaN, gap_rel = NaN)
    end
end

function _physical_violation_report(cd::CaseData, snap::Snapshot,
                                    scen::ScenarioData, x;
                                    absolute_tolerance::Real = 0.0)
    arrays = (x.g, x.rU, x.rD, x.snom, x.cW,
              getproperty(x, Symbol("\u03b2")), getproperty(x, Symbol("\u03b1S")))
    all(all(isfinite, a) for a in arrays) ||
        return (value = Inf, raw_ratio = Inf, label = "nonfinite_solution",
                raw = Inf, scale = 1.0, absolute_tolerance = Float64(absolute_tolerance),
                nchecks = 0)
    abs_tol = max(Float64(absolute_tolerance), 0.0)
    checks = NamedTuple{(:value, :raw_ratio, :label, :raw, :scale),
                        Tuple{Float64,Float64,String,Float64,Float64}}[]
    function addcheck!(label, raw, scale = 1.0)
        rawf = Float64(raw)
        scalef = max(Float64(scale), eps(Float64))
        push!(checks, (value = max(rawf - abs_tol, 0.0) / scalef,
                       raw_ratio = rawf / scalef, label = String(label),
                       raw = rawf, scale = scalef))
    end
    for j in 1:cd.nG
        scale = max(cd.pmax[j], 1.0)
        addcheck!("generator_upper[$j]", max(x.g[j] + x.rU[j] - cd.pmax[j], 0.0), scale)
        addcheck!("generator_lower[$j]", max(cd.pmin[j] - (x.g[j] - x.rD[j]), 0.0), scale)
        addcheck!("reserve_up_nonnegative[$j]", max(-x.rU[j], 0.0), scale)
        addcheck!("reserve_down_nonnegative[$j]", max(-x.rD[j], 0.0), scale)
    end
    for d in 1:cd.nD
        scale = max(snap.lf[d], 1.0)
        addcheck!("shed_nonnegative[$d]", max(-x.snom[d], 0.0), scale)
        addcheck!("shed_upper[$d]", max(x.snom[d] - snap.lf[d], 0.0), scale)
    end
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    for r in 1:cd.nW
        scale = max(snap.wf[r], 1.0)
        addcheck!("curtail_nonnegative[$r]", max(-x.cW[r], 0.0), scale)
        addcheck!("curtail_upper[$r]", max(x.cW[r] - snap.wf[r], 0.0), scale)
        addcheck!("curtail_sample_safe[$r]", max(x.cW[r] - curtail_cap[r], 0.0), scale)
    end
    balance = sum(x.g) - (sum(snap.lf) - sum(snap.wf) + sum(x.cW) - sum(x.snom))
    addcheck!("forecast_power_balance", abs(balance), max(sum(snap.lf), 1.0))
    f0 = cd.MG * x.g + cd.MW * (snap.wf - x.cW) - cd.MD * (snap.lf - x.snom)
    for ell in 1:cd.nE
        isfinite(cd.F_max[ell]) || continue
        addcheck!("nominal_line_limit[$ell]",
                  max(abs(f0[ell]) - cd.F_max[ell], 0.0),
                  max(cd.F_max[ell], 1.0))
    end
    beta = getproperty(x, Symbol("\u03b2"))
    alpha = getproperty(x, Symbol("\u03b1S"))
    addcheck!("beta_nonnegative", maximum(max.(-beta, 0.0); init = 0.0))
    addcheck!("alpha_shed_nonnegative", maximum(max.(-alpha, 0.0); init = 0.0))
    addcheck!("recourse_closure", abs(sum(beta) + sum(alpha) - 1.0))
    for j in 1:cd.nG
        cd.agc_mask[j] && continue
        addcheck!("non_agc_beta[$j]", abs(beta[j]))
        addcheck!("non_agc_reserve_up[$j]", abs(x.rU[j]), max(cd.pmax[j], 1.0))
        addcheck!("non_agc_reserve_down[$j]", abs(x.rD[j]), max(cd.pmax[j], 1.0))
    end
    isempty(checks) && return (value = 0.0, raw_ratio = 0.0, label = "none",
                               raw = 0.0, scale = 1.0,
                               absolute_tolerance = abs_tol, nchecks = 0)
    # Preserve the largest raw residual when every check is below the absolute floor.
    worst = checks[argmax([(c.value, c.raw_ratio) for c in checks])]
    return merge(worst, (absolute_tolerance = abs_tol, nchecks = length(checks)))
end

_physical_violation(cd::CaseData, snap::Snapshot, scen::ScenarioData, x) =
    _physical_violation_report(cd, snap, scen, x).value

function _physical_solution_fields(cd::CaseData, snap::Snapshot,
                                   scen::ScenarioData, x, P::Dict)
    abs_tol = Float64(get(P, "solver_accept_physical_abs", 1e-5))
    report = _physical_violation_report(cd, snap, scen, x;
                                        absolute_tolerance = abs_tol)
    return (physical_violation = report.value,
            physical_violation_raw_ratio = report.raw_ratio,
            physical_violation_label = report.label,
            physical_violation_raw = report.raw,
            physical_violation_scale = report.scale,
            physical_absolute_tolerance = report.absolute_tolerance)
end

function _solution_accepted(u, P::Dict)
    accepted_statuses = Set(("OPTIMAL", "ALMOST_OPTIMAL"))
    status_ok = string(u.status) in accepted_statuses
    residual_tol = Float64(get(P, "solver_accept_scaled_residual", 1e-5))
    physical_tol = Float64(get(P, "solver_accept_physical_violation", 1e-6))
    gap_abs_tol = Float64(get(P, "solver_accept_gap_abs",
                              get(P, "solver_accept_gap", 5e-5)))
    gap_rel_tol = Float64(get(P, "solver_accept_gap_rel",
                              get(P, "solver_accept_gap", 1e-5)))
    residual_ok = isfinite(u.primal_residual) && isfinite(u.dual_residual) &&
                  max(u.primal_residual, u.dual_residual) <= residual_tol
    physical_ok = isfinite(u.physical_violation) && u.physical_violation <= physical_tol
    gap_ok = isfinite(u.gap_abs) && isfinite(u.gap_rel) &&
             (abs(u.gap_abs) <= gap_abs_tol || abs(u.gap_rel) <= gap_rel_tol)
    target_ok = if u.target_meta === nothing
        true
    else
        tm = u.target_meta
        all(s -> string(s) in accepted_statuses, tm.statuses) &&
        all(isfinite, tm.primal_residuals) && all(isfinite, tm.dual_residuals) &&
        maximum(max.(tm.primal_residuals, tm.dual_residuals)) <= residual_tol &&
        all(isfinite, tm.gap_abs_values) && all(isfinite, tm.gap_rel_values) &&
        all((abs(tm.gap_abs_values[i]) <= gap_abs_tol ||
             abs(tm.gap_rel_values[i]) <= gap_rel_tol) for i in eachindex(tm.gap_abs_values))
    end
    return status_ok && residual_ok && physical_ok && gap_ok && target_ok
end

function _baseline_result(r)
    beta_sym = Symbol("\u03b2")
    names = (:status, :x, Symbol("\u03b2b"), :is_rs,
             :solve_time, :build_time, :solver_time, :iters, :obj,
             :n_var, :n_con, Symbol("\u03bac"), Symbol("\u03bad"),
             :z0, :z0_solve, :z0_build, :target_meta, :jump_report_distance,
             :primal_residual, :dual_residual, :gap_abs, :gap_rel)
    vals = (r.status, r.x, getproperty(r.x, beta_sym), false,
            r.solve_time, r.build_time, r.solver_time, r.iters, r.obj,
            r.n_var, r.n_con, NaN, NaN,
            NaN, NaN, NaN, nothing, r.jump_report_distance,
            r.primal_residual, r.dual_residual, r.gap_abs, r.gap_rel)
    return NamedTuple{names}(vals)
end
function _solve_one_once(model::String, cd::CaseData, snap::Snapshot,
                         scen::ScenarioData, P::Dict, rho_tau::Float64;
                         bandwidth_profile = nothing,
                         target_meta_override = nothing)
    if model in ("M4", "WDRO", "W_DRO", "WASSERSTEIN")
        radius_mode = lowercase(String(get(P, "wasserstein_radius_mode", "absolute")))
        rho_w = if radius_mode == "compression_multiple"
            Float64(get(P, "wasserstein_radius_multiplier", 1.0)) * scen.compression_l1
        elseif radius_mode == "absolute"
            Float64(get(P, "wasserstein_radius", 0.1))
        else
            error("unknown Wasserstein radius mode: $radius_mode")
        end
        r = build_wdro(cd, snap, scen, P; rho_wass = rho_w)
        u = _baseline_result(r)
        return merge(u, _physical_solution_fields(cd, snap, scen, u.x, P))
    elseif model in ("M5", "M5M", "MOMENT", "MOMENT_DRCC")
        r = build_moment_dro(cd, snap, scen, P)
        u = _baseline_result(r)
        return merge(u, _physical_solution_fields(cd, snap, scen, u.x, P))
    end
    core_model = model
    conf = MODEL_CONFIGS[core_model]
    z0t = NaN; z0_solve = NaN; z0_build = NaN; tau_cost = 0.0; tmeta = nothing
    if conf.cost == :rs
        Z0, rz = calibrate_Z0(cd, snap, scen, P)
        tau_cost = (1 + rho_tau) * Z0; z0t = Z0
        z0_solve = rz.solve_time; z0_build = rz.build_time
    elseif conf.cost == :unified_rs
        tmeta = target_meta_override === nothing ?
            calibrate_targets(cd, snap, scen, P;
                              bandwidth_profile = bandwidth_profile) :
            target_meta_override
        z0t = tmeta.Z0c
        z0_solve = target_meta_override === nothing ? tmeta.solve_time : 0.0
        z0_build = target_meta_override === nothing ? tmeta.build_time : 0.0
    end
    r = build_ols(cd, snap, scen, P; safety = conf.safety, cost = conf.cost,
                  τcost = tau_cost, target_meta = tmeta,
                  bandwidth_profile = bandwidth_profile)
    solved_target_meta = r.target_meta === nothing ? tmeta : r.target_meta
    quality = _clarabel_quality(r.model)
    physical = _physical_solution_fields(cd, snap, scen, r.x, P)
    result = (status = r.status, x = r.x, βb = r.x.β,
            is_rs = (conf.cost in (:rs, :unified_rs)),
            solve_time = r.solve_time, build_time = r.build_time,
            solver_time = r.solver_time, iters = r.iters, obj = r.obj,
            n_var = r.n_var, n_con = r.n_con,
            κc = conf.cost in (:rs, :unified_rs) ? r.κc : NaN,
            κd = conf.cost == :unified_rs ? r.κd : NaN,
            z0 = z0t, z0_solve = z0_solve, z0_build = z0_build,
            target_meta = solved_target_meta,
            jump_report_distance = _max_primal_violation(r.model),
            primal_residual = quality.primal, dual_residual = quality.dual,
            gap_abs = quality.gap_abs, gap_rel = quality.gap_rel)
    return merge(result, physical)
end

function _solve_one(model::String, cd::CaseData, snap::Snapshot, scen::ScenarioData,
                    P::Dict, rho_tau::Float64;
                    bandwidth_profile = nothing,
                    target_meta_override = nothing)
    primary = _solve_one_once(model, cd, snap, scen, P, rho_tau;
                              bandwidth_profile = bandwidth_profile,
                              target_meta_override = target_meta_override)
    primary_status = string(primary.status)
    retry_models = String.(get(P, "solver_retry_models", ["M6"]))
    retry_enabled = model in retry_models &&
                    Bool(get(P, "solver_retry_refined", false)) &&
                    !_solution_accepted(primary, P)
    if !retry_enabled
        return merge(primary, (solver_attempts = 1,
                               primary_status = primary_status,
                               retry_status = ""))
    end

    retryP = deepcopy(P)
    retryP["solver_direct_solve_method"] = "qdldl"
    retryP["solver_equilibrate_max_iter"] =
        Int(get(P, "solver_retry_equilibrate_max_iter", 20))
    retryP["solver_iterative_refinement_max_iter"] =
        Int(get(P, "solver_retry_iterative_refinement_max_iter", 20))
    retry = _solve_one_once(model, cd, snap, scen, retryP, rho_tau;
                            bandwidth_profile = bandwidth_profile,
                            target_meta_override = primary.target_meta)
    selected = _solution_accepted(retry, P) ? retry : primary
    return merge(selected, (
        solve_time = _finite_time(primary.solve_time) + _finite_time(retry.solve_time),
        build_time = _finite_time(primary.build_time) + _finite_time(retry.build_time),
        solver_time = _finite_time(primary.solver_time) + _finite_time(retry.solver_time),
        z0_solve = primary.z0_solve,
        z0_build = primary.z0_build,
        target_meta = selected.target_meta,
        solver_attempts = 2,
        primary_status = primary_status,
        retry_status = string(retry.status),
    ))
end

_finite_time(x) = try
    xf = Float64(x)
    isfinite(xf) ? xf : 0.0
catch
    0.0
end

_model_wall(u) = _finite_time(u.build_time) + _finite_time(u.solve_time) +
                 _finite_time(u.z0_build) + _finite_time(u.z0_solve)

_online_wall(u) = _finite_time(u.build_time) + _finite_time(u.solve_time)

_ctx_get(ctx, name::Symbol, default) =
    ctx !== nothing && hasproperty(ctx, name) ? getproperty(ctx, name) : default

function _context_index(ctx, names::Tuple)
    values = Any[]
    for name in names
        hasproperty(ctx, name) || continue
        value = getproperty(ctx, name)
        value === nothing || push!(values, value)
    end
    isempty(values) && return nothing
    all(==(first(values)), values) ||
        error("target context contains conflicting index fields $(join(string.(names), ", "))")
    return first(values)
end

function _validate_target_contexts(target_contexts, states::AbstractVector;
                                   require_draw_index::Bool = false)
    target_contexts === nothing && return nothing
    target_contexts isa AbstractVector ||
        throw(ArgumentError("target_contexts must be a per-state vector"))
    length(target_contexts) == length(states) ||
        throw(DimensionMismatch(
            "target_contexts length $(length(target_contexts)) must equal state count $(length(states))"))
    for i in eachindex(states)
        ctx = target_contexts[i]
        ctx === nothing && continue
        state_index = _context_index(ctx, (:state_index, :snapshot_index, :snapshot))
        state_index === nothing && error(
            "target_contexts[$i] must record state_index (or snapshot_index/snapshot)")
        Int(state_index) == Int(states[i]) || error(
            "target_contexts[$i] belongs to state $state_index, expected $(states[i])")
        if require_draw_index
            draw_index = _context_index(ctx, (:draw_index, :mc_draw))
            draw_index === nothing && error(
                "case_by_draw requires target_contexts[$i] to record draw_index")
            Int(draw_index) == i || error(
                "target_contexts[$i] belongs to draw $draw_index, expected $i")
        end
    end
    return target_contexts
end

_context_model(model::AbstractString) = uppercase(strip(model)) in ("M3", "M3G", "M6")

function _validate_context_profile(ctx, model::String, cd::CaseData,
                                   snap::Snapshot, scen::ScenarioData,
                                   P::Dict, rho_tau::Float64)
    profile = _ctx_get(ctx, :bandwidth_profile, nothing)
    profile isa FixedBandwidthProfile || error(
        "$model target context is missing a FixedBandwidthProfile")
    profile.raw_n == scen.raw_n || error(
        "$model target context raw_n=$(profile.raw_n) does not match scen.raw_n=$(scen.raw_n)")
    profile_timing_fields = (:pilot_build_time, :pilot_solve_time,
                             :profile_build_time)
    all(name -> hasproperty(ctx, name), profile_timing_fields) || error(
        "$model target context is missing profile-preparation timing fields")
    all(name -> begin
            value = Float64(getproperty(ctx, name))
            isfinite(value) && value >= 0
        end, profile_timing_fields) || error(
        "$model target context contains invalid profile-preparation timing")

    mode = ModelCore._target_rs_mode(P)
    if mode == "common_dimensionless"
        ModelCore._validate_common_profile(profile, cd, snap, P)
        expected_events = 2 * count(cd.agc_mask) +
                          2 * count(isfinite, cd.F_max) + 2 * cd.nD
        length(profile.event_h) == expected_events || error(
            "$model common profile contains $(length(profile.event_h)) elementary-event bandwidths; " *
            "expected $expected_events for this case")
        length(profile.event_labels) == expected_events || error(
            "$model common profile event labels do not match its state-specific event domain")
        all(>(0.0), profile.event_h) || error(
            "$model common profile requires strictly positive elementary-event bandwidths")
        hfloor = Float64(get(P, "target_kde_h_min_rel", 1e-6))
        minimum(profile.event_h) + 1e-14 >= hfloor || error(
            "$model elementary-event bandwidth is below the configured positive floor")
        gamma_m = max(Float64(P["gamma_m"]), 1.0 - Float64(P["eps_m"]))
        gamma_target = ModelCore._common_target_gamma(P)
        isapprox(gamma_m, gamma_target; rtol = 1e-12, atol = 1e-12) || error(
            "common M3/M6 protocol requires the same gamma: M3=$gamma_m, M6=$gamma_target")
    end

    meta = nothing
    if uppercase(strip(model)) == "M6"
        meta = _ctx_get(ctx, :target_meta, nothing)
        mode == "common_dimensionless" && meta === nothing && error(
            "common M6 target context has no calibrated target_meta")
        if meta !== nothing
            calibration_timing_fields = (:calibration_build_time,
                                         :calibration_solve_time)
            all(name -> hasproperty(ctx, name), calibration_timing_fields) || error(
                "M6 target context is missing calibration timing fields")
            all(name -> begin
                    value = Float64(getproperty(ctx, name))
                    isfinite(value) && value >= 0
                end, calibration_timing_fields) || error(
                "M6 target context contains invalid calibration timing")
            hasproperty(meta, :target_h) &&
                all(isapprox.(meta.target_h, profile.target_h;
                              rtol = 1e-12, atol = 1e-12)) || error(
                "M6 target_meta safety bandwidths do not match the shared profile")
            hasproperty(meta, :cost_h) &&
                isapprox(meta.cost_h, profile.cost_h;
                         rtol = 1e-12, atol = 1e-10) || error(
                "M6 target_meta cost bandwidth does not match the shared profile")
            if mode == "common_dimensionless"
                hasproperty(meta, :rule_mode) &&
                    String(meta.rule_mode) == mode || error(
                    "M6 target_meta was not calibrated in common_dimensionless mode")
                hasproperty(meta, :bandwidth_raw_n) &&
                    Int(meta.bandwidth_raw_n) == profile.raw_n || error(
                    "M6 target_meta raw sample count does not match the shared profile")
                hasproperty(meta, :rho_tau) &&
                    isapprox(Float64(meta.rho_tau), rho_tau;
                             rtol = 1e-12, atol = 1e-12) || error(
                    "M6 target_meta rho_tau=$(hasproperty(meta, :rho_tau) ? meta.rho_tau : missing) " *
                    "does not match evaluator rho_tau=$rho_tau")
                hasproperty(meta, :target_gamma) &&
                    isapprox(Float64(meta.target_gamma),
                             ModelCore._common_target_gamma(P);
                             rtol = 1e-12, atol = 1e-12) || error(
                    "M6 target_meta gamma does not match the common protocol")
                hasproperty(meta, :k_rule) &&
                    String(meta.k_rule) == ModelCore._common_target_k_rule(P) ||
                    error("M6 target_meta k rule does not match the common protocol")
                hasproperty(meta, :k) && length(meta.k) == 6 &&
                    all(v -> isfinite(v) && v >= 0.0, Float64.(meta.k)) ||
                    error("M6 target_meta must contain six finite nonnegative k coefficients")
                k_rule = String(meta.k_rule)
                if k_rule == "reference_target_ratio_no_eps"
                    denominator = abs(Float64(meta.Z0c))
                    denominator > 0.0 && isfinite(denominator) ||
                        error("M6 target_meta has undefined no-epsilon cost scale Z0c")
                    expected_k = abs.(Float64.(meta.Z0m)) ./ denominator
                    all(isapprox.(Float64.(meta.k), expected_k;
                                  rtol = 1e-12, atol = 0.0)) ||
                        error("M6 target_meta k coefficients do not equal |Z0m|/|Z0c|")
                elseif k_rule == "payoff_range_no_eps"
                    _validated_payoff_range_metadata(meta)
                    all(v -> isfinite(v) && v > 0.0, Float64.(meta.k)) ||
                        error("payoff-range M6 target_meta must contain six strictly positive k coefficients")
                    hasproperty(meta, :reference_solutions) &&
                        length(meta.reference_solutions) == 7 &&
                        all(x -> x !== nothing, meta.reference_solutions) || error(
                        "payoff-range M6 target_meta must retain all seven anchor decisions")
                elseif k_rule == "equal_normalized"
                    all(isapprox.(Float64.(meta.k), ones(6);
                                  rtol = 1e-12, atol = 1e-12)) ||
                        error("equal-normalized M6 target_meta must contain unit k coefficients")
                else
                    error("unknown M6 target k rule: $k_rule")
                end
                (!hasproperty(meta, :target_p) ||
                    Int(meta.target_p) == Int(P["p"])) || error(
                    "M6 target_meta HMCR order does not match the common protocol")
                required_quality = (:statuses, :primal_residuals,
                                    :dual_residuals, :gap_abs_values,
                                    :gap_rel_values)
                all(name -> hasproperty(meta, name), required_quality) || error(
                    "M6 target_meta is missing calibration quality fields")
                nrefs = length(meta.statuses)
                nrefs == 7 || error(
                    "M6 target_meta must contain cost plus six safety calibrations")
                all(length(getproperty(meta, name)) == nrefs
                    for name in required_quality[2:end]) || error(
                    "M6 target_meta calibration quality vectors have inconsistent lengths")
                all(i -> _target_reference_accepted(
                    meta.statuses[i], meta.primal_residuals[i],
                    meta.dual_residuals[i], meta.gap_abs_values[i],
                    meta.gap_rel_values[i], P), 1:nrefs) || error(
                    "M6 target_meta failed its calibration quality gate")
                if k_rule == "payoff_range_no_eps"
                    _payoff_evaluation_quality_accepted(meta, P) || error(
                        "payoff-range M6 target_meta failed its joint-row quality gate")
                end
            end
        end
    end
    return (bandwidth_profile = profile, target_meta_override = meta)
end

function _resolve_target_context(target_contexts, i::Int, model::String,
                                 cd::CaseData, snap::Snapshot,
                                 scen::ScenarioData, P::Dict,
                                 rho_tau::Float64)
    uses_context = _context_model(model)
    ctx = target_contexts === nothing ? nothing : target_contexts[i]
    required = uses_context && ModelCore._target_rs_mode(P) == "common_dimensionless"
    required && ctx === nothing && error(
        "common_dimensionless $model requires target_contexts[$i] for every state that calls the model")
    if !uses_context || ctx === nothing
        return (context = nothing, bandwidth_profile = nothing,
                target_meta_override = nothing)
    end
    resolved = _validate_context_profile(ctx, model, cd, snap, scen, P, rho_tau)
    return merge((context = ctx,), resolved)
end

function _context_timing(ctx, model::String)
    ctx === nothing && return (
        profile = 0.0, calibration = 0.0, charged = 0.0)
    profile = _finite_time(_ctx_get(ctx, :pilot_build_time, 0.0)) +
              _finite_time(_ctx_get(ctx, :pilot_solve_time, 0.0)) +
              _finite_time(_ctx_get(ctx, :profile_build_time, 0.0))
    calibration = _finite_time(_ctx_get(ctx, :calibration_build_time, 0.0)) +
                  _finite_time(_ctx_get(ctx, :calibration_solve_time, 0.0))
    charged = uppercase(strip(model)) in ("M3", "M3G") ? profile :
              uppercase(strip(model)) == "M6" ? profile + calibration : 0.0
    return (profile = profile, calibration = calibration, charged = charged)
end

function _profile_log_fields(ctx, P::Dict, model::String)
    profile = _ctx_get(ctx, :bandwidth_profile, nothing)
    gamma_m = haskey(P, "gamma_m") && haskey(P, "eps_m") ?
              max(Float64(P["gamma_m"]), 1.0 - Float64(P["eps_m"])) : NaN
    gamma = uppercase(strip(model)) == "M6" ?
            Float64(get(P, "target_gamma", gamma_m)) : gamma_m
    p = Int(get(P, "p", -1))
    quad_rule = String(get(P, "kde_quad_rule", "gauss_hermite"))
    quad_nodes = Int(get(P, "kde_quad_nodes", 0))
    quad_weight_floor = Float64(get(P, "kde_quad_weight_floor", 0.0))
    phi_type = String(get(P, "phi_type", ""))
    theta_m = Float64(get(P, "theta_m", NaN))
    relative_floor = Float64(_ctx_get(
        ctx, :bandwidth_relative_floor,
        get(P, "target_kde_h_min_rel", NaN)))
    cost_scale = Float64(_ctx_get(ctx, :cost_scale, NaN))
    timing = _context_timing(ctx, model)
    if !(profile isa FixedBandwidthProfile)
        return (
            kde_profile_used = false, kde_profile_raw_n = -1,
            kde_profile_multiplier = NaN,
            kde_profile_relative_floor = relative_floor,
            kde_event_h_min = NaN, kde_event_h_max = NaN,
            kde_target_h_min = NaN, kde_target_h_max = NaN,
            kde_cost_h = NaN, kde_cost_h_normalized = NaN,
            kde_kernel = "gaussian", kde_protocol_gamma = gamma,
            kde_protocol_p = p, kde_phi_type = phi_type,
            kde_theta_m = theta_m, kde_quad_rule = quad_rule,
            kde_quad_nodes = quad_nodes,
            kde_quad_weight_floor = quad_weight_floor,
            target_context_state_index = -1,
            target_context_draw_index = -1,
            target_context_calibration_status = "",
            target_context_pilot_status = "",
            target_context_pilot_attempt_statuses = "",
            target_context_pilot_retry_used = false,
            target_context_calibration_attempt_statuses = "",
            target_context_calibration_retry_used = false,
            context_profile_wall = timing.profile,
            context_calibration_wall = timing.calibration,
            context_charged_wall = timing.charged,
        )
    end
    event_min = isempty(profile.event_h) ? NaN : minimum(profile.event_h)
    event_max = isempty(profile.event_h) ? NaN : maximum(profile.event_h)
    target_min = isempty(profile.target_h) ? NaN : minimum(profile.target_h)
    target_max = isempty(profile.target_h) ? NaN : maximum(profile.target_h)
    return (
        kde_profile_used = true, kde_profile_raw_n = profile.raw_n,
        kde_profile_multiplier = profile.multiplier,
        kde_profile_relative_floor = relative_floor,
        kde_event_h_min = event_min, kde_event_h_max = event_max,
        kde_target_h_min = target_min, kde_target_h_max = target_max,
        kde_cost_h = profile.cost_h,
        kde_cost_h_normalized = isfinite(cost_scale) && cost_scale > 0 ?
                                profile.cost_h / cost_scale : NaN,
        kde_kernel = "gaussian", kde_protocol_gamma = gamma,
        kde_protocol_p = p, kde_phi_type = phi_type,
        kde_theta_m = theta_m, kde_quad_rule = quad_rule,
        kde_quad_nodes = quad_nodes,
        kde_quad_weight_floor = quad_weight_floor,
        target_context_state_index = Int(_context_index(
            ctx, (:state_index, :snapshot_index, :snapshot))),
        target_context_draw_index = begin
            value = _context_index(ctx, (:draw_index, :mc_draw))
            value === nothing ? -1 : Int(value)
        end,
        target_context_calibration_status = String(_ctx_get(
            ctx, :calibration_status, "")),
        target_context_pilot_status = String(_ctx_get(ctx, :pilot_status, "")),
        target_context_pilot_attempt_statuses = join(string.(_ctx_get(
            ctx, :pilot_attempt_statuses, String[])), "=>"),
        target_context_pilot_retry_used = Bool(_ctx_get(
            ctx, :pilot_retry_used, false)),
        target_context_calibration_attempt_statuses = join(string.(_ctx_get(
            ctx, :calibration_attempt_statuses, String[])), "=>"),
        target_context_calibration_retry_used = Bool(_ctx_get(
            ctx, :calibration_retry_used, false)),
        context_profile_wall = timing.profile,
        context_calibration_wall = timing.calibration,
        context_charged_wall = timing.charged,
    )
end

function _evaluation_timing(u, ctx, model::String)
    context = _context_timing(ctx, model)
    online = _online_wall(u)
    internal_calibration = _finite_time(u.z0_build) + _finite_time(u.z0_solve)
    end_to_end = online + internal_calibration + context.charged
    return (online = online, internal_calibration = internal_calibration,
            profile = context.profile, calibration = context.calibration,
            charged = context.charged, end_to_end = end_to_end)
end

"对单个 (cd, scen, 快照集) 求解一个模型并审计。返回聚合 + 逐实例计时/精确度日志。"
function _audit_tol_mw(P::Dict)
    tol = Float64(get(P, "audit_tol_mw", Audit.DEFAULT_AUDIT_TOL_MW))
    isfinite(tol) && tol >= 0 ||
        error("audit_tol_mw must be finite and nonnegative")
    return tol
end

function eval_model(system::String, model::String, cd::CaseData, scen::ScenarioData,
                    panel, tset::Vector{Int}, P::Dict, ρτ::Float64, voll::Float64,
                    lolp_thr::Float64;
                    target_contexts = nothing)
    rows = Vector{Any}(); times = Float64[]; online_times = Float64[]
    total_times = Float64[]; profile_times = Float64[]
    calibration_times = Float64[]
    κs = Float64[]; κds = Float64[]; nfail = 0
    log = DataFrame()                                   # 逐实例计时/精确度
    audit_tol_mw = _audit_tol_mw(P)
    _validate_target_contexts(target_contexts, tset)
    for (state_id, t) in enumerate(tset)
        snap = snapshot_at(panel, t, cd)
        resolved = _resolve_target_context(target_contexts, state_id, model,
                                           cd, snap, scen, P, ρτ)
        u = _solve_one(model, cd, snap, scen, P, ρτ;
                       bandwidth_profile = resolved.bandwidth_profile,
                       target_meta_override = resolved.target_meta_override)
        timing = _evaluation_timing(u, resolved.context, model)
        st = string(u.status)
        ok = _solution_accepted(u, P)
        if ok
            push!(times, u.solve_time)
            push!(online_times, timing.online)
            push!(total_times, timing.end_to_end)
            push!(profile_times, timing.profile)
            push!(calibration_times, timing.calibration)
            u.is_rs && push!(κs, u.κc)
            isfinite(u.κd) && push!(κds, u.κd)
            push!(rows, audit_snapshot(cd, u.x, snap;
                                       tol_mw = audit_tol_mw, βbar = u.βb))  # 仅审计成功解
        else
            nfail += 1
        end
        push!(log, merge((system = system, model = model, snapshot = t, status = st,
                          solver_attempts = u.solver_attempts,
                          primary_status = u.primary_status,
                          retry_status = u.retry_status,
                          build_time = round(u.build_time, digits = 4),
                          solve_wall = round(u.solve_time, digits = 4),
                          online_wall = round(timing.online, digits = 4),
                          internal_calibration_wall = round(timing.internal_calibration, digits = 4),
                          internal_model_wall = round(_model_wall(u), digits = 4),
                          model_wall = round(timing.end_to_end, digits = 4),
                          end_to_end_wall = round(timing.end_to_end, digits = 4),
                          solver_time = round(u.solver_time, digits = 4),
                          iters = u.iters, n_var = u.n_var, n_con = u.n_con,
                          accepted = ok, jump_report_distance = u.jump_report_distance,
                          primal_residual = u.primal_residual,
                           dual_residual = u.dual_residual,
                           gap_abs = u.gap_abs, gap_rel = u.gap_rel,
                           physical_violation = u.physical_violation,
                           physical_violation_raw_ratio = u.physical_violation_raw_ratio,
                           physical_violation_label = u.physical_violation_label,
                           physical_violation_raw = u.physical_violation_raw,
                           physical_violation_scale = u.physical_violation_scale,
                           physical_absolute_tolerance = u.physical_absolute_tolerance,
                           obj = u.obj, kappa_c = u.κc, kappa_d = u.κd, Z0 = u.z0,
                          Z0_solve = round(u.z0_solve, digits = 4),
                          Z0_build = round(u.z0_build, digits = 4)),
                         _target_log_fields(u.target_meta),
                         _profile_log_fields(resolved.context, P, model)); promote = true)
    end
    n_ok = length(rows)
    agg = n_ok > 0 ? aggregate_audit(rows, 1.0, voll, lolp_thr) :
          Audit.AuditAgg(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    return (agg = agg, time = isempty(times) ? NaN : mean(times),
            time_max = isempty(times) ? NaN : maximum(times),
            time_online = isempty(online_times) ? NaN : mean(online_times),
            time_online_max = isempty(online_times) ? NaN : maximum(online_times),
            time_total = isempty(total_times) ? NaN : mean(total_times),
            time_total_max = isempty(total_times) ? NaN : maximum(total_times),
            time_end_to_end = isempty(total_times) ? NaN : mean(total_times),
            time_end_to_end_max = isempty(total_times) ? NaN : maximum(total_times),
            context_profile_time = isempty(profile_times) ? NaN : mean(profile_times),
            context_calibration_time = isempty(calibration_times) ? NaN : mean(calibration_times),
            κc = isempty(κs) ? NaN : mean(κs),
            κd = isempty(κds) ? NaN : mean(κds),
            fail_rate = nfail / length(tset), n_ok = n_ok,
            audit_tol_mw = audit_tol_mw, log = log)
end

function _edns_precision(rows::Vector)
    n = length(rows)
    n == 0 && return (sd = NaN, se = NaN, cov = NaN)
    vals = [Float64(r.shed_mw) for r in rows]
    mu = mean(vals)
    if n == 1
        return (sd = NaN, se = NaN, cov = NaN)
    end
    sd = std(vals; corrected = true)
    se = sd / sqrt(n)
    cov = abs(mu) <= eps(Float64) ? (sd <= eps(Float64) ? 0.0 : Inf) : se / abs(mu)
    return (sd = sd, se = se, cov = cov)
end

function _precheck_row(pc)
    return (shed_mw = 0.0, lol = false,
            line_viol = false, max_ratio = pc.max_flow_ratio,
            reserve_viol = false, shedbound_viol = false,
            curtail = pc.curtail, curtail_audit = 0.0,
            gen_cost = pc.gen_cost)
end

function _conservative_failure_row(snap::Snapshot)
    return (shed_mw = sum(snap.lact), lol = true,
            # No accepted decision exists, so safety cannot be certified.  A
            # conservative common-denominator accounting must mark every
            # safety family as violated instead of making failure look safe.
            line_viol = true, max_ratio = Inf,
            reserve_viol = true, shedbound_viol = true,
            curtail = 0.0, curtail_audit = 0.0,
            gen_cost = 0.0)
end

"Fast no-shed DC feasibility check for an sampled actual operating state."
function dc_state_precheck(cd::CaseData, snap::Snapshot)
    t0 = time()
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "max_iter", 500)
    set_optimizer_attribute(m, "equilibrate_enable", true)
    set_optimizer_attribute(m, "tol_gap_abs", 1e-8)
    set_optimizer_attribute(m, "tol_gap_rel", 1e-8)
    set_optimizer_attribute(m, "tol_feas", 1e-8)
    set_optimizer_attribute(m, "reduced_tol_gap_abs", 5e-5)
    set_optimizer_attribute(m, "reduced_tol_gap_rel", 1e-5)
    set_optimizer_attribute(m, "reduced_tol_feas", 1e-5)
    nG, nD, nW, nE = cd.nG, cd.nD, cd.nW, cd.nE

    @variable(m, cd.pmin[j] <= g[j=1:nG] <= cd.pmax[j])
    @variable(m, 0.0 <= cW[r=1:nW] <= snap.wact[r])
    @constraint(m, sum(g) == sum(snap.lact) - sum(snap.wact) + sum(cW))

    f = [sum(cd.MG[ell,j] * g[j] for j in 1:nG) +
         sum(cd.MW[ell,r] * (snap.wact[r] - cW[r]) for r in 1:nW) -
         sum(cd.MD[ell,d] * snap.lact[d] for d in 1:nD) for ell in 1:nE]
    for ell in 1:nE
        isfinite(cd.F_max[ell]) || continue
        @constraint(m, f[ell] <= cd.F_max[ell])
        @constraint(m, -f[ell] <= cd.F_max[ell])
    end
    @objective(m, Min, sum(cd.cgen[j] * g[j] for j in 1:nG))
    optimize!(m)
    st = string(termination_status(m))
    quality = _clarabel_quality(m)
    ok = st in ("OPTIMAL", "ALMOST_OPTIMAL") &&
         isfinite(quality.primal) && isfinite(quality.dual) &&
         max(quality.primal, quality.dual) <= 1e-5
    dt = time() - t0
    if ok
        gv = value.(g)
        cv = value.(cW)
        fv = [sum(cd.MG[ell,j] * gv[j] for j in 1:nG) +
              sum(cd.MW[ell,r] * (snap.wact[r] - cv[r]) for r in 1:nW) -
              sum(cd.MD[ell,d] * snap.lact[d] for d in 1:nD) for ell in 1:nE]
        ratios = [isfinite(cd.F_max[ell]) ? abs(fv[ell]) / cd.F_max[ell] : 0.0 for ell in 1:nE]
        return (ok = true, status = st, time = dt,
                max_flow_ratio = isempty(ratios) ? 0.0 : maximum(ratios),
                curtail = sum(cv),
                gen_cost = sum(cd.cgen[j] * gv[j] for j in 1:nG))
    end
    return (ok = false, status = st, time = dt,
            max_flow_ratio = NaN, curtail = NaN, gen_cost = NaN)
end

"Evaluate one model on independently sampled operating states."
function eval_model_mc(system::String, model::String, cd::CaseData, scen::ScenarioData,
                       panel, draws::Vector{Int}, P::Dict, rho_tau::Float64,
                       voll::Float64, lolp_thr::Float64;
                       target_contexts = nothing)
    rows = Vector{Any}(); times = Float64[]; online_times = Float64[]
    total_times = Float64[]; profile_times = Float64[]
    calibration_times = Float64[]
    kappas = Float64[]; kappads = Float64[]; nfail = 0
    log = DataFrame()
    audit_tol_mw = _audit_tol_mw(P)
    _validate_target_contexts(target_contexts, draws)
    for (draw_id, t) in enumerate(draws)
        snap = snapshot_at(panel, t, cd)
        resolved = _resolve_target_context(target_contexts, draw_id, model,
                                           cd, snap, scen, P, rho_tau)
        u = _solve_one(model, cd, snap, scen, P, rho_tau;
                       bandwidth_profile = resolved.bandwidth_profile,
                       target_meta_override = resolved.target_meta_override)
        timing = _evaluation_timing(u, resolved.context, model)
        st = string(u.status)
        ok = _solution_accepted(u, P)
        if ok
            push!(times, u.solve_time)
            push!(online_times, timing.online)
            push!(total_times, timing.end_to_end)
            push!(profile_times, timing.profile)
            push!(calibration_times, timing.calibration)
            u.is_rs && push!(kappas, u.κc)
            isfinite(u.κd) && push!(kappads, u.κd)
            push!(rows, audit_snapshot(cd, u.x, snap;
                                       tol_mw = audit_tol_mw, βbar = u.βb))
        else
            nfail += 1
        end
        push!(log, merge((system = system, model = model, mc_draw = draw_id,
                          snapshot = t, status = st,
                          build_time = round(u.build_time, digits = 4),
                          solve_wall = round(u.solve_time, digits = 4),
                          online_wall = round(timing.online, digits = 4),
                          internal_calibration_wall = round(timing.internal_calibration, digits = 4),
                          internal_model_wall = round(_model_wall(u), digits = 4),
                          model_wall = round(timing.end_to_end, digits = 4),
                          end_to_end_wall = round(timing.end_to_end, digits = 4),
                          solver_time = round(u.solver_time, digits = 4),
                          iters = u.iters, n_var = u.n_var, n_con = u.n_con,
                          accepted = ok, jump_report_distance = u.jump_report_distance,
                          primal_residual = u.primal_residual,
                           dual_residual = u.dual_residual,
                           gap_abs = u.gap_abs, gap_rel = u.gap_rel,
                           physical_violation = u.physical_violation,
                           physical_violation_raw_ratio = u.physical_violation_raw_ratio,
                           physical_violation_label = u.physical_violation_label,
                           physical_violation_raw = u.physical_violation_raw,
                           physical_violation_scale = u.physical_violation_scale,
                           physical_absolute_tolerance = u.physical_absolute_tolerance,
                           obj = u.obj, kappa_c = u.κc, kappa_d = u.κd, Z0 = u.z0,
                          Z0_solve = round(u.z0_solve, digits = 4),
                          Z0_build = round(u.z0_build, digits = 4)),
                         _target_log_fields(u.target_meta),
                         _profile_log_fields(resolved.context, P, model)); promote = true)
    end
    n_ok = length(rows)
    agg = n_ok > 0 ? aggregate_audit(rows, 1.0, voll, lolp_thr) :
          Audit.AuditAgg(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    prec = _edns_precision(rows)
    return (agg = agg, time = isempty(times) ? NaN : mean(times),
            time_max = isempty(times) ? NaN : maximum(times),
            time_online = isempty(online_times) ? NaN : mean(online_times),
            time_online_max = isempty(online_times) ? NaN : maximum(online_times),
            time_total = isempty(total_times) ? NaN : mean(total_times),
            time_total_max = isempty(total_times) ? NaN : maximum(total_times),
            time_end_to_end = isempty(total_times) ? NaN : mean(total_times),
            time_end_to_end_max = isempty(total_times) ? NaN : maximum(total_times),
            context_profile_time = isempty(profile_times) ? NaN : mean(profile_times),
            context_calibration_time = isempty(calibration_times) ? NaN : mean(calibration_times),
            kappa_c = isempty(kappas) ? NaN : mean(kappas),
            kappa_d = isempty(kappads) ? NaN : mean(kappads),
            edns_sd = prec.sd, edns_se = prec.se, cov_edns = prec.cov,
            fail_rate = nfail / length(draws), n_ok = n_ok,
            audit_tol_mw = audit_tol_mw, log = log)
end

"Evaluate one model with MCS-style no-shed DC precheck before calling OLS/DRCC-OLS."
function eval_model_mc_screened(system::String, model::String, cd::CaseData, scen::ScenarioData,
                                 panel, draws::Vector{Int}, P::Dict, rho_tau::Float64,
                                 voll::Float64, lolp_thr::Float64;
                                 prechecks = nothing,
                                 case_by_draw = nothing,
                                 target_contexts = nothing,
                                 bypass_prescreen::Bool = true,
                                 failure_policy::AbstractString = "drop")
    rows = Vector{Any}(); times = Float64[]; online_times = Float64[]
    total_times = Float64[]; profile_times = Float64[]
    calibration_times = Float64[]
    all_attempt_online_times = Float64[]
    all_attempt_total_times = Float64[]
    kappas = Float64[]; kappads = Float64[]; nfail = 0
    n_prescreen_ok = 0; n_model_called = 0
    log = DataFrame()
    audit_tol_mw = _audit_tol_mw(P)
    case_by_draw === nothing ||
        length(case_by_draw) == length(draws) ||
        throw(DimensionMismatch("case_by_draw length must equal draw count"))
    prechecks === nothing ||
        length(prechecks) == length(draws) ||
        throw(DimensionMismatch("prechecks length must equal draw count"))
    _validate_target_contexts(target_contexts, draws;
                              require_draw_index = case_by_draw !== nothing)
    failure_policy = lowercase(strip(failure_policy))
    failure_policy in ("drop", "conservative_full_shed") ||
        error("unknown screened-evaluation failure policy: $failure_policy")

    for (draw_id, t) in enumerate(draws)
        cdi = case_by_draw === nothing ? cd : case_by_draw[draw_id]
        snap = snapshot_at(panel, t, cdi)
        pc = prechecks === nothing ? dc_state_precheck(cdi, snap) : prechecks[draw_id]
        n_gen_out = count(cdi.pmax .<= 0.0) - count(cd.pmax .<= 0.0)
        available_capacity = sum(cdi.pmax)
        pc.ok && (n_prescreen_ok += 1)
        if pc.ok && bypass_prescreen
            push!(rows, _precheck_row(pc))
            push!(log, merge((system = system, model = model, mc_draw = draw_id,
                              snapshot = t, precheck_ok = true,
                              generator_outages = max(n_gen_out, 0),
                              available_capacity_mw = available_capacity,
                              precheck_status = pc.status,
                              precheck_time = round(pc.time, digits = 4),
                              precheck_max_flow = round(pc.max_flow_ratio, digits = 4),
                              called_model = false, status = "PRESCREEN_OK",
                               solve_error = "",
                               solver_attempts = 0, primary_status = "PRESCREEN_OK",
                               retry_status = "",
                               attempt_wall = 0.0,
                              build_time = 0.0, solve_wall = 0.0, solver_time = 0.0,
                              online_wall = 0.0,
                              online_wall_all_attempts = 0.0,
                              internal_calibration_wall = 0.0,
                              internal_model_wall = 0.0,
                              model_wall = 0.0, end_to_end_wall = 0.0,
                              end_to_end_wall_all_attempts = 0.0,
                              iters = 0, n_var = 0, n_con = 0, obj = pc.gen_cost,
                              accepted = true, accounted = true,
                              jump_report_distance = 0.0,
                               primal_residual = 0.0, dual_residual = 0.0,
                               gap_abs = 0.0, gap_rel = 0.0,
                               physical_violation = 0.0,
                               physical_violation_raw_ratio = 0.0,
                               physical_violation_label = "prescreen_ok",
                               physical_violation_raw = 0.0,
                               physical_violation_scale = 1.0,
                               physical_absolute_tolerance = 0.0,
                               kappa_c = NaN, kappa_d = NaN, Z0 = NaN, Z0_solve = NaN,
                              Z0_build = NaN),
                             _target_log_fields(nothing),
                             _profile_log_fields(nothing, P, model)); promote = true)
            continue
        end

        n_model_called += 1
        # Resolve and validate outside the solver try/catch.  A missing or
        # mismatched common context is a protocol error, never a reliability
        # outcome that may be converted into conservative full shedding.
        resolved = _resolve_target_context(target_contexts, draw_id, model,
                                           cdi, snap, scen, P, rho_tau)
        u = nothing; st = "NOT_SOLVED"; solve_error = ""
        attempt_t0 = time()
        try
            u = _solve_one(model, cdi, snap, scen, P, rho_tau;
                           bandwidth_profile = resolved.bandwidth_profile,
                           target_meta_override = resolved.target_meta_override)
            st = string(u.status)
        catch err
            st = "EXCEPTION"
            solve_error = sprint(showerror, err)
        end
        attempt_wall = time() - attempt_t0
        ok = u !== nothing && _solution_accepted(u, P)
        timing = u === nothing ? nothing :
                 _evaluation_timing(u, resolved.context, model)
        context_timing = _context_timing(resolved.context, model)
        attempt_online = u === nothing ? attempt_wall : timing.online
        attempt_total = u === nothing ? attempt_wall + context_timing.charged :
                                       timing.end_to_end
        push!(all_attempt_online_times, attempt_online)
        push!(all_attempt_total_times, attempt_total)
        if ok
            push!(times, u.solve_time)
            push!(online_times, timing.online)
            push!(total_times, timing.end_to_end)
            push!(profile_times, timing.profile)
            push!(calibration_times, timing.calibration)
            u.is_rs && push!(kappas, u.κc)
            isfinite(u.κd) && push!(kappads, u.κd)
            push!(rows, audit_snapshot(cdi, u.x, snap;
                                       tol_mw = audit_tol_mw, βbar = u.βb))
        else
            nfail += 1
            failure_policy == "conservative_full_shed" &&
                push!(rows, _conservative_failure_row(snap))
        end
        push!(log, merge((system = system, model = model, mc_draw = draw_id,
                          snapshot = t, precheck_ok = pc.ok,
                          generator_outages = max(n_gen_out, 0),
                          available_capacity_mw = available_capacity,
                          precheck_status = pc.status,
                          precheck_time = round(pc.time, digits = 4),
                          precheck_max_flow = pc.max_flow_ratio,
                          called_model = true, status = st,
                           solve_error = solve_error,
                           solver_attempts = u === nothing ? 0 : u.solver_attempts,
                           primary_status = u === nothing ? "EXCEPTION" : u.primary_status,
                           retry_status = u === nothing ? "" : u.retry_status,
                           attempt_wall = round(attempt_wall, digits = 4),
                          build_time = u === nothing ? NaN : round(u.build_time, digits = 4),
                          solve_wall = u === nothing ? NaN : round(u.solve_time, digits = 4),
                          online_wall = u === nothing ? NaN : round(timing.online, digits = 4),
                          internal_calibration_wall = u === nothing ? NaN :
                              round(timing.internal_calibration, digits = 4),
                          internal_model_wall = u === nothing ? NaN :
                              round(_model_wall(u), digits = 4),
                          model_wall = u === nothing ? NaN :
                              round(timing.end_to_end, digits = 4),
                          end_to_end_wall = u === nothing ? NaN :
                              round(timing.end_to_end, digits = 4),
                          solver_time = u === nothing ? NaN : round(u.solver_time, digits = 4),
                          iters = u === nothing ? -1 : u.iters,
                          n_var = u === nothing ? -1 : u.n_var,
                          n_con = u === nothing ? -1 : u.n_con,
                          accepted = ok,
                          accounted = ok || failure_policy == "conservative_full_shed",
                          online_wall_all_attempts = round(attempt_online, digits = 4),
                          end_to_end_wall_all_attempts = round(attempt_total, digits = 4),
                          jump_report_distance = u === nothing ? Inf : u.jump_report_distance,
                          primal_residual = u === nothing ? Inf : u.primal_residual,
                          dual_residual = u === nothing ? Inf : u.dual_residual,
                           gap_abs = u === nothing ? Inf : u.gap_abs,
                           gap_rel = u === nothing ? Inf : u.gap_rel,
                           physical_violation = u === nothing ? Inf : u.physical_violation,
                           physical_violation_raw_ratio = u === nothing ? Inf : u.physical_violation_raw_ratio,
                           physical_violation_label = u === nothing ? "no_solution" : u.physical_violation_label,
                           physical_violation_raw = u === nothing ? Inf : u.physical_violation_raw,
                           physical_violation_scale = u === nothing ? 1.0 : u.physical_violation_scale,
                           physical_absolute_tolerance = u === nothing ? 0.0 : u.physical_absolute_tolerance,
                           obj = u === nothing ? NaN : u.obj,
                          kappa_c = u === nothing ? NaN : u.κc,
                          kappa_d = u === nothing ? NaN : u.κd,
                          Z0 = u === nothing ? NaN : u.z0,
                          Z0_solve = u === nothing ? NaN : round(u.z0_solve, digits = 4),
                          Z0_build = u === nothing ? NaN : round(u.z0_build, digits = 4)),
                         _target_log_fields(u === nothing ? nothing : u.target_meta),
                         _profile_log_fields(resolved.context, P, model)); promote = true)
    end
    n_accounted = length(rows)
    n_accepted = length(draws) - nfail
    agg = n_accounted > 0 ? aggregate_audit(rows, 1.0, voll, lolp_thr) :
          Audit.AuditAgg(0, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    prec = _edns_precision(rows)
    return (agg = agg, time = isempty(times) ? 0.0 : mean(times),
            time_max = isempty(times) ? 0.0 : maximum(times),
            time_online = isempty(online_times) ? 0.0 : mean(online_times),
            time_online_max = isempty(online_times) ? 0.0 : maximum(online_times),
            time_total = isempty(total_times) ? 0.0 : mean(total_times),
            time_total_max = isempty(total_times) ? 0.0 : maximum(total_times),
             time_end_to_end = isempty(total_times) ? 0.0 : mean(total_times),
             time_end_to_end_max = isempty(total_times) ? 0.0 : maximum(total_times),
             time_online_all_attempts = isempty(all_attempt_online_times) ? 0.0 :
                                        mean(all_attempt_online_times),
             time_online_all_attempts_max = isempty(all_attempt_online_times) ? 0.0 :
                                            maximum(all_attempt_online_times),
             time_end_to_end_all_attempts = isempty(all_attempt_total_times) ? 0.0 :
                                            mean(all_attempt_total_times),
             time_end_to_end_all_attempts_max = isempty(all_attempt_total_times) ? 0.0 :
                                                maximum(all_attempt_total_times),
            context_profile_time = isempty(profile_times) ? 0.0 : mean(profile_times),
            context_calibration_time = isempty(calibration_times) ? 0.0 : mean(calibration_times),
            kappa_c = isempty(kappas) ? NaN : mean(kappas),
            kappa_d = isempty(kappads) ? NaN : mean(kappads),
            edns_sd = prec.sd, edns_se = prec.se, cov_edns = prec.cov,
             fail_rate = nfail / length(draws), n_ok = n_accepted,
             n_accepted = n_accepted, n_accounted = n_accounted,
             n_all_attempt_states = length(all_attempt_online_times),
             audit_tol_mw = audit_tol_mw,
             metrics_scope = failure_policy == "conservative_full_shed" ?
                             "common_denominator_worst_case_failures" :
                             "accepted_conditional_drop_failures",
             log = log,
            prescreen_ok = n_prescreen_ok,
            prescreen_ok_rate = n_prescreen_ok / length(draws),
             model_called = n_model_called,
             model_call_rate = n_model_called / length(draws),
             prescreen_bypass_enabled = bypass_prescreen,
             failure_policy = failure_policy)
end

function run_fixed_experiment(cfg::Dict, root::String)
    P = merge(cfg["model"], get(cfg, "baseline", Dict{String,Any}()))  # 含 M5 参数
    ρτ = get(P, "rho_tau", get(P, "rho_cost", 0.0)); voll = P["voll"]
    lolp_thr = cfg["output"]["lolp_threshold"]
    out = DataFrame(); timing = DataFrame()
    for system in cfg["run"]["systems"]
        res = load_or_run_pipeline(system, cfg, root)
        panel = res.panel
        gk = hasproperty(res, :gen_keep) ? res.gen_keep : nothing
        cd = build_casedata(res.case, panel.load_buses, panel.wind_buses, panel.wind_cap, cfg;
                            gen_keep = gk)
        scen = build_scenariodata(res.scenarios)
        te = collect(res.split.test)
        caps = get(cfg["run"], "audit_caps", Dict{String,Any}())
        cap = get(caps, system, get(cfg["run"], "max_audit_snapshots", length(te)))
        stride = max(1, cld(length(te), cap))
        tset = te[1:stride:end]
        @info "系统 $system: 审计快照 $(length(tset)) 个 (stride=$stride)  nE=$(cd.nE)"
        for model in cfg["run"]["models"]
            r = eval_model(system, model, cd, scen, panel, tset, P, ρτ, voll, lolp_thr)
            a = r.agg
            append!(timing, r.log; promote = true)
            push!(out, (system = system, model = model,
                        kappa_c = round(r.κc, digits = 4),
                        kappa_d = round(r.κd, digits = 4),
                        Cost = round(a.Cost, digits = 2),
                        EDNS = round(a.EDNS, digits = 4),
                        EENS = round(a.EENS, digits = 2),
                        LOLP = round(a.LOLP, digits = 4),
                        LOLP_thr = round(a.LOLP_thr, digits = 4),
                        line_prob = round(a.line_prob, digits = 4),
                        reserve_prob = round(a.reserve_prob, digits = 4),
                        shedbound_prob = round(a.shedbound_prob, digits = 4),
                        max_flow_ratio = round(a.max_flow_ratio, digits = 3),
                        AvgCurtail = round(a.AvgCurtail, digits = 3),
                        WindCurtailAudit = round(a.WindCurtailAudit, digits = 4),
                        Time = round(r.time, digits = 3),
                        Time_max = round(r.time_max, digits = 3),
                        n_audit = r.n_ok,
                        fail_rate = round(r.fail_rate, digits = 3)); promote = true)
            @printf("  %-9s %-3s  κc=%-8s Cost=%-9.1f EDNS=%-7.3f LOLP=%-6.3f line=%-6.3f res=%-6.3f t=%.2fs\n",
                    system, model, string(round(r.κc, digits = 3)),
                    a.Cost, a.EDNS, a.LOLP, a.line_prob, a.reserve_prob, r.time)
        end
    end
    rdir = joinpath(root, cfg["run"]["results_dir"]); mkpath(rdir)
    # 每系统单独存一份, 避免分次运行互相覆盖
    for system in unique(out.system)
        CSV.write(joinpath(rdir, "main_$(system).csv"), out[out.system .== system, :])
        CSV.write(joinpath(rdir, "timing_$(system).csv"), timing[timing.system .== system, :])
    end
    CSV.write(joinpath(rdir, "main_results.csv"), out)
    CSV.write(joinpath(rdir, "timing_all.csv"), timing)   # 逐实例计时/精确度基准
    @info "主结果表 + 逐实例计时日志已写入 $rdir"
    return (summary = out, timing = timing)
end

function _mc_draws(system::String, te::Vector{Int}, cfg::Dict)
    mc = get(cfg, "monte_carlo", Dict{String,Any}())
    caps = get(mc, "n_samples_by_system", Dict{String,Any}())
    nmc = Int(get(caps, system, get(mc, "n_samples", min(length(te), 60))))
    replace = get(mc, "replace", true)
    seed = Int(get(mc, "seed", get(cfg["run"], "seed", 20260615)))
    rng = StableRNG(seed + sum(Int, codeunits(system)))
    if replace
        return [te[rand(rng, 1:length(te))] for _ in 1:nmc]
    end
    n = min(nmc, length(te))
    return te[randperm(rng, length(te))[1:n]]
end

"Non-sequential Monte Carlo over operating states sampled from the test pool."
function run_nsmc_experiment(cfg::Dict, root::String)
    P = merge(cfg["model"], get(cfg, "baseline", Dict{String,Any}()))  # 含 M5 参数
    rho_tau = get(P, "rho_tau", get(P, "rho_cost", 0.0)); voll = P["voll"]
    lolp_thr = cfg["output"]["lolp_threshold"]
    rel = get(cfg, "reliability", Dict{String,Any}())
    annual_hours = Float64(get(rel, "annual_hours", 8760.0))
    out = DataFrame(); timing = DataFrame(); draws_out = DataFrame()
    for system in cfg["run"]["systems"]
        res = load_or_run_pipeline(system, cfg, root)
        panel = res.panel
        gk = hasproperty(res, :gen_keep) ? res.gen_keep : nothing
        cd = build_casedata(res.case, panel.load_buses, panel.wind_buses, panel.wind_cap, cfg;
                            gen_keep = gk)
        scen = build_scenariodata(res.scenarios)
        te = collect(res.split.test)
        draws = _mc_draws(system, te, cfg)
        for (i, t) in enumerate(draws)
            push!(draws_out, (system = system, mc_draw = i, snapshot = t); promote = true)
        end
        @info "系统 $system: 非时序MC运行状态 $(length(draws)) 个 nE=$(cd.nE)"
        for model in cfg["run"]["models"]
            r = eval_model_mc(system, model, cd, scen, panel, draws, P, rho_tau, voll, lolp_thr)
            a = r.agg
            append!(timing, r.log; promote = true)
            push!(out, (system = system, model = model,
                        mc_states = length(draws),
                        kappa_c = round(r.kappa_c, digits = 4),
                        kappa_d = round(r.kappa_d, digits = 4),
                        Cost = round(a.Cost, digits = 2),
                        EDNS = round(a.EDNS, digits = 4),
                        EDNS_sd = round(r.edns_sd, digits = 4),
                        EDNS_se = round(r.edns_se, digits = 4),
                        COV_EDNS = round(r.cov_edns, digits = 4),
                        EENS = round(annual_hours * a.EDNS, digits = 2),
                        EENS_sample = round(a.EENS, digits = 2),
                        T_hours = annual_hours,
                        LOLP = round(a.LOLP, digits = 4),
                        LOLP_thr = round(a.LOLP_thr, digits = 4),
                        line_prob = round(a.line_prob, digits = 4),
                        reserve_prob = round(a.reserve_prob, digits = 4),
                        shedbound_prob = round(a.shedbound_prob, digits = 4),
                        max_flow_ratio = round(a.max_flow_ratio, digits = 3),
                        AvgCurtail = round(a.AvgCurtail, digits = 3),
                        WindCurtailAudit = round(a.WindCurtailAudit, digits = 4),
                        Time = round(r.time, digits = 3),
                        Time_max = round(r.time_max, digits = 3),
                        n_audit = r.n_ok,
                        fail_rate = round(r.fail_rate, digits = 3)); promote = true)
            @printf("  MC %-6s %-3s  states=%-4d Cost=%-9.1f EDNS=%-7.3f LOLP=%-6.3f line=%-6.3f res=%-6.3f t=%.2fs\n",
                    system, model, length(draws), a.Cost, a.EDNS, a.LOLP,
                    a.line_prob, a.reserve_prob, r.time)
        end
    end
    rdir = joinpath(root, cfg["run"]["results_dir"]); mkpath(rdir)
    for system in unique(out.system)
        CSV.write(joinpath(rdir, "nsmc_$(system).csv"), out[out.system .== system, :])
        CSV.write(joinpath(rdir, "nsmc_timing_$(system).csv"), timing[timing.system .== system, :])
    end
    CSV.write(joinpath(rdir, "nsmc_results.csv"), out)
    CSV.write(joinpath(rdir, "nsmc_timing_all.csv"), timing)
    CSV.write(joinpath(rdir, "nsmc_draws.csv"), draws_out)
    @info "非时序MC结果表 + 逐状态计时日志已写入 $rdir"
    return (summary = out, timing = timing, draws = draws_out)
end

end # module
