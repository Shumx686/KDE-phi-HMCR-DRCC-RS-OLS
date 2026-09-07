"""
FinalSevenTargetModel

Executable implementation of the final model in
`KDE-ϕ-HMCR-RS-DC-OLS.tex`, equations (12)--(128).

The module is intentionally isolated from the legacy `ModelCore` and
`ScalarMWModel` implementations.  In particular it has exactly one MW cost
loss, six class-max normalized safety losses, one cost fragility, and six
independent safety fragilities.  It contains no `k_m` scale factors, no shared
`kappa_d`, no VOLL multiplication in the cost loss, no clipping of sample
load-shedding, and no fixed weighted-sum regularizer.  When the primary
fragility optimum is numerically zero, a solver-audited lexicographic second
stage minimizes expected nonnegative load shedding without relaxing that
primary optimum beyond a predeclared numerical tolerance.
"""
module FinalSevenTargetModel

using JuMP
using Clarabel
using LinearAlgebra
using Statistics
import MathOptInterface as MOI

# This runtime-only compatibility layer replaces allocation-heavy Clarabel
# KKT, SOC-workspace, and MOI triplet-growth operations.  It is not an
# optimization-model change; see its source header and
# RTS79_CALIBRATION_RUNTIME_AUDIT_20260729.md for the exact failure and
# equality evidence.
include(joinpath(@__DIR__, "ClarabelKKTMemoryPatch.jl"))
using .ClarabelKKTMemoryPatch

using Main.DRCCExp.CaseInterface: CaseData, ScenarioData, Snapshot,
                                      sample_safe_curtailment_cap

export FinalBandwidth, FinalCalibration, FinalResult,
       solve_no_smoothing_pilot, estimate_bandwidths,
       calibrate_reference_targets, retarget,
       solve_final_model, solve_cost_reference, evaluate_losses, evaluate_hmcr

"Fixed, pre-solve KDE bandwidths in the native loss units of the final model."
struct FinalBandwidth
    cost_h::Float64
    safety_h::Vector{Float64}
    raw_n::Int
    multiplier::Float64
    floor_cost::Float64
    floor_safety::Float64
end

"The seven reference objectives and the fixed final targets derived from them."
struct FinalCalibration
    Z0c::Float64
    Z0m::Vector{Float64}
    tau_c::Float64
    tau_m::Vector{Float64}
    rho_tau_c::Float64
    rho_tau_s::Float64
    bandwidth::FinalBandwidth
    reference_status::Vector{String}
    reference_primal_residual::Vector{Float64}
    reference_dual_residual::Vector{Float64}
    reference_gap_abs::Vector{Float64}
    reference_gap_rel::Vector{Float64}
end

# Backward-compatible constructor for diagnostic callers that supply only
# statuses. Formal experiment paths populate the four quality vectors above.
FinalCalibration(Z0c::Float64, Z0m::Vector{Float64}, tau_c::Float64,
                 tau_m::Vector{Float64}, rho_tau_c::Float64, rho_tau_s::Float64,
                 bandwidth::FinalBandwidth, reference_status::Vector{String}) =
    FinalCalibration(Z0c, Z0m, tau_c, tau_m, rho_tau_c, rho_tau_s, bandwidth,
                     reference_status, fill(NaN, length(reference_status)),
                     fill(NaN, length(reference_status)),
                     fill(NaN, length(reference_status)),
                     fill(NaN, length(reference_status)))

"Solution of the exact finite-sample final model."
struct FinalResult
    status
    x::NamedTuple
    kappa_c::Float64
    kappa_m::Vector{Float64}
    objective::Float64
    secondary_objective::Float64
    lexicographic_enabled::Bool
    lexicographic_activated::Bool
    lexicographic_stage1_status::String
    lexicographic_stage2_status::String
    lexicographic_primary_star::Float64
    lexicographic_primary_tolerance::Float64
    lexicographic_primary_final::Float64
    stage1_solve_time::Float64
    stage2_solve_time::Float64
    solve_time::Float64
    build_time::Float64
    n_var::Int
    n_con::Int
    primal_residual::Float64
    dual_residual::Float64
    gap_abs::Float64
    gap_rel::Float64
end

const CLASS_LABELS = ("reserve_up", "reserve_down", "line_pos", "line_neg",
                      "shed_low", "shed_high")

function _normal_nodes(Q::Int)
    Q >= 1 || error("quadrature order must be positive")
    Q == 1 && return ([0.0], [1.0])
    jacobi = SymTridiagonal(zeros(Q), sqrt.(Float64.(1:Q-1)))
    eig = eigen(jacobi)
    nodes = collect(eig.values)
    weights = vec(eig.vectors[1, :] .^ 2)
    weights ./= sum(weights)
    return nodes, weights
end

function _model(P::Dict)
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "max_iter", Int(get(P, "max_iter", 3000)))
    set_optimizer_attribute(m, "equilibrate_enable", true)
    set_optimizer_attribute(m, "tol_gap_abs", Float64(get(P, "tol_gap_abs", 1e-8)))
    set_optimizer_attribute(m, "tol_gap_rel", Float64(get(P, "tol_gap_rel", 1e-8)))
    set_optimizer_attribute(m, "tol_feas", Float64(get(P, "tol_feas", 1e-8)))
    set_optimizer_attribute(m, "reduced_tol_gap_abs", Float64(get(P, "reduced_tol_gap_abs", 5e-5)))
    set_optimizer_attribute(m, "reduced_tol_gap_rel", Float64(get(P, "reduced_tol_gap_rel", 1e-5)))
    set_optimizer_attribute(m, "reduced_tol_feas", Float64(get(P, "reduced_tol_feas", 1e-5)))
    haskey(P, "direct_solve_method") &&
        set_optimizer_attribute(m, "direct_solve_method", Symbol(P["direct_solve_method"]))
    haskey(P, "equilibrate_max_iter") &&
        set_optimizer_attribute(m, "equilibrate_max_iter", Int(P["equilibrate_max_iter"]))
    haskey(P, "iterative_refinement_max_iter") &&
        set_optimizer_attribute(m, "iterative_refinement_max_iter", Int(P["iterative_refinement_max_iter"]))
    haskey(P, "dynamic_regularization_delta") &&
        set_optimizer_attribute(m, "dynamic_regularization_delta", Float64(P["dynamic_regularization_delta"]))
    haskey(P, "static_regularization_constant") &&
        set_optimizer_attribute(m, "static_regularization_constant", Float64(P["static_regularization_constant"]))
    haskey(P, "static_regularization_proportional") &&
        set_optimizer_attribute(m, "static_regularization_proportional", Float64(P["static_regularization_proportional"]))
    haskey(P, "min_terminate_step_length") &&
        set_optimizer_attribute(m, "min_terminate_step_length", Float64(P["min_terminate_step_length"]))
    return m
end

function _quality(m)
    try
        info = unsafe_backend(m).solver_info
        return (primal = Float64(info.res_primal), dual = Float64(info.res_dual),
                gap_abs = Float64(info.gap_abs), gap_rel = Float64(info.gap_rel))
    catch
        return (primal = NaN, dual = NaN, gap_abs = NaN, gap_rel = NaN)
    end
end

"Build exactly X_phys and the seven sample losses.  `ScenarioData.Ω` is the
internal net-injection error, i.e. the negative of the paper's net-load error."
function _physics_and_losses!(m, cd::CaseData, snap::Snapshot, scen::ScenarioData;
                              responsive_loads = trues(cd.nD))
    nG, nD, nW, nE, K = cd.nG, cd.nD, cd.nW, cd.nE, scen.K
    length(responsive_loads) == nD || error("responsive-load mask has wrong length")

    @variable(m, g[1:nG])
    @variable(m, rU[1:nG] >= 0)
    @variable(m, rD[1:nG] >= 0)
    @variable(m, snom[1:nD] >= 0)
    @variable(m, cW[1:nW] >= 0)
    @variable(m, beta[1:nG] >= 0)
    @variable(m, alphaS[1:nD] >= 0)

    @constraint(m, g .+ rU .<= cd.pmax)
    @constraint(m, g .- rD .>= cd.pmin)
    @constraint(m, snom .<= snap.lf)
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    @constraint(m, cW .<= curtail_cap)
    @constraint(m, sum(g) == sum(snap.lf) - sum(snap.wf) + sum(cW) - sum(snom))
    @constraint(m, sum(beta) + sum(alphaS) == 1)
    for j in 1:nG
        if !cd.agc_mask[j]
            @constraint(m, rU[j] == 0)
            @constraint(m, rD[j] == 0)
            @constraint(m, beta[j] == 0)
        end
    end
    for d in 1:nD
        responsive_loads[d] || @constraint(m, alphaS[d] == 0)
    end

    f0 = [sum(cd.MG[ell, j] * g[j] for j in 1:nG) +
          sum(cd.MW[ell, w] * (snap.wf[w] - cW[w]) for w in 1:nW) -
          sum(cd.MD[ell, d] * (snap.lf[d] - snom[d]) for d in 1:nD)
          for ell in 1:nE]
    for ell in 1:nE
        isfinite(cd.F_max[ell]) || continue
        @constraint(m, f0[ell] <= cd.F_max[ell])
        @constraint(m, -f0[ell] <= cd.F_max[ell])
    end

    # The following epigraph variables are solver representations of the
    # class maxima q_m^(i), not additional physical decisions.
    q_up = [Any[] for _ in 1:K]
    q_down = [Any[] for _ in 1:K]
    q_pos = [Any[] for _ in 1:K]
    q_neg = [Any[] for _ in 1:K]
    q_slow = [Any[] for _ in 1:K]
    q_shigh = [Any[] for _ in 1:K]

    for j in 1:nG
        cd.agc_mask[j] || continue
        scale = max(cd.pmax[j], 1.0)
        for i in 1:K
            # Ω_paper=-Ω_internal: Ω_paper β-rU and -Ω_paper β-rD.
            push!(q_up[i], (-scen.Ω[i] * beta[j] - rU[j]) / scale)
            push!(q_down[i], (scen.Ω[i] * beta[j] - rD[j]) / scale)
        end
    end

    response = [sum(cd.MG[ell, j] * beta[j] for j in 1:nG) +
                sum(cd.MD[ell, d] * alphaS[d] for d in 1:nD)
                for ell in 1:nE]
    PTDF_delta = scen.δ * cd.PTDF'
    for ell in 1:nE
        isfinite(cd.F_max[ell]) || continue
        scale = max(cd.F_max[ell], 1.0)
        for i in 1:K
            # f=f0+PTDF*delta_internal-Ω_internal(MG beta+MD alphaS).
            f = f0[ell] + PTDF_delta[i, ell] - scen.Ω[i] * response[ell]
            push!(q_pos[i], (f - cd.F_max[ell]) / scale)
            push!(q_neg[i], (-f - cd.F_max[ell]) / scale)
        end
    end

    for d in 1:nD
        scale = max(snap.lf[d], 1.0)
        for i in 1:K
            # s=s0+Ω_paper alphaS=s0-Ω_internal alphaS.
            s = snom[d] - scen.Ω[i] * alphaS[d]
            push!(q_slow[i], -s / scale)
            push!(q_shigh[i], (s - (snap.lf[d] + scen.eL[i, d])) / scale)
        end
    end

    function class_max(parts)
        values = Vector{Any}(undef, K)
        for i in 1:K
            if isempty(parts[i])
                values[i] = 0.0
            else
                q = @variable(m)
                for term in parts[i]
                    @constraint(m, q >= term)
                end
                values[i] = q
            end
        end
        return values
    end

    safety = [class_max(q_up), class_max(q_down), class_max(q_pos),
              class_max(q_neg), class_max(q_slow), class_max(q_shigh)]
    # Exact final-paper cost loss: total sample load shedding in MW, no VOLL
    # and no nonnegative clipping.
    cost = [sum(snom[d] - scen.Ω[i] * alphaS[d] for d in 1:nD) for i in 1:K]
    return (g = g, rU = rU, rD = rD, snom = snom, cW = cW,
            beta = beta, alphaS = alphaS, cost = cost, safety = safety)
end

function _fix_x!(m, v, x)
    for (field, vars) in ((:g, v.g), (:rU, v.rU), (:rD, v.rD),
                          (:snom, v.snom), (:cW, v.cW),
                          (:beta, v.beta), (:alphaS, v.alphaS))
        values = getproperty(x, field)
        length(values) == length(vars) || error("fixed-x dimension mismatch at $field")
        for k in eachindex(vars)
            @constraint(m, vars[k] == values[k])
        end
    end
end

function _kde_hinge!(m, loss, alpha, h::Float64, p::Int, nodes, weights)
    h >= 0 || error("KDE bandwidth must be nonnegative")
    z = @variable(m, lower_bound = 0)
    t = @variable(m, [1:length(nodes)], lower_bound = 0)
    for q in eachindex(nodes)
        @constraint(m, t[q] >= loss - alpha - h * nodes[q])
    end
    if p == 1
        @constraint(m, z >= sum(weights[q] * t[q] for q in eachindex(nodes)))
    elseif p == 2
        @constraint(m, [z; [sqrt(weights[q]) * t[q] for q in eachindex(nodes)]] in SecondOrderCone())
    else
        error("the RBTS final-model runner supports p=1 or p=2; received p=$p")
    end
    return z
end

function _add_hmcr!(m, losses, pi, gamma::Float64, h::Float64, p::Int, nodes, weights)
    0 < gamma < 1 || error("all final-model HMCR confidence levels must lie in (0,1)")
    alpha = @variable(m)
    z = [_kde_hinge!(m, losses[i], alpha, h, p, nodes, weights) for i in eachindex(losses)]
    tail = @variable(m, lower_bound = 0)
    if p == 1
        @constraint(m, tail >= sum(pi[i] * z[i] for i in eachindex(losses)))
    else
        @constraint(m, [tail; [sqrt(pi[i]) * z[i] for i in eachindex(losses)]] in SecondOrderCone())
    end
    return alpha + tail / (1 - gamma)
end

"Closed perspective of the Pearson generator phi(t)=(t-1)^2 on t>=0."
function _pearson_perspective!(m, s, mu)
    t = @variable(m, lower_bound = 0)
    g = @variable(m, lower_bound = 0)
    @constraint(m, t >= s + 2 * mu)
    @constraint(m, [2 * mu, g, t] in RotatedSecondOrderCone())
    return g - mu
end

function _add_rs_constraint!(m, losses, pi, tau::Float64, kappa,
                             gamma::Float64, h::Float64, p::Int,
                             nodes, weights, eps_y::Float64)
    eps_y > 0 || error("eps_y must be strictly positive")
    alpha = @variable(m)
    z = [_kde_hinge!(m, losses[i], alpha, h, p, nodes, weights) for i in eachindex(losses)]
    eta = @variable(m)
    v = @variable(m, [1:length(losses)], lower_bound = 0)
    if p == 1
        for i in eachindex(losses)
            @constraint(m, v[i] >= z[i] / (1 - gamma))
        end
        perspective_sum = sum(pi[i] * _pearson_perspective!(m, v[i] - eta, kappa)
                              for i in eachindex(losses))
        @constraint(m, alpha - tau + eta + perspective_sum <= 0)
    elseif p == 2
        y = @variable(m, lower_bound = eps_y)
        for i in eachindex(losses)
            # z_i^2/[2(1-gamma)y] <= v_i.
            @constraint(m, [(1 - gamma) * y, v[i], z[i]] in RotatedSecondOrderCone())
        end
        perspective_sum = sum(pi[i] * _pearson_perspective!(m, v[i] - eta, kappa)
                              for i in eachindex(losses))
        @constraint(m, alpha - tau + y / (2 * (1 - gamma)) + eta + perspective_sum <= 0)
    else
        error("the RBTS final-model runner supports p=1 or p=2; received p=$p")
    end
end

function _extract_x(v)
    return (g = value.(v.g), rU = value.(v.rU), rD = value.(v.rD),
            snom = value.(v.snom), cW = value.(v.cW), beta = value.(v.beta),
            alphaS = value.(v.alphaS))
end

function _empty_x(v)
    return (g = fill(NaN, length(v.g)), rU = fill(NaN, length(v.rU)),
            rD = fill(NaN, length(v.rD)), snom = fill(NaN, length(v.snom)),
            cW = fill(NaN, length(v.cW)), beta = fill(NaN, length(v.beta)),
            alphaS = fill(NaN, length(v.alphaS)))
end

function _result(m, v, kappa_c, kappa_m, objective, build_time;
                 lex_secondary_builder = nothing,
                 lex_abs_tol::Float64 = 0.0,
                 lex_rel_tol::Float64 = 0.0,
                 lex_gap_multiplier::Float64 = 0.0,
                 lex_activation_tol::Float64 = Inf)
    lex_abs_tol >= 0.0 || error("lexicographic absolute tolerance must be nonnegative")
    lex_rel_tol >= 0.0 || error("lexicographic relative tolerance must be nonnegative")
    lex_gap_multiplier >= 0.0 || error("lexicographic gap multiplier must be nonnegative")
    lex_activation_tol >= 0.0 || error("lexicographic activation tolerance must be nonnegative")
    lex_enabled = lex_secondary_builder !== nothing
    t0 = time()
    optimize!(m)
    stage1_time = time() - t0
    stage1_status = termination_status(m)
    stage1_good = string(stage1_status) in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(m)
    stage1_quality = _quality(m)
    x = stage1_good ? _extract_x(v) : _empty_x(v)
    if kappa_c === nothing || kappa_m === nothing
        kc = 0.0
        km = zeros(6)
    else
        kc = stage1_good ? value(kappa_c) : NaN
        km = stage1_good ? value.(kappa_m) : fill(NaN, 6)
    end
    primary_star = stage1_good ? value(objective) : NaN
    primary_final = primary_star
    secondary_final = NaN
    primary_tol = 0.0
    stage2_time = 0.0
    stage2_status = "NOT_RUN"
    lex_activated = false
    final_status = stage1_status
    final_quality = stage1_quality

    if lex_enabled && stage1_good && primary_star <= lex_activation_tol
        gap_term = isfinite(stage1_quality.gap_abs) ?
                   lex_gap_multiplier * abs(stage1_quality.gap_abs) : 0.0
        primary_tol = max(lex_abs_tol, lex_rel_tol * abs(primary_star), gap_term)
        setup_start = time()
        @constraint(m, objective <= max(primary_star, 0.0) + primary_tol)
        secondary = lex_secondary_builder(m)
        @objective(m, Min, secondary)
        build_time += time() - setup_start
        t2 = time()
        optimize!(m)
        stage2_time = time() - t2
        stage2_moi_status = termination_status(m)
        stage2_status = string(stage2_moi_status)
        stage2_good = stage2_status in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(m)
        primary_limit = max(primary_star, 0.0) + primary_tol
        if stage2_good && value(objective) <= primary_limit + max(lex_abs_tol, 1e-9)
            lex_activated = true
            final_status = stage2_moi_status
            final_quality = _quality(m)
            x = _extract_x(v)
            primary_final = value(objective)
            secondary_final = value(secondary)
            if kappa_c !== nothing && kappa_m !== nothing
                kc = value(kappa_c)
                km = value.(kappa_m)
            end
        else
            # Retain the accepted stage-1 decision if the optional tie-break
            # solve fails.  The stage-2 status remains explicit in the audit.
            stage2_status *= stage2_good ? "_PRIMARY_LOCK_VIOLATION" : "_FALLBACK_STAGE1"
        end
    elseif lex_enabled && stage1_good
        stage2_status = "SKIPPED_ACTIVE_PRIMARY"
    end

    nvar = num_variables(m)
    ncon = num_constraints(m; count_variable_in_set_constraints = true)
    obj = stage1_good ? primary_final : NaN
    return FinalResult(final_status, x, kc, Float64.(km), obj, secondary_final,
                       lex_enabled, lex_activated, string(stage1_status), stage2_status,
                       primary_star, primary_tol, primary_final,
                       stage1_time, stage2_time, stage1_time + stage2_time,
                       build_time, nvar, ncon, final_quality.primal,
                       final_quality.dual, final_quality.gap_abs, final_quality.gap_rel)
end

"Add the common MW-valued operating-cost tie-break only after stage 1."
function _lexicographic_shed_objective!(m, v, cd::CaseData,
                                        scen::ScenarioData, P::Dict)
    @variable(m, lex_shed[1:scen.K, 1:cd.nD] >= 0)
    for i in 1:scen.K, d in 1:cd.nD
        @constraint(m, lex_shed[i, d] >= v.snom[d] - scen.Ω[i] * v.alphaS[d])
    end
    expected_shed = sum(scen.π[i] * lex_shed[i, d]
                        for i in 1:scen.K, d in 1:cd.nD)
    eps_curtail = Float64(get(P, "eps_curtail", 1e-3))
    eps_gen = Float64(get(P, "eps_gen_tiebreak", 1e-6))
    primary_scale = Float64(get(P, "voll", maximum(cd.cshed)))
    isfinite(primary_scale) && primary_scale > 0.0 ||
        error("lexicographic operating-cost scale must be finite and positive")
    generation_tiebreak = (eps_gen / primary_scale) *
                          sum(cd.cgen[j] * v.g[j] for j in 1:cd.nG)
    return expected_shed + eps_curtail * sum(v.cW) + generation_tiebreak
end

"No-smoothing cost-reference pilot used exclusively to freeze KDE bandwidths.
It has the same X_phys and six reference HMCR constraints as the final model,
but uses h=0 in this pre-solve calibration stage."
function solve_no_smoothing_pilot(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                                  P::Dict; responsive_loads = trues(cd.nD))
    p = Int(P["p"]); gamma_c = Float64(P["gamma_c"]); gamma_m = Float64(P["gamma_m"])
    nodes, weights = _normal_nodes(Int(P["quad_order"]))
    m = _model(P); tbuild = time(); v = _physics_and_losses!(m, cd, snap, scen;
                                                               responsive_loads = responsive_loads)
    for mi in 1:6
        @constraint(m, _add_hmcr!(m, v.safety[mi], scen.π, gamma_m, 0.0, p, nodes, weights) <= 0)
    end
    objective = _add_hmcr!(m, v.cost, scen.π, gamma_c, 0.0, p, nodes, weights)
    @objective(m, Min, objective)
    return _result(m, v, nothing, nothing, objective, time() - tbuild)
end

function _raw_losses(cd::CaseData, snap::Snapshot, delta, omega, eL, x)
    K = length(omega); nG, nD, nE = cd.nG, cd.nD, cd.nE
    length(x.beta) == nG || error("pilot beta dimension mismatch")
    length(x.alphaS) == nD || error("pilot alphaS dimension mismatch")
    up = fill(-Inf, K); down = fill(-Inf, K); pos = fill(-Inf, K); neg = fill(-Inf, K)
    low = fill(-Inf, K); high = fill(-Inf, K)
    for j in 1:nG
        cd.agc_mask[j] || continue
        scale = max(cd.pmax[j], 1.0)
        up = max.(up, (-omega .* x.beta[j] .- x.rU[j]) ./ scale)
        down = max.(down, (omega .* x.beta[j] .- x.rD[j]) ./ scale)
    end
    f0 = cd.MG * x.g + cd.MW * (snap.wf - x.cW) - cd.MD * (snap.lf - x.snom)
    response = cd.MG * x.beta + cd.MD * x.alphaS
    ptdf = delta * cd.PTDF'
    for ell in 1:nE
        isfinite(cd.F_max[ell]) || continue
        scale = max(cd.F_max[ell], 1.0)
        flow = ptdf[:, ell] .+ f0[ell] .- omega .* response[ell]
        pos = max.(pos, (flow .- cd.F_max[ell]) ./ scale)
        neg = max.(neg, (-flow .- cd.F_max[ell]) ./ scale)
    end
    for d in 1:nD
        scale = max(snap.lf[d], 1.0)
        shed = x.snom[d] .- omega .* x.alphaS[d]
        low = max.(low, -shed ./ scale)
        high = max.(high, (shed .- (snap.lf[d] .+ eL[:, d])) ./ scale)
    end
    return [sum(x.snom[d] - omega[i] * x.alphaS[d] for d in 1:nD) for i in 1:K],
           [up, down, pos, neg, low, high]
end

function _silverman(values, multiplier, floor)
    n = length(values)
    n >= 2 || error("at least two raw training samples are required for bandwidth estimation")
    sigma = std(values; corrected = true)
    raw = isfinite(sigma) && sigma > 0 ? 1.06 * multiplier * sigma * n^(-1 / 5) : 0.0
    return max(raw, floor)
end

"Estimate seven fixed bandwidths from raw training samples and the frozen
no-smoothing pilot.  This is parameter precomputation, not a final-model term."
function estimate_bandwidths(cd::CaseData, snap::Snapshot, delta, omega, eL, pilot_x;
                             multiplier::Float64 = 1.0,
                             cost_floor::Float64 = 1e-6,
                             safety_floor::Float64 = 1e-6)
    multiplier > 0 || error("bandwidth multiplier must be positive")
    cost_floor > 0 || error("cost bandwidth floor must be strictly positive")
    safety_floor > 0 || error("safety bandwidth floor must be strictly positive")
    costs, safeties = _raw_losses(cd, snap, delta, omega, eL, pilot_x)
    return FinalBandwidth(_silverman(costs, multiplier, cost_floor),
                          [_silverman(safeties[m], multiplier, safety_floor) for m in 1:6],
                          length(costs), multiplier, cost_floor, safety_floor)
end

"Validate the two distinct target-relaxation parameters required by the paper."
function _validate_target_relaxations(rho_tau_c::Float64, rho_tau_s::Float64)
    0 <= rho_tau_c <= 1 || error("rho_tau_c must lie in [0,1]")
    0 <= rho_tau_s <= 1 || error("rho_tau_s must lie in [0,1]")
    return nothing
end

"Solve the one cost and six rotated reference problems exactly as specified in
the final paper.  No safety objective is evaluated at a cost anchor."
function calibrate_reference_targets(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                                     P::Dict, bandwidth::FinalBandwidth;
                                     responsive_loads = trues(cd.nD),
                                     rho_tau_c::Float64, rho_tau_s::Float64)
    _validate_target_relaxations(rho_tau_c, rho_tau_s)
    Int(P["p"]) in (1, 2) || error("only p=1 or p=2 are implemented")
    all(>(0.0), bandwidth.safety_h) && bandwidth.cost_h > 0 ||
        error("the final model requires strictly positive fixed bandwidths")
    p = Int(P["p"]); gamma_c = Float64(P["gamma_c"]); gamma_m = Float64(P["gamma_m"])
    nodes, weights = _normal_nodes(Int(P["quad_order"]))
    statuses = String[]
    primal_residuals = Float64[]
    dual_residuals = Float64[]
    gap_abs_values = Float64[]
    gap_rel_values = Float64[]

    function reference(target)
        m = _model(P); v = _physics_and_losses!(m, cd, snap, scen; responsive_loads = responsive_loads)
        srisk = [_add_hmcr!(m, v.safety[mi], scen.π, gamma_m, bandwidth.safety_h[mi], p, nodes, weights)
                 for mi in 1:6]
        crisk = _add_hmcr!(m, v.cost, scen.π, gamma_c, bandwidth.cost_h, p, nodes, weights)
        if target === :cost
            for mi in 1:6; @constraint(m, srisk[mi] <= 0); end
            @objective(m, Min, crisk)
        else
            mi = Int(target)
            for other in 1:6
                other == mi || @constraint(m, srisk[other] <= 0)
            end
            @objective(m, Min, srisk[mi])
        end
        optimize!(m)
        status = string(termination_status(m)); push!(statuses, status)
        quality = _quality(m)
        push!(primal_residuals, quality.primal)
        push!(dual_residuals, quality.dual)
        push!(gap_abs_values, quality.gap_abs)
        push!(gap_rel_values, quality.gap_rel)
        status in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(m) ||
            error("reference target $target did not solve: $status")
        return value(target === :cost ? crisk : srisk[Int(target)])
    end

    Z0c = reference(:cost)
    # Each reference call owns a large transient JuMP/Clarabel model. The
    # scalar target and quality fields have now been copied, so reclaiming the
    # backend before the next of the six independent references is purely
    # runtime memory hygiene; it cannot alter a target or optimization model.
    GC.gc()
    Z0c > 0 || error("RBTS batch rejects the final-model reference rule because Z0c=$Z0c is not strictly positive")
    Z0m = Float64[]
    for mi in 1:6
        push!(Z0m, reference(mi))
        GC.gc()
    end
    all(z -> z <= 0.0, Z0m) ||
        error("safety-reference nonpositivity certification failed: Z0m=$Z0m")
    # The two-target-relaxation model uses independent cost and safety
    # parameters.  The reference nonpositivity check above makes this exact
    # even when a safety reference target is zero.
    tau_m = [(1 - rho_tau_s) * z for z in Z0m]
    return FinalCalibration(Z0c, Z0m, (1 + rho_tau_c) * Z0c, tau_m,
                            rho_tau_c, rho_tau_s, bandwidth, statuses,
                            primal_residuals, dual_residuals,
                            gap_abs_values, gap_rel_values)
end

"""The exact cost-reference problem used to construct Z0c: minimize the cost
KDE-HMCR subject to all six reference-distribution safety HMCR constraints."""
function solve_cost_reference(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                              P::Dict, bandwidth::FinalBandwidth;
                              responsive_loads = trues(cd.nD))
    p = Int(P["p"]); gamma_c = Float64(P["gamma_c"]); gamma_m = Float64(P["gamma_m"])
    nodes, weights = _normal_nodes(Int(P["quad_order"]))
    m = _model(P); tbuild = time(); v = _physics_and_losses!(m, cd, snap, scen;
                                                               responsive_loads = responsive_loads)
    for mi in 1:6
        risk = _add_hmcr!(m, v.safety[mi], scen.π, gamma_m, bandwidth.safety_h[mi], p, nodes, weights)
        @constraint(m, risk <= 0)
    end
    objective = _add_hmcr!(m, v.cost, scen.π, gamma_c, bandwidth.cost_h, p, nodes, weights)
    @objective(m, Min, objective)
    return _result(m, v, nothing, nothing, objective, time() - tbuild)
end
function retarget(cal::FinalCalibration; rho_tau_c::Float64, rho_tau_s::Float64)
    _validate_target_relaxations(rho_tau_c, rho_tau_s)
    cal.Z0c > 0.0 || error("cost reference must remain strictly positive")
    all(z -> z <= 0.0, cal.Z0m) ||
        error("cannot retarget a calibration with a positive safety reference")
    return FinalCalibration(cal.Z0c, copy(cal.Z0m), (1 + rho_tau_c) * cal.Z0c,
                            [(1 - rho_tau_s) * z for z in cal.Z0m],
                            rho_tau_c, rho_tau_s, cal.bandwidth,
                            copy(cal.reference_status),
                            copy(cal.reference_primal_residual),
                            copy(cal.reference_dual_residual),
                            copy(cal.reference_gap_abs),
                            copy(cal.reference_gap_rel))
end

function solve_final_model(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                           P::Dict, cal::FinalCalibration;
                           omega::AbstractVector = ones(6),
                           responsive_loads = trues(cd.nD))
    length(omega) == 6 && all(>(0.0), omega) ||
        error("the final-model safety preference vector must have six strictly positive entries")
    p = Int(P["p"]); gamma_c = Float64(P["gamma_c"]); gamma_m = Float64(P["gamma_m"])
    eps_y = Float64(P["eps_y"]); nodes, weights = _normal_nodes(Int(P["quad_order"]))
    m = _model(P); tbuild = time(); v = _physics_and_losses!(m, cd, snap, scen;
                                                               responsive_loads = responsive_loads)
    @variable(m, kappa_c >= 0)
    @variable(m, kappa_m[1:6] >= 0)
    _add_rs_constraint!(m, v.cost, scen.π, cal.tau_c, kappa_c, gamma_c,
                        cal.bandwidth.cost_h, p, nodes, weights, eps_y)
    for mi in 1:6
        _add_rs_constraint!(m, v.safety[mi], scen.π, cal.tau_m[mi], kappa_m[mi],
                            gamma_m, cal.bandwidth.safety_h[mi], p, nodes, weights, eps_y)
    end
    objective = kappa_c + sum(Float64(omega[mi]) * kappa_m[mi] for mi in 1:6)
    @objective(m, Min, objective)
    lex_enabled = Bool(get(P, "lexicographic_secondary_shed", true))
    secondary_builder = lex_enabled ?
        model -> _lexicographic_shed_objective!(model, v, cd, scen, P) : nothing
    return _result(m, v, kappa_c, kappa_m, objective, time() - tbuild;
        lex_secondary_builder = secondary_builder,
        # The primary lock uses the same absolute numerical scale as the
        # predeclared formal acceptance gate; a tighter 1e-7 lock caused the
        # second Clarabel solve to terminate before producing an auditable
        # tie-break decision on real outage states.
        lex_abs_tol = Float64(get(P, "lexicographic_abs_tol", 5e-5)),
        lex_rel_tol = Float64(get(P, "lexicographic_rel_tol", 0.0)),
        lex_gap_multiplier = Float64(get(P, "lexicographic_gap_multiplier", 10.0)),
        lex_activation_tol = Float64(get(P, "lexicographic_activation_tol", 1e-4)))
end

"Evaluate the seven raw losses at an already-fixed decision, using the same
internal-sign convention as the optimization model."
function evaluate_losses(cd::CaseData, snap::Snapshot, scen::ScenarioData, x)
    return _raw_losses(cd, snap, scen.δ, scen.Ω, scen.eL, x)
end

"Independent numerical evaluator for a p=1/p=2 KDE-HMCR value."
function evaluate_hmcr(losses, pi, gamma::Float64, h::Float64, p::Int, Q::Int)
    nodes, weights = _normal_nodes(Q)
    shifted = [losses[i] - h * nodes[q] for i in eachindex(losses), q in eachindex(nodes)]
    span = max(maximum(shifted) - minimum(shifted), 1.0)
    lo, hi = minimum(shifted) - 10span, maximum(shifted) + span
    function value_at(alpha)
        moments = [sum(pi[i] * weights[q] * max(shifted[i, q] - alpha, 0.0)^p
                       for i in eachindex(losses), q in eachindex(nodes))]
        return alpha + moments[1]^(1 / p) / (1 - gamma)
    end
    # A convex one-dimensional bisection on a finite bracket is adequate for
    # the independent audit evaluator; it is not used to construct a solution.
    for _ in 1:160
        a = lo + (hi - lo) / 3; b = hi - (hi - lo) / 3
        if value_at(a) <= value_at(b); hi = b else; lo = a end
    end
    return value_at((lo + hi) / 2)
end

end # module
