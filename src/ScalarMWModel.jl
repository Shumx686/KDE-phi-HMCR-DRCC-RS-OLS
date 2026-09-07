module ScalarMWModel

using JuMP
using Clarabel
using LinearAlgebra
using Statistics

using ..CaseInterface: CaseData, ScenarioData, Snapshot,
                       sample_safe_curtailment_cap

export ScalarMWBandwidthProfile,
       raw_loss_samples,
       solve_bandwidth_pilot,
       build_raw_profile,
       calibrate_scalar_targets,
       solve_scalar_mw,
       scalar_event_count,
       scalar_event_labels

"""
Fixed KDE bandwidths for the canonical raw-MW model.

`cost_h` is the bandwidth of the unpenalized total-shedding loss and
`event_h[r]` is the bandwidth of scalar safety event `event_labels[r]`.
Every entry is a strictly positive, precomputed `Float64`; bandwidths are
never JuMP variables in this module.
"""
struct ScalarMWBandwidthProfile
    cost_h::Float64
    event_h::Vector{Float64}
    event_labels::Vector{String}
    raw_n::Int
    multiplier::Float64
    h_floor_mw::Float64
    binding::NamedTuple
end

const _BETA = Symbol(Char(0x03b2))
const _ALPHA_S = Symbol(string(Char(0x03b1), "S"))

_beta(x) = getproperty(x, _BETA)
_alpha_s(x) = getproperty(x, _ALPHA_S)

function _finite_lines(cd::CaseData)
    length(cd.F_max) == cd.nE || throw(DimensionMismatch(
        "F_max has length $(length(cd.F_max)); expected nE=$(cd.nE)"))
    all(isfinite, cd.F_max) || error(
        "the canonical scalar-event set contains every line; every F_max must be finite")
    all(>=(0.0), cd.F_max) || error("every line limit F_max must be nonnegative")
    return collect(1:cd.nE)
end

function scalar_event_count(cd::CaseData)
    return 2 * count(cd.agc_mask) + 2 * length(_finite_lines(cd)) + 2 * cd.nD
end

function scalar_event_labels(cd::CaseData)
    labels = String[]
    for j in findall(cd.agc_mask)
        push!(labels, "reserve_up_G$(j)")
    end
    for j in findall(cd.agc_mask)
        push!(labels, "reserve_down_G$(j)")
    end
    for ell in _finite_lines(cd)
        push!(labels, "line_pos_L$(ell)")
    end
    for ell in _finite_lines(cd)
        push!(labels, "line_neg_L$(ell)")
    end
    for d in 1:cd.nD
        push!(labels, "shed_low_D$(d)")
    end
    for d in 1:cd.nD
        push!(labels, "shed_high_D$(d)")
    end
    return labels
end

function _shed_mask(mask, nD::Int)
    mask === nothing && return trues(nD)
    length(mask) == nD || throw(DimensionMismatch(
        "shed_response_mask has length $(length(mask)); expected $nD"))
    return Bool.(collect(mask))
end

function _validate_probabilities(pi)
    weights = Float64.(collect(pi))
    isempty(weights) && error("reference weights must be nonempty")
    all(isfinite, weights) || error("reference weights contain nonfinite values")
    all(>(0.0), weights) || error("the canonical model requires strictly positive reference weights")
    isapprox(sum(weights), 1.0; rtol = 1e-10, atol = 1e-12) ||
        error("reference weights must sum to one; received $(sum(weights))")
    return weights
end

function _validate_pilot(cd::CaseData, pilot_x, shed_mask)
    fields = ((:g, cd.nG), (:rU, cd.nG), (:rD, cd.nG),
              (:snom, cd.nD), (:cW, cd.nW), (_BETA, cd.nG),
              (_ALPHA_S, cd.nD))
    for (name, expected) in fields
        hasproperty(pilot_x, name) || error("pilot decision is missing field $name")
        values = getproperty(pilot_x, name)
        length(values) == expected || throw(DimensionMismatch(
            "pilot field $name has length $(length(values)); expected $expected"))
        all(isfinite, values) || error("pilot field $name contains nonfinite values")
    end
    alpha_s = _alpha_s(pilot_x)
    for d in eachindex(shed_mask)
        !shed_mask[d] && abs(alpha_s[d]) > 1e-9 &&
            error("pilot alphaS[$d] is nonzero outside the declared responsive-load set")
    end
    return nothing
end

function _reconstruct_delta(cd::CaseData, eL::AbstractMatrix,
                            eW::AbstractMatrix)
    n = size(eL, 1)
    size(eL) == (n, cd.nD) || throw(DimensionMismatch(
        "load-error matrix must have size ($n,$(cd.nD))"))
    size(eW) == (n, cd.nW) || throw(DimensionMismatch(
        "wind-error matrix must have size ($n,$(cd.nW))"))
    length(cd.load_buses) == cd.nD || throw(DimensionMismatch(
        "load_buses length does not match nD"))
    length(cd.wind_buses) == cd.nW || throw(DimensionMismatch(
        "wind_buses length does not match nW"))
    delta = zeros(Float64, n, cd.nbus)
    for w in 1:cd.nW
        bus = cd.wind_buses[w]
        haskey(cd.busidx, bus) || error("wind bus $bus is absent from busidx")
        delta[:, cd.busidx[bus]] .+= Float64.(view(eW, :, w))
    end
    for d in 1:cd.nD
        bus = cd.load_buses[d]
        haskey(cd.busidx, bus) || error("load bus $bus is absent from busidx")
        delta[:, cd.busidx[bus]] .-= Float64.(view(eL, :, d))
    end
    return delta
end

function _validate_paired_errors(cd::CaseData, delta::AbstractMatrix,
                                 eL::AbstractMatrix, eW::AbstractMatrix,
                                 label::AbstractString)
    size(delta, 1) == size(eL, 1) || throw(DimensionMismatch(
        "$label delta/eL row counts differ"))
    all(isfinite, eL) || error("$label eL contains nonfinite values")
    all(isfinite, eW) || error("$label eW contains nonfinite values")
    expected = _reconstruct_delta(cd, eL, eW)
    size(delta) == size(expected) || throw(DimensionMismatch(
        "$label delta must have size $(size(expected))"))
    all(isfinite, delta) || error("$label delta contains nonfinite values")
    consistent = all(isapprox.(Float64.(delta), expected;
                               rtol = 1e-10, atol = 1e-8))
    if !consistent
        mismatch = maximum(abs.(Float64.(delta) .- expected))
        error("$label violates paired-error consistency: delta must equal " *
              "mapped wind error minus mapped load error; max mismatch=$mismatch")
    end
    return expected
end

function _case_binding(cd::CaseData)
    return (nbus = cd.nbus, nG = cd.nG, nD = cd.nD,
            nW = cd.nW, nE = cd.nE,
            pmax = copy(cd.pmax), pmin = copy(cd.pmin),
            agc_mask = copy(cd.agc_mask), F_max = copy(cd.F_max),
            PTDF = copy(cd.PTDF), MG = copy(cd.MG),
            MD = copy(cd.MD), MW = copy(cd.MW))
end

_snapshot_binding(snap::Snapshot) =
    (lf = copy(snap.lf), wf = copy(snap.wf))

function _numeric_x(x)
    return (g = Float64.(x.g), rU = Float64.(x.rU),
            rD = Float64.(x.rD), snom = Float64.(x.snom),
            cW = Float64.(x.cW), beta = Float64.(_beta(x)),
            alphaS = Float64.(_alpha_s(x)))
end

function _raw_summary(delta::AbstractMatrix)
    n = size(delta, 1)
    n > 0 || error("raw sample must be nonempty")
    values = Float64.(delta)
    mean_value = vec(sum(values, dims = 1)) ./ n
    centered = values .- mean_value'
    return (n = n, mean = mean_value,
            covariance = centered' * centered / n,
            lower = vec(minimum(values, dims = 1)),
            upper = vec(maximum(values, dims = 1)))
end

function _profile_binding(cd::CaseData, snap::Snapshot,
                          delta_raw, omega_raw, eL_raw, eW_raw,
                          pilot_x, response_mask, pilot_feasibility_tol)
    return (case_data = _case_binding(cd),
            snapshot = _snapshot_binding(snap),
            raw = (delta = Float64.(delta_raw),
                   omega = Float64.(omega_raw),
                   eL = Float64.(eL_raw), eW = Float64.(eW_raw),
                   summary = _raw_summary(delta_raw)),
            pilot_x = _numeric_x(pilot_x),
            shed_response_mask = Bool.(collect(response_mask)),
            pilot_feasibility_tol = Float64(pilot_feasibility_tol))
end

function _sample_sigma(values::AbstractVector)
    length(values) <= 1 && return 0.0
    sigma = std(values; corrected = true)
    return isfinite(sigma) && sigma > eps(Float64) ? Float64(sigma) : 0.0
end

function _silverman(values::AbstractVector, multiplier::Float64,
                    h_floor_mw::Float64)
    sigma = _sample_sigma(values)
    sigma == 0.0 && return h_floor_mw
    return max(multiplier * 1.06 * sigma * length(values)^(-1 / 5),
               h_floor_mw)
end

"""
    raw_loss_samples(cd, snap, delta_raw, omega_raw, eL_raw, eW_raw, x;
                     shed_response_mask=nothing)

Evaluate the canonical raw-MW loss samples at a fixed physical decision
`x`.  The returned scalar events use the six contiguous blocks

`GU(all), GD(all), L+(all), L-(all), S-(all), S+(all)`,

exactly as reported by `scalar_event_labels`.  The cost samples are the
unpenalized affine total shedding values.  This numerical oracle performs no
KDE smoothing, positive-part clipping, normalization, or category maximum;
it is intended for bandwidth construction and formula-level validation.

The codebase stores net-injection error, the negative of the paper's
net-load error.  Thus the code-space sample shedding is
`snom - omega_raw * alphaS`.
"""
function raw_loss_samples(cd::CaseData, snap::Snapshot,
                          delta_raw::AbstractMatrix,
                          omega_raw::AbstractVector,
                          eL_raw::AbstractMatrix,
                          eW_raw::AbstractMatrix, x;
                          shed_response_mask = nothing)
    n = length(omega_raw)
    n > 0 || error("raw training sample must be nonempty")
    size(delta_raw) == (n, cd.nbus) || throw(DimensionMismatch(
        "delta_raw must have size ($n,$(cd.nbus))"))
    _validate_paired_errors(cd, delta_raw, eL_raw, eW_raw, "raw sample")
    all(isfinite, omega_raw) || error("omega_raw contains nonfinite values")
    omega_check = vec(sum(delta_raw, dims = 2))
    all(isapprox.(Float64.(omega_raw), omega_check;
                  rtol = 1e-10, atol = 1e-8)) ||
        error("omega_raw must be the row sum of the same delta_raw sample")

    response_mask = _shed_mask(shed_response_mask, cd.nD)
    _validate_pilot(cd, x, response_mask)
    beta = Float64.(_beta(x))
    alpha_s = Float64.(_alpha_s(x))
    omega = Float64.(omega_raw)
    delta = Float64.(delta_raw)
    eL = Float64.(eL_raw)
    eW = Float64.(eW_raw)

    reserve_up = [[-omega[i] * beta[j] - Float64(x.rU[j]) for i in 1:n]
                  for j in findall(cd.agc_mask)]
    reserve_down = [[omega[i] * beta[j] - Float64(x.rD[j]) for i in 1:n]
                    for j in findall(cd.agc_mask)]

    finite_lines = _finite_lines(cd)
    f0 = cd.MG * Float64.(x.g) +
         cd.MW * (snap.wf - Float64.(x.cW)) -
         cd.MD * (snap.lf - Float64.(x.snom))
    response = cd.MG * beta + cd.MD * alpha_s
    ptdf_error = delta * cd.PTDF[finite_lines, :]'
    flow = [[f0[ell] + ptdf_error[i, lcol] - omega[i] * response[ell]
             for i in 1:n]
            for (lcol, ell) in enumerate(finite_lines)]
    line_pos = [flow[lcol] .- cd.F_max[ell]
                for (lcol, ell) in enumerate(finite_lines)]
    line_neg = [-flow[lcol] .- cd.F_max[ell]
                for (lcol, ell) in enumerate(finite_lines)]

    sample_shed = [Float64(x.snom[d]) - omega[i] * alpha_s[d]
                   for i in 1:n, d in 1:cd.nD]
    shed_low = [-collect(view(sample_shed, :, d)) for d in 1:cd.nD]
    shed_high = [[sample_shed[i, d] - (snap.lf[d] + eL[i, d])
                  for i in 1:n] for d in 1:cd.nD]

    events = Vector{Vector{Float64}}()
    append!(events, reserve_up)
    append!(events, reserve_down)
    append!(events, line_pos)
    append!(events, line_neg)
    append!(events, shed_low)
    append!(events, shed_high)
    labels = scalar_event_labels(cd)
    length(events) == length(labels) ||
        error("internal scalar-event ordering mismatch")
    cost = [sum(sample_shed[i, d] for d in 1:cd.nD) for i in 1:n]
    return (cost = cost, events = events, event_labels = labels,
            sample_shed = sample_shed, nominal_flow = f0,
            sample_flow = flow, load_error = eL,
            wind_error = eW, omega = omega)
end

function raw_loss_samples(cd::CaseData, snap::Snapshot,
                          delta_raw::AbstractMatrix,
                          omega_raw::AbstractVector,
                          eL_raw::AbstractMatrix, x;
                          shed_response_mask = nothing)
    error("raw_loss_samples requires paired eW_raw so delta_raw can be " *
          "verified as mapped wind error minus mapped load error")
end

"""
    build_raw_profile(cd, snap, delta_raw, omega_raw, eL_raw, eW_raw, pilot_x;
                      multiplier=1.0, h_floor_mw=1e-6,
                      shed_response_mask=nothing,
                      pilot_feasibility_tol=1e-5)

Build fixed positive Silverman bandwidths from the complete raw training
sample and one decision fixed before reference calibration.  All losses are
in MW.  No price coefficient, positive-part clipping, device normalization,
or category maximum is used.

The codebase stores net-injection error, the negative of the paper's
net-load error.  Consequently sample shedding is
`snom - omega_raw * alphaS`, which is algebraically identical to the paper's
`snom + Omega_netload * alphaS`.
"""
function build_raw_profile(cd::CaseData, snap::Snapshot,
                           delta_raw::AbstractMatrix,
                           omega_raw::AbstractVector,
                           eL_raw::AbstractMatrix,
                           eW_raw::AbstractMatrix, pilot_x;
                           multiplier::Real = 1.0,
                           h_floor_mw::Real = 1e-6,
                           shed_response_mask = nothing,
                           pilot_feasibility_tol::Real = 1e-5)
    b = Float64(multiplier)
    floor_mw = Float64(h_floor_mw)
    isfinite(b) && b > 0 || error("bandwidth multiplier must be positive and finite")
    isfinite(floor_mw) && floor_mw > 0 ||
        error("h_floor_mw must be strictly positive and finite")
    pilot_tol = Float64(pilot_feasibility_tol)
    isfinite(pilot_tol) && pilot_tol >= 0.0 ||
        error("pilot_feasibility_tol must be nonnegative and finite")
    response_mask = _shed_mask(shed_response_mask, cd.nD)
    _validate_pilot(cd, pilot_x, response_mask)
    pilot_numeric = _numeric_x(pilot_x)
    pilot_physical = _physical_residuals(cd, snap, pilot_numeric,
                                         response_mask)
    pilot_physical.max_violation <= pilot_tol || error(
        "bandwidth pilot is outside Xphys: max violation=" *
        "$(pilot_physical.max_violation) exceeds pilot_feasibility_tol=$pilot_tol")

    losses = raw_loss_samples(cd, snap, delta_raw, omega_raw, eL_raw,
                              eW_raw, pilot_x;
                              shed_response_mask = response_mask)
    event_h = [_silverman(values, b, floor_mw) for values in losses.events]
    cost_h = _silverman(losses.cost, b, floor_mw)
    all(h -> isfinite(h) && h > 0.0, event_h) ||
        error("every scalar-event bandwidth must be strictly positive")
    isfinite(cost_h) && cost_h > 0.0 ||
        error("the cost bandwidth must be strictly positive")
    binding = _profile_binding(cd, snap, delta_raw, omega_raw,
                               eL_raw, eW_raw, pilot_x, response_mask,
                               pilot_tol)
    return ScalarMWBandwidthProfile(cost_h, event_h, losses.event_labels,
                                    length(omega_raw), b, floor_mw, binding)
end

function build_raw_profile(cd::CaseData, snap::Snapshot,
                           delta_raw::AbstractMatrix,
                           omega_raw::AbstractVector,
                           eL_raw::AbstractMatrix, pilot_x;
                           multiplier::Real = 1.0,
                           h_floor_mw::Real = 1e-6,
                           shed_response_mask = nothing,
                           pilot_feasibility_tol::Real = 1e-5)
    error("build_raw_profile requires paired eW_raw so delta_raw can be " *
          "verified before any Silverman bandwidth is frozen")
end

function _validate_profile(profile::ScalarMWBandwidthProfile, cd::CaseData,
                           snap::Snapshot, response_mask;
                           scen = nothing)
    labels = scalar_event_labels(cd)
    profile.event_labels == labels ||
        error("bandwidth-profile event labels do not match the canonical scalar-event order")
    length(profile.event_h) == scalar_event_count(cd) ||
        error("bandwidth profile has $(length(profile.event_h)) events; expected $(scalar_event_count(cd))")
    profile.raw_n > 0 || error("bandwidth profile raw_n must be positive")
    isfinite(profile.cost_h) && profile.cost_h > 0.0 ||
        error("cost_h must be strictly positive and finite")
    all(h -> isfinite(h) && h > 0.0, profile.event_h) ||
        error("all event_h values must be strictly positive and finite")
    binding = profile.binding
    binding.case_data == _case_binding(cd) || error(
        "bandwidth profile was built for different physical case data or line scaling")
    binding.snapshot == _snapshot_binding(snap) || error(
        "bandwidth profile was built for a different forecast snapshot (lf/wf)")
    binding.shed_response_mask == Bool.(collect(response_mask)) || error(
        "bandwidth profile was built for a different shed_response_mask")
    raw = binding.raw
    size(raw.delta, 1) == profile.raw_n || error(
        "bandwidth profile raw binding does not match raw_n")
    _validate_paired_errors(cd, raw.delta, raw.eL, raw.eW,
                            "bound raw sample")
    all(isapprox.(raw.omega, vec(sum(raw.delta, dims = 2));
                  rtol = 1e-10, atol = 1e-8)) || error(
        "bound raw omega is inconsistent with bound raw delta")
    summary = _raw_summary(raw.delta)
    bound_summary = raw.summary
    summary.n == bound_summary.n &&
        all(isapprox.(summary.mean, bound_summary.mean;
                      rtol = 1e-12, atol = 1e-12)) &&
        all(isapprox.(summary.covariance, bound_summary.covariance;
                      rtol = 1e-12, atol = 1e-12)) &&
        summary.lower == bound_summary.lower &&
        summary.upper == bound_summary.upper || error(
            "bandwidth profile raw binding was mutated after construction")
    pilot_physical = _physical_residuals(cd, snap, binding.pilot_x,
                                         response_mask)
    pilot_physical.max_violation <= binding.pilot_feasibility_tol || error(
        "bound bandwidth pilot is no longer feasible in Xphys")
    if scen !== nothing
        Int(scen.raw_n) == profile.raw_n || error(
            "scenario compression and bandwidth profile use different raw_n")
        length(scen.raw_mean) == cd.nbus || throw(DimensionMismatch(
            "scenario raw_mean dimension does not match nbus"))
        size(scen.raw_cov) == (cd.nbus, cd.nbus) || throw(DimensionMismatch(
            "scenario raw_cov dimension does not match nbus"))
        length(scen.raw_lower) == cd.nbus || throw(DimensionMismatch(
            "scenario raw_lower dimension does not match nbus"))
        length(scen.raw_upper) == cd.nbus || throw(DimensionMismatch(
            "scenario raw_upper dimension does not match nbus"))
        all(isapprox.(scen.raw_mean, summary.mean;
                      rtol = 1e-10, atol = 1e-10)) &&
            all(isapprox.(scen.raw_cov, summary.covariance;
                          rtol = 1e-9, atol = 1e-8)) &&
            all(isapprox.(scen.raw_lower, summary.lower;
                          rtol = 1e-10, atol = 1e-10)) &&
            all(isapprox.(scen.raw_upper, summary.upper;
                          rtol = 1e-10, atol = 1e-10)) || error(
                "scenario compression raw summaries do not match the bandwidth raw sample")
    end
    return nothing
end

function _new_model(P::AbstractDict)
    model = Model(Clarabel.Optimizer)
    set_silent(model)
    set_optimizer_attribute(model, "max_iter", Int(get(P, "solver_max_iter", 2000)))
    set_optimizer_attribute(model, "equilibrate_enable", true)
    set_optimizer_attribute(model, "tol_gap_abs", 1e-8)
    set_optimizer_attribute(model, "tol_gap_rel", 1e-8)
    set_optimizer_attribute(model, "tol_feas", 1e-8)
    set_optimizer_attribute(model, "reduced_tol_gap_abs", 5e-5)
    set_optimizer_attribute(model, "reduced_tol_gap_rel", 1e-5)
    set_optimizer_attribute(model, "reduced_tol_feas", 1e-5)
    if haskey(P, "solver_direct_solve_method")
        set_optimizer_attribute(model, "direct_solve_method",
                                Symbol(P["solver_direct_solve_method"]))
    end
    if haskey(P, "solver_equilibrate_max_iter")
        set_optimizer_attribute(model, "equilibrate_max_iter",
                                Int(P["solver_equilibrate_max_iter"]))
    end
    if haskey(P, "solver_iterative_refinement_max_iter")
        set_optimizer_attribute(model, "iterative_refinement_max_iter",
                                Int(P["solver_iterative_refinement_max_iter"]))
    end
    if haskey(P, "solver_dynamic_regularization_delta")
        set_optimizer_attribute(model, "dynamic_regularization_delta",
                                Float64(P["solver_dynamic_regularization_delta"]))
    end
    if haskey(P, "solver_min_terminate_step_length")
        set_optimizer_attribute(model, "min_terminate_step_length",
                                Float64(P["solver_min_terminate_step_length"]))
    end
    return model
end

function _quality(model)
    try
        info = unsafe_backend(model).solver_info
        return (primal_residual = Float64(info.res_primal),
                dual_residual = Float64(info.res_dual),
                gap_abs = Float64(info.gap_abs),
                gap_rel = Float64(info.gap_rel))
    catch
        return (primal_residual = NaN, dual_residual = NaN,
                gap_abs = NaN, gap_rel = NaN)
    end
end

function _acceptance_tolerances(P::AbstractDict)
    residual = Float64(get(P, "solver_accept_scaled_residual", 1e-5))
    gap_abs = Float64(get(P, "solver_accept_gap_abs",
                          get(P, "solver_accept_gap", 5e-5)))
    gap_rel = Float64(get(P, "solver_accept_gap_rel",
                          get(P, "solver_accept_gap", 1e-5)))
    all(isfinite, (residual, gap_abs, gap_rel)) ||
        error("solver acceptance tolerances must be finite")
    residual >= 0.0 || error("solver_accept_scaled_residual must be nonnegative")
    gap_abs >= 0.0 || error("solver_accept_gap_abs must be nonnegative")
    gap_rel >= 0.0 || error("solver_accept_gap_rel must be nonnegative")
    return (residual = residual, gap_abs = gap_abs, gap_rel = gap_rel)
end

function _quality_accepted(status, quality, P::AbstractDict)
    string(status) in ("OPTIMAL", "ALMOST_OPTIMAL") || return false
    values = (quality.primal_residual, quality.dual_residual,
              quality.gap_abs, quality.gap_rel)
    all(isfinite, values) || return false
    tol = _acceptance_tolerances(P)
    return max(abs(quality.primal_residual), abs(quality.dual_residual)) <=
           tol.residual &&
           (abs(quality.gap_abs) <= tol.gap_abs ||
            abs(quality.gap_rel) <= tol.gap_rel)
end

function _require_quality_accepted(status, quality, P::AbstractDict,
                                   label::AbstractString)
    _quality_accepted(status, quality, P) && return nothing
    tol = _acceptance_tolerances(P)
    error("$label failed strict solver-quality acceptance: status=$status, " *
          "primal=$(quality.primal_residual), dual=$(quality.dual_residual), " *
          "gap_abs=$(quality.gap_abs), gap_rel=$(quality.gap_rel); " *
          "limits are residual<=$(tol.residual) and " *
          "(gap_abs<=$(tol.gap_abs) or gap_rel<=$(tol.gap_rel))")
end

function _normal_nodes(Q::Int, rule::AbstractString; weight_floor::Real = 0.0)
    Q >= 1 || error("kde_quad_nodes must be positive")
    Float64(weight_floor) == 0.0 || error(
        "the canonical Gauss-Hermite rule requires kde_quad_weight_floor=0")
    quad_rule = lowercase(strip(rule))
    if quad_rule in ("gauss_hermite", "gh", "normal")
        Q == 1 && return ([0.0], [1.0])
        jacobi = SymTridiagonal(zeros(Q), sqrt.(Float64.(1:Q-1)))
        eig = eigen(jacobi)
        nodes = collect(eig.values)
        weights = vec(eig.vectors[1, :] .^ 2)
        weights ./= sum(weights)
        return nodes, weights
    end
    error("the canonical model requires the Gauss-Hermite quadrature rule; received $rule")
end

function _risk_settings(P::AbstractDict, M::Int;
                        gamma_cost = nothing, gamma_event = nothing)
    p = Int(get(P, "p", 2))
    p in (1, 2) || error("ScalarMWModel currently supports HMCR p=1 or p=2; received p=$p")
    fallback_gamma = if haskey(P, "target_gamma")
        Float64(P["target_gamma"])
    elseif haskey(P, "gamma_m") && haskey(P, "eps_m")
        max(Float64(P["gamma_m"]), 1.0 - Float64(P["eps_m"]))
    else
        0.95
    end
    gc = gamma_cost === nothing ? fallback_gamma : Float64(gamma_cost)
    gr = if gamma_event === nothing
        fill(fallback_gamma, M)
    elseif gamma_event isa Real
        fill(Float64(gamma_event), M)
    else
        values = Float64.(collect(gamma_event))
        length(values) == M || throw(DimensionMismatch(
            "gamma_event has length $(length(values)); expected $M"))
        values
    end
    0.0 <= gc < 1.0 || error("gamma_cost must lie in [0,1)")
    all(g -> 0.0 <= g < 1.0, gr) || error("every event gamma must lie in [0,1)")
    Q = Int(get(P, "kde_quad_nodes", 45))
    rule = String(get(P, "kde_quad_rule", "gauss_hermite"))
    weight_floor = Float64(get(P, "kde_quad_weight_floor", 0.0))
    nodes, weights = _normal_nodes(Q, rule; weight_floor = weight_floor)
    return (p = p, gamma_cost = gc, gamma_event = gr,
            nodes = nodes, quad_weights = weights,
            quad_rule = "gauss_hermite", quad_nodes = Q,
            quad_weight_floor = weight_floor)
end

function _validate_scenario_samples(cd::CaseData, scen::ScenarioData)
    K = scen.K
    K > 0 || error("scenario support must be nonempty")
    length(scen.π) == K || throw(DimensionMismatch(
        "scenario probability length does not match K=$K"))
    size(scen.δ) == (K, cd.nbus) || throw(DimensionMismatch(
        "scenario delta must have size ($K,$(cd.nbus))"))
    size(scen.eL) == (K, cd.nD) || throw(DimensionMismatch(
        "scenario eL must have size ($K,$(cd.nD))"))
    size(scen.eW) == (K, cd.nW) || throw(DimensionMismatch(
        "scenario eW must have size ($K,$(cd.nW))"))
    length(scen.Ω) == K || throw(DimensionMismatch(
        "scenario Omega length does not match K=$K"))
    _validate_paired_errors(cd, scen.δ, scen.eL, scen.eW, "scenario sample")
    all(isfinite, scen.Ω) || error("scenario Omega contains nonfinite values")
    omega_check = vec(sum(scen.δ, dims = 2))
    all(isapprox.(scen.Ω, omega_check; rtol = 1e-10, atol = 1e-8)) ||
        error("scenario Omega must be the row sum of the same scenario delta")
    return nothing
end

function _calibration_binding(cd::CaseData, snap::Snapshot,
                              scen::ScenarioData, response_mask,
                              settings)
    return (
        case_data = _case_binding(cd),
        snapshot = _snapshot_binding(snap),
        scenario = (K = scen.K, π = copy(scen.π),
                    δ = copy(scen.δ), Ω = copy(scen.Ω),
                    eL = copy(scen.eL), eW = copy(scen.eW)),
        shed_response_mask = Bool.(collect(response_mask)),
        quadrature = (rule = settings.quad_rule,
                      nodes = copy(settings.nodes),
                      weights = copy(settings.quad_weights)))
end

function _require_same_binding(meta, cd::CaseData, snap::Snapshot,
                               scen::ScenarioData, response_mask,
                               settings)
    hasproperty(meta, :binding) || error(
        "calibration metadata has no instance binding; recalibration is required")
    expected = _calibration_binding(cd, snap, scen, response_mask, settings)
    meta.binding.case_data == expected.case_data || error(
        "calibration was produced for different physical case data or line scaling")
    meta.binding.snapshot == expected.snapshot || error(
        "calibration was produced for a different forecast snapshot (lf/wf)")
    meta.binding.scenario == expected.scenario || error(
        "calibration was produced for a different scenario support, weights, or paired errors")
    meta.binding.shed_response_mask == expected.shed_response_mask || error(
        "calibration was produced for a different shed_response_mask")
    bound_quad = meta.binding.quadrature
    current_quad = expected.quadrature
    bound_quad.rule == current_quad.rule || error(
        "quadrature rule changed after reference calibration")
    length(bound_quad.nodes) == length(current_quad.nodes) || error(
        "quadrature node count changed after reference calibration")
    all(isapprox.(Float64.(bound_quad.nodes), current_quad.nodes;
                  rtol = 1e-13, atol = 1e-15)) || error(
        "quadrature nodes changed after reference calibration")
    all(isapprox.(Float64.(bound_quad.weights), current_quad.weights;
                  rtol = 1e-13, atol = 1e-15)) || error(
        "quadrature weights changed after reference calibration")
    return nothing
end

function _kde_hinge!(model, loss, alpha, h::Float64, p::Int,
                     nodes::Vector{Float64}, weights::Vector{Float64})
    h > 0.0 || error("canonical KDE bandwidths must be strictly positive")
    length(nodes) == length(weights) || error("quadrature node/weight mismatch")
    all(isfinite, weights) || error("quadrature weights must be finite")
    all(>=(0.0), weights) || error("quadrature weights must be nonnegative")
    isapprox(sum(weights), 1.0; rtol = 1e-10, atol = 1e-12) ||
        error("quadrature weights must sum to one")
    p in (1, 2) || error("unsupported HMCR order p=$p")
    z = @variable(model, lower_bound = 0.0)
    u = @variable(model, [1:length(nodes)], lower_bound = 0.0)
    if p == 1
        for q in eachindex(nodes)
            @constraint(model,
                u[q] >= weights[q] * (loss - alpha - h * nodes[q]))
        end
        @constraint(model, z >= sum(u))
    else
        for q in eachindex(nodes)
            @constraint(model,
                u[q] >= sqrt(weights[q]) *
                        (loss - alpha - h * nodes[q]))
        end
        @constraint(model, [z; u] in SecondOrderCone())
    end
    return z
end

function _add_hmcr_risk!(model, losses::Vector, pi::Vector{Float64},
                         p::Int, gamma::Float64, h::Float64,
                         nodes::Vector{Float64}, quad_weights::Vector{Float64})
    K = length(losses)
    K >= 1 || error("HMCR reference loss vector must be nonempty")
    length(pi) == K || throw(DimensionMismatch("loss/weight length mismatch"))
    length(nodes) == length(quad_weights) ||
        throw(DimensionMismatch("quadrature node/weight mismatch"))
    !isempty(nodes) || error("quadrature rule must be nonempty")
    all(isfinite, pi) || error("reference probabilities must be finite")
    all(>=(0.0), pi) || error("reference probabilities must be nonnegative")
    isapprox(sum(pi), 1.0; rtol = 1e-10, atol = 1e-12) ||
        error("reference probabilities must sum to one")
    all(isfinite, nodes) || error("quadrature nodes must be finite")
    all(isfinite, quad_weights) || error("quadrature weights must be finite")
    all(>=(0.0), quad_weights) ||
        error("quadrature weights must be nonnegative")
    isapprox(sum(quad_weights), 1.0; rtol = 1e-10, atol = 1e-12) ||
        error("quadrature weights must sum to one")
    p in (1, 2) || error("unsupported HMCR order p=$p")
    0.0 <= gamma < 1.0 || error("HMCR gamma must lie in [0,1)")
    isfinite(h) && h > 0.0 ||
        error("canonical KDE bandwidths must be strictly positive and finite")

    alpha = @variable(model)
    Q = length(nodes)
    u = @variable(model, [1:K, 1:Q], lower_bound = 0.0)
    tail = @variable(model, lower_bound = 0.0)
    if p == 1
        for i in 1:K, q in 1:Q
            @constraint(model,
                u[i, q] >= pi[i] * quad_weights[q] *
                           (losses[i] - alpha - h * nodes[q]))
        end
        @constraint(model, tail >= sum(u))
    else
        for i in 1:K, q in 1:Q
            @constraint(model,
                u[i, q] >= sqrt(pi[i] * quad_weights[q]) *
                           (losses[i] - alpha - h * nodes[q]))
        end
        @constraint(model, [tail; vec(u)] in SecondOrderCone())
    end
    risk = alpha + tail / (1.0 - gamma)
    # Keep the legacy `z` field present for callers that inspect the return
    # schema.  Scenario-level z variables no longer exist in the exact flat
    # epigraph; `u` exposes the actual K-by-Q auxiliary matrix.
    return (risk = risk, alpha = alpha, z = nothing, tail = tail, u = u)
end

# Closed perspective of phi*(s) for Pearson phi(t)=(t-1)^2, t>=0:
#   overlinePhi(s,mu)=((s+2mu)_+)^2/(4mu)-mu for mu>0,
# with its lower-semicontinuous closure at mu=0.
function _add_pearson_perspective!(model, s, mu)
    t = @variable(model, lower_bound = 0.0)
    g = @variable(model, lower_bound = 0.0)
    @constraint(model, t >= s + 2 * mu)
    @constraint(model, [2 * mu, g, t] in RotatedSecondOrderCone())
    return g - mu
end

function _add_rs_constraint!(model, losses::Vector, pi::Vector{Float64},
                             tau::Float64, Gamma, p::Int, gamma::Float64,
                             h::Float64, nodes::Vector{Float64},
                             quad_weights::Vector{Float64}, eps_y::Float64)
    K = length(losses)
    alpha = @variable(model)
    eta = @variable(model)
    z = [_kde_hinge!(model, losses[i], alpha, h, p, nodes, quad_weights)
         for i in 1:K]
    v = @variable(model, [1:K])
    y = nothing
    coefficient = 0.0
    if p == 1
        for i in 1:K
            @constraint(model, v[i] >= z[i] / (1.0 - gamma))
        end
    else
        y = @variable(model, lower_bound = eps_y)
        for i in 1:K
            @constraint(model,
                [(1.0 - gamma) * y, v[i], z[i]] in RotatedSecondOrderCone())
        end
        coefficient = (p - 1.0) / (p * (1.0 - gamma))
    end
    perspective = [_add_pearson_perspective!(model, v[i] - eta, Gamma)
                   for i in 1:K]
    lhs = alpha - tau + coefficient * (y === nothing ? 0.0 : y) + eta +
          sum(pi[i] * perspective[i] for i in 1:K)
    constraint = @constraint(model, lhs <= 0.0)
    return (lhs = lhs, constraint = constraint, alpha = alpha, eta = eta,
            y = y, z = z, v = v, perspective = perspective,
            tau = tau, gamma = gamma, h = h)
end

function _build_xphys!(model, cd::CaseData, snap::Snapshot;
                       scen::Union{Nothing,ScenarioData} = nothing,
                       shed_response_mask = nothing)
    nG, nD, nW, nE = cd.nG, cd.nD, cd.nW, cd.nE
    response_mask = _shed_mask(shed_response_mask, nD)

    @variable(model, g[1:nG])
    @variable(model, rU[1:nG] >= 0.0)
    @variable(model, rD[1:nG] >= 0.0)
    @variable(model, snom[1:nD] >= 0.0)
    @variable(model, cW[1:nW] >= 0.0)
    @variable(model, beta[1:nG] >= 0.0)
    @variable(model, alphaS[1:nD] >= 0.0)

    @constraint(model, g .+ rU .<= cd.pmax)
    @constraint(model, g .- rD .>= cd.pmin)
    @constraint(model, snom .<= snap.lf)
    curtail_cap = scen === nothing ? snap.wf :
                  sample_safe_curtailment_cap(cd, snap, scen)
    @constraint(model, cW .<= curtail_cap)
    @constraint(model,
        sum(g) == sum(snap.lf) - sum(snap.wf) + sum(cW) - sum(snom))
    @constraint(model, sum(beta) + sum(alphaS) == 1.0)
    for j in 1:nG
        if !cd.agc_mask[j]
            @constraint(model, beta[j] == 0.0)
            @constraint(model, rU[j] == 0.0)
            @constraint(model, rD[j] == 0.0)
        end
    end
    for d in 1:nD
        !response_mask[d] && @constraint(model, alphaS[d] == 0.0)
    end

    f0 = [sum(cd.MG[ell, j] * g[j] for j in 1:nG) +
          sum(cd.MW[ell, w] * (snap.wf[w] - cW[w]) for w in 1:nW) -
          sum(cd.MD[ell, d] * (snap.lf[d] - snom[d]) for d in 1:nD)
          for ell in 1:nE]
    for ell in _finite_lines(cd)
        @constraint(model, f0[ell] <= cd.F_max[ell])
        @constraint(model, -f0[ell] <= cd.F_max[ell])
    end

    x = (g = g, rU = rU, rD = rD, snom = snom, cW = cW,
         beta = beta, alphaS = alphaS)
    return (x = x, f0 = f0)
end

function _build_xphys_events!(model, cd::CaseData, snap::Snapshot,
                              scen::ScenarioData; shed_response_mask = nothing)
    nG, nD, nE, K = cd.nG, cd.nD, cd.nE, scen.K
    _validate_scenario_samples(cd, scen)
    base = _build_xphys!(model, cd, snap; scen = scen,
                         shed_response_mask = shed_response_mask)
    x, f0 = base.x, base.f0
    g, rU, rD = x.g, x.rU, x.rD
    snom, beta, alphaS = x.snom, x.beta, x.alphaS

    events = Vector{Vector{Any}}()
    for j in findall(cd.agc_mask)
        push!(events, [-scen.Ω[i] * beta[j] - rU[j] for i in 1:K])
    end
    for j in findall(cd.agc_mask)
        push!(events, [ scen.Ω[i] * beta[j] - rD[j] for i in 1:K])
    end

    ptdf_error = scen.δ * cd.PTDF'
    MGbeta = [sum(cd.MG[ell, j] * beta[j] for j in 1:nG) for ell in 1:nE]
    MDalpha = [sum(cd.MD[ell, d] * alphaS[d] for d in 1:nD) for ell in 1:nE]
    finite_lines = _finite_lines(cd)
    line_flow = [[f0[ell] + ptdf_error[i, ell] -
                  scen.Ω[i] * (MGbeta[ell] + MDalpha[ell]) for i in 1:K]
                 for ell in finite_lines]
    for (lcol, ell) in enumerate(finite_lines)
        push!(events, [line_flow[lcol][i] - cd.F_max[ell] for i in 1:K])
    end
    for (lcol, ell) in enumerate(finite_lines)
        push!(events, [-line_flow[lcol][i] - cd.F_max[ell] for i in 1:K])
    end

    sample_shed = [[snom[d] - scen.Ω[i] * alphaS[d] for i in 1:K]
                   for d in 1:nD]
    for d in 1:nD
        push!(events, [-sample_shed[d][i] for i in 1:K])
    end
    for d in 1:nD
        push!(events,
              [sample_shed[d][i] - (snap.lf[d] + scen.eL[i, d]) for i in 1:K])
    end
    labels = scalar_event_labels(cd)
    length(events) == length(labels) || error("internal scalar-event count mismatch")

    # Canonical cost-side loss: raw MW, affine, unpenalized, and unclipped.
    cost = [sum(sample_shed[d][i] for d in 1:nD) for i in 1:K]
    return (x = x, events = events, event_labels = labels, cost = cost,
            f0 = f0, sample_shed = sample_shed)
end

function _xvalue(x)
    return (g = value.(x.g), rU = value.(x.rU), rD = value.(x.rD),
            snom = value.(x.snom), cW = value.(x.cW),
            beta = value.(x.beta), alphaS = value.(x.alphaS),
            β = value.(x.beta), αS = value.(x.alphaS))
end

function _physical_residuals(cd::CaseData, snap::Snapshot, x, response_mask)
    arrays = (x.g, x.rU, x.rD, x.snom, x.cW, x.beta, x.alphaS)
    all(values -> all(isfinite, values), arrays) ||
        return (max_violation = Inf, balance = Inf, closure = Inf,
                bounds = Inf, nominal_line = Inf,
                fixed_response = Inf)
    bound_terms = Float64[]
    append!(bound_terms, x.g .+ x.rU .- cd.pmax)
    append!(bound_terms, cd.pmin .- (x.g .- x.rD))
    append!(bound_terms, .-x.rU)
    append!(bound_terms, .-x.rD)
    append!(bound_terms, .-x.snom)
    append!(bound_terms, x.snom .- snap.lf)
    append!(bound_terms, .-x.cW)
    append!(bound_terms, x.cW .- snap.wf)
    append!(bound_terms, .-x.beta)
    append!(bound_terms, .-x.alphaS)
    bounds = maximum([0.0; bound_terms])
    balance = abs(sum(x.g) -
                  (sum(snap.lf) - sum(snap.wf) + sum(x.cW) - sum(x.snom)))
    closure = abs(sum(x.beta) + sum(x.alphaS) - 1.0)
    fixed_terms = Float64[]
    for j in 1:cd.nG
        if !cd.agc_mask[j]
            append!(fixed_terms, (abs(x.beta[j]), abs(x.rU[j]), abs(x.rD[j])))
        end
    end
    for d in 1:cd.nD
        !response_mask[d] && push!(fixed_terms, abs(x.alphaS[d]))
    end
    fixed_response = maximum([0.0; fixed_terms])
    f0 = cd.MG * x.g + cd.MW * (snap.wf - x.cW) -
         cd.MD * (snap.lf - x.snom)
    line_terms = Float64[]
    for ell in _finite_lines(cd)
        push!(line_terms, f0[ell] - cd.F_max[ell])
        push!(line_terms, -f0[ell] - cd.F_max[ell])
    end
    nominal_line = maximum([0.0; line_terms])
    return (max_violation = maximum((bounds, balance, closure,
                                     fixed_response, nominal_line)),
            balance = balance, closure = closure, bounds = bounds,
            nominal_line = nominal_line,
            fixed_response = fixed_response)
end

"""
    solve_bandwidth_pilot(cd, snap, P; shed_response_mask=nothing,
                          lex_abs_tol_mw=1e-6,
                          lex_abs_tol_cost=1e-6)

Compute the unique, forecast-only bandwidth pilot by a three-stage
lexicographic solve over exactly the deterministic physical set `Xphys`:

1. minimize `sum(snom)`;
2. fix stage 1 within an absolute MW tolerance and minimize `cgen' * g`;
3. fix both earlier objectives and minimize the normalized squared norm of
   `(g, rU, rD, snom, cW, beta, alphaS)`.

Only `snap.lf` and `snap.wf` are read.  No scenario realization, actual value,
KDE bandwidth, HMCR parameter, target, quadrature rule, or final-model weight
enters this pilot.  Zero-capacity generators are omitted from the normalized
stage-3 norm because `Xphys` already fixes their corresponding variables to
zero.  Every stage must pass the configured solver-quality gate before the
next stage is constructed.
"""
function solve_bandwidth_pilot(cd::CaseData, snap::Snapshot,
                               P::AbstractDict;
                               shed_response_mask = nothing,
                               lex_abs_tol_mw::Real = 1e-6,
                               lex_abs_tol_cost::Real = 1e-6)
    tol_mw = Float64(lex_abs_tol_mw)
    tol_cost = Float64(lex_abs_tol_cost)
    isfinite(tol_mw) && tol_mw >= 0.0 ||
        error("lex_abs_tol_mw must be nonnegative and finite")
    isfinite(tol_cost) && tol_cost >= 0.0 ||
        error("lex_abs_tol_cost must be nonnegative and finite")
    all(isfinite, snap.lf) || error("forecast load contains nonfinite values")
    all(isfinite, snap.wf) || error("forecast wind contains nonfinite values")
    all(isfinite, cd.pmax) || error("pmax contains nonfinite values")
    all(cd.pmax .>= 0.0) || error("forecast pilot requires nonnegative pmax")
    all(isfinite, cd.cgen) || error("cgen contains nonfinite values")

    stage_statuses = Any[]
    stage_objectives = Float64[]
    stage_qualities = Any[]
    stage_build_times = Float64[]
    stage_solve_times = Float64[]

    build_start = time()
    model = _new_model(P)
    built = _build_xphys!(model, cd, snap;
                           shed_response_mask = shed_response_mask)
    x = built.x
    shed_objective = sum(x.snom)
    @objective(model, Min, shed_objective)
    push!(stage_build_times, time() - build_start)

    solve_start = time()
    optimize!(model)
    push!(stage_solve_times, time() - solve_start)
    status = termination_status(model)
    quality = _quality(model)
    has_values(model) || error("bandwidth pilot stage 1 returned no primal solution; status=$status")
    _require_quality_accepted(status, quality, P, "bandwidth pilot stage 1")
    shed_star = Float64(objective_value(model))
    push!(stage_statuses, status)
    push!(stage_objectives, shed_star)
    push!(stage_qualities, quality)

    build_start = time()
    @constraint(model, shed_objective >= shed_star - tol_mw)
    @constraint(model, shed_objective <= shed_star + tol_mw)
    generation_objective = sum(cd.cgen[j] * x.g[j] for j in 1:cd.nG)
    @objective(model, Min, generation_objective)
    push!(stage_build_times, time() - build_start)

    solve_start = time()
    optimize!(model)
    push!(stage_solve_times, time() - solve_start)
    status = termination_status(model)
    quality = _quality(model)
    has_values(model) || error("bandwidth pilot stage 2 returned no primal solution; status=$status")
    _require_quality_accepted(status, quality, P, "bandwidth pilot stage 2")
    generation_star = Float64(objective_value(model))
    push!(stage_statuses, status)
    push!(stage_objectives, generation_star)
    push!(stage_qualities, quality)

    build_start = time()
    @constraint(model, generation_objective >= generation_star - tol_cost)
    @constraint(model, generation_objective <= generation_star + tol_cost)
    active_capacity = findall(>(0.0), cd.pmax)
    normalized_norm =
        sum((x.g[j] / cd.pmax[j])^2 +
            (x.rU[j] / cd.pmax[j])^2 +
            (x.rD[j] / cd.pmax[j])^2 for j in active_capacity) +
        sum((x.snom[d] / max(Float64(snap.lf[d]), 1.0))^2
            for d in 1:cd.nD) +
        sum((x.cW[w] / max(Float64(snap.wf[w]), 1.0))^2
            for w in 1:cd.nW) +
        sum(x.beta[j]^2 for j in 1:cd.nG) +
        sum(x.alphaS[d]^2 for d in 1:cd.nD)
    @objective(model, Min, normalized_norm)
    push!(stage_build_times, time() - build_start)

    solve_start = time()
    optimize!(model)
    push!(stage_solve_times, time() - solve_start)
    status = termination_status(model)
    quality = _quality(model)
    has_values(model) || error("bandwidth pilot stage 3 returned no primal solution; status=$status")
    _require_quality_accepted(status, quality, P, "bandwidth pilot stage 3")
    norm_star = Float64(objective_value(model))
    push!(stage_statuses, status)
    push!(stage_objectives, norm_star)
    push!(stage_qualities, quality)

    return (model = model, x = _xvalue(x), status = status,
            quality = quality, accepted = true,
            stage_statuses = stage_statuses,
            stage_objectives = stage_objectives,
            stage_qualities = stage_qualities,
            stage_build_times = stage_build_times,
            stage_solve_times = stage_solve_times,
            build_time = sum(stage_build_times),
            solve_time = sum(stage_solve_times),
            lex_abs_tol_mw = tol_mw,
            lex_abs_tol_cost = tol_cost,
            n_var = num_variables(model),
            n_con = length(all_constraints(model;
                                           include_variable_in_set_constraints = true)))
end

function _reference_solve(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                          P::AbstractDict, profile::ScalarMWBandwidthProfile,
                          settings, target;
                          shed_response_mask = nothing)
    build_start = time()
    model = _new_model(P)
    pi = _validate_probabilities(scen.π)
    built = _build_xphys_events!(model, cd, snap, scen;
                                 shed_response_mask = shed_response_mask)
    M = length(built.events)
    safety = [_add_hmcr_risk!(model, built.events[r], pi, settings.p,
                              settings.gamma_event[r], profile.event_h[r],
                              settings.nodes, settings.quad_weights)
              for r in 1:M]
    objective_risk = if target === :cost
        cost = _add_hmcr_risk!(model, built.cost, pi, settings.p,
                               settings.gamma_cost, profile.cost_h,
                               settings.nodes, settings.quad_weights)
        for r in 1:M
            @constraint(model, safety[r].risk <= 0.0)
        end
        cost.risk
    else
        rtarget = Int(target)
        1 <= rtarget <= M || error("invalid scalar reference target $target")
        for r in 1:M
            r == rtarget && continue
            @constraint(model, safety[r].risk <= 0.0)
        end
        safety[rtarget].risk
    end
    @objective(model, Min, objective_risk)
    build_time = time() - build_start
    solve_start = time()
    optimize!(model)
    solve_time = time() - solve_start
    status = termination_status(model)
    ok = string(status) in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(model)
    quality = _quality(model)
    return (value = ok ? objective_value(model) : NaN,
            status = status, x = ok ? _xvalue(built.x) : nothing,
            build_time = build_time, solve_time = solve_time,
            quality = quality,
            accepted = ok && _quality_accepted(status, quality, P),
            n_var = num_variables(model),
            n_con = length(all_constraints(model;
                                           include_variable_in_set_constraints = true)))
end

function _require_reference_ok(result, label::AbstractString, P::AbstractDict)
    if !(string(result.status) in ("OPTIMAL", "ALMOST_OPTIMAL"))
        q = result.quality
        error("reference target $label failed with status $(result.status); " *
              "quality=(primal=$(q.primal_residual), " *
              "dual=$(q.dual_residual), gap_abs=$(q.gap_abs), " *
              "gap_rel=$(q.gap_rel))")
    end
    isfinite(result.value) || error("reference target $label is nonfinite")
    result.x === nothing && error("reference target $label has no primal solution")
    _require_quality_accepted(result.status, result.quality, P,
                              "reference target $label")
    return nothing
end

_tau_event(z::Float64, rho::Float64) = z < 0.0 ? 0.0 : (1.0 + rho) * z

"""
    calibrate_scalar_targets(cd, snap, scen, P, profile;
                             rho_tau=get(P,"rho_tau",0), eps_z=1e-8,
                             gamma_cost=nothing, gamma_event=nothing,
                             shed_response_mask=nothing,
                             parallel_references=false)

Compute exactly `1 + M` reference targets.  The cost reference is minimized
subject to all `M` scalar reference-HMCR constraints.  Reference event `r` is
then genuinely rotated into the objective while every `r' != r` remains a
constraint.  No event is evaluated at a cost-optimal fixed decision.

Calibration fails closed unless `Z0c > eps_z`.  The returned scales are
exactly `k_c=Z0c` and `k_r=abs(Z0r)`; no floor is applied to `k_r`.
Because the cost-reference feasible set is contained in every rotated-event
reference feasible set, each exact `Z0r` must be nonpositive.  A positive
value beyond `z0r_positive_tol` is therefore treated as a solver-consistency
failure.  Event references may be solved independently with
`parallel_references=true`; results are always stored in canonical index
order.
"""
function calibrate_scalar_targets(cd::CaseData, snap::Snapshot,
                                  scen::ScenarioData, P::AbstractDict,
                                  profile::ScalarMWBandwidthProfile;
                                  rho_tau::Real = get(P, "rho_tau", 0.0),
                                  eps_z::Real = get(P, "target_z0_eps_mw", 1e-8),
                                  z0r_positive_tol::Real = get(
                                      P, "target_z0r_positive_tol_mw",
                                      get(P, "target_z0r_tol_mw",
                                          get(P, "solver_accept_gap_abs",
                                              get(P, "solver_accept_gap", 5e-5)))),
                                  gamma_cost = nothing,
                                  gamma_event = nothing,
                                  shed_response_mask = nothing,
                                  parallel_references::Bool = false)
    response_mask = _shed_mask(shed_response_mask, cd.nD)
    _validate_profile(profile, cd, snap, response_mask; scen = scen)
    _validate_scenario_samples(cd, scen)
    _validate_probabilities(scen.π)
    rho = Float64(rho_tau)
    epsilon_z = Float64(eps_z)
    z0r_tol = Float64(z0r_positive_tol)
    isfinite(rho) && 0.0 <= rho <= 1.0 ||
        error("rho_tau must lie in [0,1]")
    isfinite(epsilon_z) && epsilon_z > 0.0 || error("eps_z must be positive and finite")
    isfinite(z0r_tol) && z0r_tol >= 0.0 ||
        error("z0r_positive_tol must be nonnegative and finite")
    M = scalar_event_count(cd)
    settings = _risk_settings(P, M; gamma_cost = gamma_cost,
                              gamma_event = gamma_event)
    binding = _calibration_binding(cd, snap, scen, response_mask, settings)

    wall_start = time()
    cost_result = _reference_solve(cd, snap, scen, P, profile, settings, :cost;
                                   shed_response_mask = response_mask)
    _require_reference_ok(cost_result, "cost", P)
    Z0c = Float64(cost_result.value)
    Z0c > epsilon_z || error(
        "invalid reference batch: Z0c=$Z0c must be strictly greater than eps_z=$epsilon_z")

    Z0r = Vector{Float64}(undef, M)
    event_results = Vector{Any}(undef, M)
    if parallel_references && M > 1
        Threads.@threads for r in 1:M
            event_results[r] = _reference_solve(
                cd, snap, scen, P, profile, settings, r;
                shed_response_mask = response_mask)
        end
    else
        for r in 1:M
            event_results[r] = _reference_solve(
                cd, snap, scen, P, profile, settings, r;
                shed_response_mask = response_mask)
        end
    end
    for r in 1:M
        result = event_results[r]
        _require_reference_ok(result, profile.event_labels[r], P)
        Z0r[r] = Float64(result.value)
    end
    reference_wall_time = time() - wall_start

    z0r_positive_max = isempty(Z0r) ? 0.0 : max(0.0, maximum(Z0r))
    if z0r_positive_max > z0r_tol
        rbad = argmax(Z0r)
        error("reference nesting consistency failure: " *
              "Z0r[$rbad] ($(profile.event_labels[rbad]))=$(Z0r[rbad]) " *
              "is positive beyond z0r_positive_tol=$z0r_tol")
    end

    tau_c = (1.0 + rho) * Z0c
    tau_r = [_tau_event(Z0r[r], rho) for r in 1:M]
    k_c = Z0c
    k_r = abs.(Z0r) # Deliberately no numerical floor.
    results = [cost_result; event_results]
    qualities = [r.quality for r in results]
    primal_residuals = [q.primal_residual for q in qualities]
    dual_residuals = [q.dual_residual for q in qualities]
    gap_abs_values = [q.gap_abs for q in qualities]
    gap_rel_values = [q.gap_rel for q in qualities]
    max_quality = (
        primal_residual = maximum(abs.(primal_residuals)),
        dual_residual = maximum(abs.(dual_residuals)),
        gap_abs = maximum(abs.(gap_abs_values)),
        gap_rel = maximum(abs.(gap_rel_values)),
        gap_accept_metric = maximum(min.(abs.(gap_abs_values),
                                         abs.(gap_rel_values))))
    total_build_time = sum(r.build_time for r in results)
    total_solve_time = sum(r.solve_time for r in results)
    return (Z0c = Z0c, Z0r = Z0r,
            tau_c = tau_c, tau_r = tau_r,
            k_c = k_c, k_r = k_r,
            rho_tau = rho, eps_z = epsilon_z,
            p = settings.p,
            gamma_cost = settings.gamma_cost,
            gamma_event = copy(settings.gamma_event),
            quad_rule = settings.quad_rule,
            quad_nodes = settings.quad_nodes,
            quad_weight_floor = settings.quad_weight_floor,
            quadrature_nodes = copy(settings.nodes),
            quadrature_weights = copy(settings.quad_weights),
            cost_h = profile.cost_h,
            event_h = copy(profile.event_h),
            event_labels = copy(profile.event_labels),
            raw_n = profile.raw_n,
            statuses = [r.status for r in results],
            qualities = qualities,
            primal_residuals = primal_residuals,
            dual_residuals = dual_residuals,
            gap_abs_values = gap_abs_values,
            gap_rel_values = gap_rel_values,
            max_quality = max_quality,
            acceptance_tolerances = _acceptance_tolerances(P),
            z0r_positive_tol = z0r_tol,
            z0r_positive_max = z0r_positive_max,
            z0r_consistent = true,
            parallel_references = parallel_references,
            reference_wall_time = reference_wall_time,
            total_build_time = total_build_time,
            total_solve_time = total_solve_time,
            build_time = total_build_time,
            solve_time = total_solve_time,
            binding = binding,
            reference_solutions = [r.x for r in results])
end

function _validate_calibration(meta, profile::ScalarMWBandwidthProfile,
                               cd::CaseData, snap::Snapshot,
                               scen::ScenarioData, settings, response_mask)
    M = scalar_event_count(cd)
    length(meta.Z0r) == M || error("calibration Z0r length does not match M")
    length(meta.k_r) == M || error("calibration k_r length does not match M")
    length(meta.event_labels) == M || error("calibration labels do not match M")
    meta.event_labels == profile.event_labels ||
        error("calibration and bandwidth profile use different event orders")
    Int(meta.raw_n) == profile.raw_n ||
        error("calibration and bandwidth profile use different raw sample counts")
    all(isapprox.(Float64.(meta.event_h), profile.event_h;
                  rtol = 1e-12, atol = 0.0)) ||
        error("calibration and final model must use identical event bandwidths")
    isapprox(Float64(meta.cost_h), profile.cost_h; rtol = 1e-12, atol = 0.0) ||
        error("calibration and final model must use the identical cost bandwidth")
    Int(meta.p) == settings.p || error("HMCR p changed after reference calibration")
    isapprox(Float64(meta.gamma_cost), settings.gamma_cost;
             rtol = 1e-12, atol = 1e-14) ||
        error("cost gamma changed after reference calibration")
    all(isapprox.(Float64.(meta.gamma_event), settings.gamma_event;
                  rtol = 1e-12, atol = 1e-14)) ||
        error("event gamma changed after reference calibration")
    String(meta.quad_rule) == settings.quad_rule ||
        error("quadrature rule changed after reference calibration")
    Int(meta.quad_nodes) == settings.quad_nodes ||
        error("quadrature node count changed after reference calibration")
    Float64(meta.quad_weight_floor) == settings.quad_weight_floor ||
        error("quadrature weight-floor setting changed after reference calibration")
    length(meta.quadrature_nodes) == length(settings.nodes) ||
        error("stored quadrature nodes do not match the final model")
    all(isapprox.(Float64.(meta.quadrature_nodes), settings.nodes;
                  rtol = 1e-13, atol = 1e-15)) ||
        error("quadrature nodes changed after reference calibration")
    all(isapprox.(Float64.(meta.quadrature_weights), settings.quad_weights;
                  rtol = 1e-13, atol = 1e-15)) ||
        error("quadrature weights changed after reference calibration")
    Float64(meta.Z0c) > Float64(meta.eps_z) ||
        error("final model received an invalid nonpositive cost reference")
    all(isfinite, meta.Z0r) || error("final model received nonfinite safety references")
    maximum([0.0; Float64.(meta.Z0r)]) <= Float64(meta.z0r_positive_tol) ||
        error("final model received a positive Z0r beyond its calibration tolerance")
    isapprox(Float64(meta.k_c), Float64(meta.Z0c); rtol = 0.0, atol = 0.0) ||
        error("canonical k_c must equal Z0c exactly")
    all(isapprox.(Float64.(meta.k_r), abs.(Float64.(meta.Z0r));
                  rtol = 0.0, atol = 0.0)) ||
        error("canonical k_r must equal abs(Z0r) exactly")
    _require_same_binding(meta, cd, snap, scen, response_mask, settings)
    return nothing
end

"""
    solve_scalar_mw(cd, snap, scen, P, profile, calibration;
                    rho_tau=get(P,"rho_tau",calibration.rho_tau),
                    omega_d=get(P,"omega_d",1),
                    shed_response_mask=nothing)

Solve the canonical finite-sample model with Pearson divergence.  The final
objective is exactly `kappa_c + omega_d*kappa_d`.  Effective fragility
coefficients are exactly `calibration.k_c*kappa_c` and
`calibration.k_r[r]*kappa_d`.

Changing `p`, gamma, or any bandwidth after calibration is rejected.  Changing
`rho_tau` only retargets `tau`; changing `omega_d` only changes the pure final
objective.
"""
function solve_scalar_mw(cd::CaseData, snap::Snapshot, scen::ScenarioData,
                         P::AbstractDict, profile::ScalarMWBandwidthProfile,
                         calibration;
                         rho_tau::Real = get(P, "rho_tau", calibration.rho_tau),
                         omega_d::Real = get(P, "omega_d", 1.0),
                         gamma_cost = nothing,
                         gamma_event = nothing,
                         shed_response_mask = nothing)
    response_mask = _shed_mask(shed_response_mask, cd.nD)
    _validate_profile(profile, cd, snap, response_mask; scen = scen)
    _validate_scenario_samples(cd, scen)
    pi = _validate_probabilities(scen.π)
    M = scalar_event_count(cd)
    settings = _risk_settings(P, M;
                              gamma_cost = gamma_cost,
                              gamma_event = gamma_event)
    _validate_calibration(calibration, profile, cd, snap, scen,
                          settings, response_mask)
    phi = lowercase(String(get(P, "phi_type", "chi2")))
    phi in ("chi2", "pearson", "pearson_chi2", "pchi2") ||
        error("ScalarMWModel canonical path requires Pearson chi-square divergence")
    rho = Float64(rho_tau)
    omega = Float64(omega_d)
    isfinite(rho) && 0.0 <= rho <= 1.0 ||
        error("rho_tau must lie in [0,1]")
    isfinite(omega) && omega >= 0.0 || error("omega_d must be nonnegative and finite")
    eps_y = Float64(get(P, "eps_y", 1e-9))
    isfinite(eps_y) && eps_y > 0.0 || error("eps_y must be strictly positive and finite")

    tau_c = (1.0 + rho) * Float64(calibration.Z0c)
    tau_r = [_tau_event(Float64(calibration.Z0r[r]), rho) for r in 1:M]
    build_start = time()
    model = _new_model(P)
    built = _build_xphys_events!(model, cd, snap, scen;
                                 shed_response_mask = response_mask)
    @variable(model, kappa_c >= 0.0)
    @variable(model, kappa_d >= 0.0)

    blocks = Vector{Any}(undef, M + 1)
    blocks[1] = _add_rs_constraint!(
        model, built.cost, pi, tau_c,
        Float64(calibration.k_c) * kappa_c,
        settings.p, settings.gamma_cost, profile.cost_h,
        settings.nodes, settings.quad_weights, eps_y)
    for r in 1:M
        blocks[r + 1] = _add_rs_constraint!(
            model, built.events[r], pi, tau_r[r],
            Float64(calibration.k_r[r]) * kappa_d,
            settings.p, settings.gamma_event[r], profile.event_h[r],
            settings.nodes, settings.quad_weights, eps_y)
    end
    @objective(model, Min, kappa_c + omega * kappa_d)

    build_time = time() - build_start
    solve_start = time()
    optimize!(model)
    solve_time = time() - solve_start
    status = termination_status(model)
    solved = string(status) in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(model)
    quality = _quality(model)
    solver_accepted = solved && _quality_accepted(status, quality, P)
    labels = ["cost"; profile.event_labels]
    residuals = solved ? [value(block.lhs) for block in blocks] : fill(Inf, M + 1)
    y_margin = if solved
        [block.y === nothing ? Inf : value(block.y) - eps_y for block in blocks]
    else
        fill(-Inf, M + 1)
    end
    xvalue = solved ? _xvalue(built.x) : nothing
    physical = solved ? _physical_residuals(cd, snap, xvalue, response_mask) :
                        (max_violation = Inf, balance = Inf, closure = Inf,
                         bounds = Inf, nominal_line = Inf,
                         fixed_response = Inf)
    acceptance = _acceptance_tolerances(P)
    rs_tolerance = Float64(get(P, "solver_accept_rs_residual",
                               max(acceptance.residual, 1e-5)))
    physical_tolerance = Float64(get(P, "solver_accept_physical_abs", 1e-5))
    y_tolerance = Float64(get(P, "solver_accept_y_violation",
                              acceptance.residual))
    all(isfinite, (rs_tolerance, physical_tolerance, y_tolerance)) &&
        rs_tolerance >= 0.0 && physical_tolerance >= 0.0 && y_tolerance >= 0.0 ||
        error("final acceptance tolerances must be nonnegative and finite")
    rs_accepted = solved && maximum(residuals) <= rs_tolerance
    y_accepted = solved && minimum(y_margin) >= -y_tolerance
    physical_accepted = solved && physical.max_violation <= physical_tolerance
    fragility_accepted = solved && isfinite(value(kappa_c)) &&
                         isfinite(value(kappa_d)) &&
                         value(kappa_c) >= -physical_tolerance &&
                         value(kappa_d) >= -physical_tolerance
    accepted = solver_accepted && rs_accepted && y_accepted &&
               physical_accepted && fragility_accepted
    return (model = model, status = status, solved = solved, x = xvalue,
            kappa_c = solved ? value(kappa_c) : NaN,
            kappa_d = solved ? value(kappa_d) : NaN,
            objective = solved ? objective_value(model) : NaN,
            rho_tau = rho, omega_d = omega,
            tau_c = tau_c, tau_r = tau_r,
            k_c = Float64(calibration.k_c),
            k_r = Float64.(calibration.k_r),
            event_labels = copy(profile.event_labels),
            rs_labels = labels, rs_residuals = residuals,
            max_rs_residual = maximum(residuals),
            worst_rs_label = labels[argmax(residuals)],
            y_margin = y_margin, min_y_margin = minimum(y_margin),
            build_time = build_time, solve_time = solve_time,
            quality = quality,
            solver_accepted = solver_accepted,
            rs_accepted = rs_accepted,
            y_accepted = y_accepted,
            physical_accepted = physical_accepted,
            fragility_accepted = fragility_accepted,
            accepted = accepted,
            rs_tolerance = rs_tolerance,
            y_tolerance = y_tolerance,
            physical_tolerance = physical_tolerance,
            physical_residuals = physical,
            n_var = num_variables(model),
            n_con = length(all_constraints(model;
                                           include_variable_in_set_constraints = true)),
            calibration = calibration)
end

end # module ScalarMWModel
