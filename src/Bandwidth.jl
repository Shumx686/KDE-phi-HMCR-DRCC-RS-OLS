module Bandwidth

using Statistics
using ..CaseInterface: CaseData, Snapshot

export FixedBandwidthProfile, build_fixed_bandwidth_profile,
       build_target_specific_bandwidth_profile

struct FixedBandwidthProfile
    event_h::Vector{Float64}
    target_h::Vector{Float64}
    cost_h::Float64
    event_labels::Vector{String}
    target_labels::Vector{String}
    raw_n::Int
    multiplier::Float64
end

function _sample_sigma(v::AbstractVector)
    length(v) <= 1 && return 0.0
    sigma = std(v; corrected = true)
    return isfinite(sigma) && sigma > eps(Float64) ? Float64(sigma) : 0.0
end

function _silverman(v::AbstractVector, multiplier::Float64;
                    bandwidth_floor::Float64 = 0.0)
    bandwidth_floor >= 0 || error("bandwidth_floor must be nonnegative")
    sigma = _sample_sigma(v)
    sigma == 0.0 && return bandwidth_floor
    return max(multiplier * 1.06 * sigma * length(v)^(-1 / 5),
               bandwidth_floor)
end

function _category_max(values::AbstractMatrix)
    # ModelCore defines an empty grouped category (for example after an outage
    # removes every AGC unit) as the constant-zero loss.  Keep bandwidth
    # estimation on exactly the same domain; the configured positive floor then
    # supplies a valid KDE bandwidth for that degenerate sample.
    size(values, 2) == 0 && return zeros(Float64, size(values, 1))
    return vec(maximum(values, dims = 2))
end

"""
    build_fixed_bandwidth_profile(cd, snap, delta, omega, eL, pilot_x; multiplier=1.0)

Compute fixed, target-specific Silverman bandwidths from uncompressed training
errors under one supplied pilot affine-recourse decision. Losses use the same
normalization and event order as `ModelCore.build_ols`.
"""
function build_fixed_bandwidth_profile(cd::CaseData, snap::Snapshot,
                                       delta::AbstractMatrix,
                                       omega::AbstractVector,
                                       eL::AbstractMatrix, pilot_x;
                                       multiplier::Real = 1.0,
                                       relative_floor::Real = 0.0,
                                       cost_scale::Real = 1.0,
                                       cost_loss::Symbol = :affine)
    n = length(omega)
    size(delta, 1) == n || error("delta/omega row mismatch")
    size(eL, 1) == n || error("eL/omega row mismatch")
    size(delta, 2) == cd.nbus || error("delta bus dimension mismatch")
    size(eL, 2) == cd.nD || error("eL load dimension mismatch")
    multiplier > 0 || error("bandwidth multiplier must be positive")
    relative_floor >= 0 || error("relative bandwidth floor must be nonnegative")
    cost_scale > 0 || error("cost_scale must be positive")
    cost_loss in (:affine, :positive_part) ||
        error("cost_loss must be :affine or :positive_part")

    safety_floor = Float64(relative_floor)
    cost_floor = Float64(relative_floor * cost_scale)

    beta = getproperty(pilot_x, Symbol(Char(0x03b2)))
    alphaS = getproperty(pilot_x, Symbol(string(Char(0x03b1), "S")))
    agc = findall(cd.agc_mask)
    finite_lines = findall(isfinite, cd.F_max)

    pscale = max.(cd.pmax[agc], 1.0)
    q_up = (-omega .* beta[agc]' .- pilot_x.rU[agc]') ./ pscale'
    q_down = (omega .* beta[agc]' .- pilot_x.rD[agc]') ./ pscale'

    f0 = cd.MG * pilot_x.g + cd.MW * (snap.wf - pilot_x.cW) -
         cd.MD * (snap.lf - pilot_x.snom)
    response = cd.MG * beta + cd.MD * alphaS
    ptdf_error = delta * cd.PTDF[finite_lines, :]'
    flow = ptdf_error .+ f0[finite_lines]' .- omega .* response[finite_lines]'
    fscale = max.(cd.F_max[finite_lines], 1.0)
    q_line_pos = (flow .- cd.F_max[finite_lines]') ./ fscale'
    q_line_neg = (-flow .- cd.F_max[finite_lines]') ./ fscale'

    shed = ones(n) * pilot_x.snom' .- omega .* alphaS'
    lscale = max.(snap.lf, 1.0)
    q_shed_low = -shed ./ lscale'
    q_shed_high = (shed .- (ones(n) * snap.lf' .+ eL)) ./ lscale'

    scale = Float64(multiplier)
    event_h = Float64[]
    event_labels = String[]
    for (jcol, j) in enumerate(agc)
        push!(event_h, _silverman(view(q_up, :, jcol), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "reserve_up_G$(j)")
        push!(event_h, _silverman(view(q_down, :, jcol), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "reserve_down_G$(j)")
    end
    for (lcol, ell) in enumerate(finite_lines)
        push!(event_h, _silverman(view(q_line_pos, :, lcol), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "line_pos_L$(ell)")
        push!(event_h, _silverman(view(q_line_neg, :, lcol), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "line_neg_L$(ell)")
    end
    for d in 1:cd.nD
        push!(event_h, _silverman(view(q_shed_low, :, d), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "shed_low_D$(d)")
        push!(event_h, _silverman(view(q_shed_high, :, d), scale;
                                  bandwidth_floor = safety_floor))
        push!(event_labels, "shed_high_D$(d)")
    end

    target_values = [
        _category_max(q_up),
        _category_max(q_down),
        _category_max(q_line_pos),
        _category_max(q_line_neg),
        _category_max(q_shed_low),
        _category_max(q_shed_high),
    ]
    target_labels = ["reserve_up", "reserve_down", "line_pos", "line_neg",
                     "shed_low", "shed_high"]
    target_h = [_silverman(v, scale; bandwidth_floor = safety_floor)
                for v in target_values]

    cost = if cost_loss == :positive_part
        [sum(cd.cshed[d] * max(pilot_x.snom[d] - omega[i] * alphaS[d], 0.0)
             for d in 1:cd.nD) for i in 1:n]
    else
        [sum(cd.cshed[d] * (pilot_x.snom[d] - omega[i] * alphaS[d])
             for d in 1:cd.nD) for i in 1:n]
    end
    cost_h = _silverman(cost, scale; bandwidth_floor = cost_floor)

    return FixedBandwidthProfile(event_h, target_h, cost_h, event_labels,
                                 target_labels, n, scale)
end

"""
    build_target_specific_bandwidth_profile(cd, snap, delta, omega, eL,
                                            reference_solutions;
                                            event_pilot, multiplier=1.0)

Construct the six M6 safety bandwidths from target-specific no-smoothing
reference solutions. `reference_solutions` follows `calibrate_targets` order:
cost first, then safety targets 1--6. The elementary-event bandwidths are kept
from one explicitly supplied common pilot for M2/M3 diagnostics.
"""
function build_target_specific_bandwidth_profile(cd::CaseData, snap::Snapshot,
                                                 delta::AbstractMatrix,
                                                 omega::AbstractVector,
                                                 eL::AbstractMatrix,
                                                 reference_solutions;
                                                 event_pilot,
                                                 multiplier::Real = 1.0,
                                                 relative_floor::Real = 0.0,
                                                 cost_scale::Real = 1.0,
                                                 cost_loss::Symbol = :affine)
    length(reference_solutions) == 7 ||
        error("reference_solutions must contain cost plus six safety solutions")
    any(isnothing, reference_solutions) &&
        error("all no-smoothing reference solutions must be available")

    event_profile = build_fixed_bandwidth_profile(cd, snap, delta, omega, eL,
                                                    event_pilot;
                                                    multiplier = multiplier,
                                                    relative_floor = relative_floor,
                                                    cost_scale = cost_scale,
                                                    cost_loss = cost_loss)
    cost_profile = build_fixed_bandwidth_profile(cd, snap, delta, omega, eL,
                                                   reference_solutions[1];
                                                   multiplier = multiplier,
                                                   relative_floor = relative_floor,
                                                   cost_scale = cost_scale,
                                                   cost_loss = cost_loss)
    target_h = zeros(6)
    for mi in 1:6
        profile = build_fixed_bandwidth_profile(cd, snap, delta, omega, eL,
                                                reference_solutions[mi + 1];
                                                multiplier = multiplier,
                                                relative_floor = relative_floor,
                                                cost_scale = cost_scale,
                                                cost_loss = cost_loss)
        target_h[mi] = profile.target_h[mi]
    end
    return FixedBandwidthProfile(event_profile.event_h, target_h,
                                 cost_profile.cost_h,
                                 event_profile.event_labels,
                                 event_profile.target_labels,
                                 event_profile.raw_n, Float64(multiplier))
end

end # module
