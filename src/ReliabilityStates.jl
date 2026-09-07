# =====================================================================
#  ReliabilityStates.jl — enhanced NSMC component-state sampling
#
#  The main experiment historically sampled only load/wind operating
#  snapshots. This module adds reproducible generator forced-outage states
#  without changing the optimization model or silently changing dimensions.
#
#  Line outages are deliberately not sampled here. A physically correct line
#  outage implementation must rebuild the network by connected component and
#  impose island-wise power balance/recourse. Setting an outaged line's limit
#  to zero under the intact PTDF is not an acceptable substitute.
# =====================================================================
module ReliabilityStates

using Random
using StableRNGs
using ..CaseInterface

export generator_unavailability, sample_generator_states
export apply_generator_state, generator_state_summary

"""
    generator_unavailability(system, cd; profile="standard")

Return per-unit forced-outage probabilities in the exact generator order used
by `CaseData`.

- RTS79 uses the MTTF/MTTR data supplied with the reviewed RA_NSMCS package:
  `q = MTTR / (MTTF + MTTR)`.
- RBTS uses the standard educational-system FOR values mapped to the unit
  order in `RBTS.txt` (40, 40, 10, 20 MW at bus 1; then 5, 5, 40 and four
  20 MW units at bus 2).

The values are explicit and version-controlled so that every experiment
records the same reliability profile.
"""
function generator_unavailability(system::AbstractString, cd::CaseData;
                                  profile::AbstractString = "standard")
    lowercase(profile) == "standard" ||
        error("unknown generator reliability profile: $profile")
    sys = uppercase(strip(system))

    q = if sys == "RTS79"
        mttf = Float64[
            450, 450, 1960, 1960, 450,
            450, 1960, 1960, 1200, 1200,
            1200, 950, 950, 950, 10000,
            2940, 2940, 2940, 2940, 2940,
            960, 960, 1100, 1100, 1980,
            1980, 1980, 1980, 1980, 1980,
            960, 960, 1150,
        ]
        mttr = Float64[
            50, 50, 40, 40, 50,
            50, 40, 40, 50, 50,
            50, 50, 50, 50, 0.1,
            60, 60, 60, 60, 60,
            40, 40, 150, 150, 20,
            20, 20, 20, 20, 20,
            40, 40, 100,
        ]
        mttr ./ (mttf .+ mttr)
    elseif sys == "RBTS"
        Float64[
            0.015, 0.015, 0.020, 0.025,
            0.020, 0.020, 0.030,
            0.025, 0.025, 0.025, 0.025,
        ]
    else
        error("no standard generator reliability profile is configured for $system")
    end

    length(q) == cd.nG ||
        error("$system reliability profile has $(length(q)) units but CaseData has $(cd.nG)")
    all((q .>= 0.0) .& (q .< 1.0)) ||
        error("invalid generator unavailability values for $system")
    return q
end

"""
    sample_generator_states(system, cd, n, seed; profile="standard")

Sample an `n × nG` Boolean matrix with `true` meaning the generator is down.
The StableRNG seed makes the component states reproducible and shared by all
models in a comparison.
"""
function sample_generator_states(system::AbstractString, cd::CaseData, n::Integer,
                                 seed::Integer; profile::AbstractString = "standard")
    n >= 1 || error("number of generator-state draws must be positive")
    q = generator_unavailability(system, cd; profile = profile)
    rng = StableRNG(seed)
    down = falses(n, cd.nG)
    for i in 1:n, j in 1:cd.nG
        down[i, j] = rand(rng) < q[j]
    end
    return (down = down, q = q, seed = Int(seed), profile = String(profile))
end

"Apply one sampled generator state to a fixed-dimension `CaseData`."
function apply_generator_state(cd::CaseData, down::AbstractVector{Bool})
    return apply_generator_availability(cd, .!down)
end

"Compact diagnostics for one sampled generator state."
function generator_state_summary(base_cd::CaseData, state_cd::CaseData,
                                 down::AbstractVector{Bool})
    return (
        generator_outages = count(down),
        available_generators = count(.!down),
        available_capacity_mw = sum(state_cd.pmax),
        available_capacity_fraction =
            sum(base_cd.pmax) > 0 ? sum(state_cd.pmax) / sum(base_cd.pmax) : NaN,
    )
end

end # module
