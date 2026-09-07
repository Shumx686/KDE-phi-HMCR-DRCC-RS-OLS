#!/usr/bin/env julia

# Minimal executable guard for the two distinct target-relaxation equations.
using Test
const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "DRCCExp.jl"))
include(joinpath(@__DIR__, "FinalSevenTargetModel.jl"))
include(joinpath(@__DIR__, "FinalExperimentSupport.jl"))
using .FinalSevenTargetModel
using .FinalExperimentSupport

bandwidth = FinalBandwidth(1.0, fill(1.0, 6), 20, 1.0, 1.0e-6, 1.0e-6)
base = FinalCalibration(2.0, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                        2.0, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0],
                        0.0, 0.0, bandwidth, fill("OPTIMAL", 7))
retargeted = retarget(base; rho_tau_c = 1.0, rho_tau_s = 0.75)

@test retargeted.tau_c == 4.0
@test retargeted.tau_m == [-0.25, -0.5, -0.75, -1.0, -1.25, -1.5]
@test retargeted.rho_tau_c == 1.0
@test retargeted.rho_tau_s == 0.75
@test isdefined(FinalExperimentSupport, :select_two_target_weight_scale)
positive_safety = FinalCalibration(2.0, [1e-7, -2.0, -3.0, -4.0, -5.0, -6.0],
    2.0, [1e-7, -2.0, -3.0, -4.0, -5.0, -6.0], 0.0, 0.0,
    bandwidth, fill("OPTIMAL", 7))
@test_throws ErrorException retarget(positive_safety; rho_tau_c = 0.0, rho_tau_s = 0.0)
println("two-target relaxation unit test passed")
