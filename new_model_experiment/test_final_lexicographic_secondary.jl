#!/usr/bin/env julia

using Test
using JuMP

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "DRCCExp.jl"))
include(joinpath(@__DIR__, "FinalSevenTargetModel.jl"))
using .FinalSevenTargetModel

@testset "final-model lexicographic secondary" begin
    P = Dict{String,Any}(
        "max_iter" => 1000,
        "tol_gap_abs" => 1e-9,
        "tol_gap_rel" => 1e-9,
        "tol_feas" => 1e-9,
    )
    m = FinalSevenTargetModel._model(P)
    @variable(m, g[1:1])
    @variable(m, rU[1:1])
    @variable(m, rD[1:1])
    @variable(m, 0 <= snom[1:1] <= 10)
    @variable(m, cW[1:1])
    @variable(m, beta[1:1])
    @variable(m, alphaS[1:1])
    @variable(m, kappa_c >= 0)
    @variable(m, kappa_m[1:6] >= 0)
    v = (g = g, rU = rU, rD = rD, snom = snom, cW = cW,
         beta = beta, alphaS = alphaS)
    primary = kappa_c + sum(kappa_m)
    @objective(m, Min, primary)
    result = FinalSevenTargetModel._result(m, v, kappa_c, kappa_m, primary, 0.0;
        lex_secondary_builder = _ -> snom[1],
        lex_abs_tol = 5e-5,
        lex_gap_multiplier = 10.0,
        lex_activation_tol = 1e-4)

    @test result.lexicographic_enabled
    @test result.lexicographic_activated
    @test result.lexicographic_stage1_status in ("OPTIMAL", "ALMOST_OPTIMAL")
    @test result.lexicographic_stage2_status in ("OPTIMAL", "ALMOST_OPTIMAL")
    @test result.x.snom[1] <= 1e-6
    @test result.lexicographic_primary_final <=
          result.lexicographic_primary_star + result.lexicographic_primary_tolerance + 1e-7
    @test result.stage1_solve_time >= 0.0
    @test result.stage2_solve_time >= 0.0
end

println("final-model lexicographic secondary unit test passed")
