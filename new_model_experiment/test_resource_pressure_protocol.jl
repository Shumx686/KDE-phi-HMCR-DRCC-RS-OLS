#!/usr/bin/env julia

using Test
using Dates
using TOML

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "DRCCExp.jl"))
using .DRCCExp
using .DRCCExp: Cases, CaseInterface, DataPipeline
include(joinpath(@__DIR__, "FinalSevenTargetModel.jl"))
include(joinpath(@__DIR__, "FinalExperimentSupport.jl"))
using .FinalExperimentSupport

function two_bus_case()
    bus = zeros(2, 13)
    bus[:, 1] .= [1, 2]
    bus[:, 2] .= [3, 1]
    gen = zeros(2, 10)
    gen[:, 1] .= [1, 2]
    gen[:, 8] .= 1.0
    gen[:, 9] .= [120.0, 80.0]
    gen[:, 10] .= [20.0, 10.0]
    branch = zeros(1, 11)
    branch[1, 1] = 1
    branch[1, 2] = 2
    branch[1, 4] = 0.1
    branch[1, 6] = 100.0
    branch[1, 11] = 1.0
    gencost = [2.0 0.0 0.0 2.0 10.0 0.0;
               2.0 0.0 0.0 2.0 20.0 0.0]
    return Cases.RawCase("TWO_BUS", 100.0, bus, gen, branch, gencost)
end

@testset "revised resource-pressure transformation" begin
    rawcase = two_bus_case()
    cfg = TOML.parsefile(joinpath(ROOT, "config", "experiment.toml"))
    panel = DataPipeline.Panel("TEST", [DateTime(2020, 1, 1), DateTime(2020, 1, 1, 1)],
        [2], [2], [50.0], reshape([100.0, 100.0], 2, 1),
        reshape([101.0, 99.0], 2, 1), reshape([20.0, 30.0], 2, 1),
        reshape([10.0, 40.0], 2, 1), zeros(2, 2), zeros(2))
    panel = DataPipeline.build_delta!(panel, rawcase)
    cd = build_casedata(rawcase, panel.load_buses, panel.wind_buses, panel.wind_cap, cfg)

    transformed = wind_scaled_panel(panel, rawcase, 2.0; lambda_eps = 1.5)
    @test transformed.wind_cap == [100.0]
    @test transformed.Wf[:, 1] == [40.0, 60.0]
    # Original errors (-10,+10) are multiplied by 2*1.5 and then clipped.
    @test transformed.Wact[:, 1] == [10.0, 90.0]
    @test transformed.Lf == panel.Lf
    @test transformed.Lact == panel.Lact

    scale = target_wind_scale(cd, 0.40, 1.0)
    stressed, meta = replacement_casedata(cd, scale; replacement_ratio = 1.0,
                                           target_dispatchable_pmax_mw = 150.0)
    @test isapprox(meta.replacement_wind_penetration, 0.40; atol = 1e-12)
    @test isapprox(sum(stressed.pmax), 150.0; atol = 1e-10)
    @test all(stressed.pmin .<= stressed.pmax)
    @test isapprox(sum(stressed.βbar), 1.0; atol = 1e-12)
    @test_throws ErrorException replacement_casedata(cd, scale;
        replacement_ratio = 1.0, target_dispatchable_pmax_mw = 250.0)

    rebuilt = apply_dispatchable_profile(cd, stressed)
    @test rebuilt.pmax == stressed.pmax
    @test rebuilt.pmin == stressed.pmin
    unavailable = CaseInterface.apply_generator_availability(rebuilt, Bool[false, true])
    @test unavailable.pmax[1] == 0.0
    @test unavailable.pmax[2] == stressed.pmax[2]
end

println("resource-pressure protocol unit test passed")
