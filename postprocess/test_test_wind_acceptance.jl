using Test
include(joinpath(@__DIR__, "TestWindAcceptance.jl"))
using .TestWindAcceptance

function sample_row(excess; accepted = true, model = "M0")
    row = Dict{String,Any}("model" => model, "accepted" => accepted,
        "shed_mw" => accepted ? 2.0 : 100.0, "curtail_mw" => 3.0,
        "LOLP_event" => true, "failure_reason" => "", "status" => "ALMOST_OPTIMAL",
        "T_on_s" => 7.0, "T_e2e_s" => 8.0, "max_safety_excess_mw" => 0.0,
        "wind_curtail_excess_mw" => excess, "wind_curtail_event" => false)
    for direction in TestWindAcceptance.DIRECTIONS
        row["$(direction)_event"] = false
        row["$(direction)_excess_mw"] = 0.0
    end
    return row
end

@testset "Uniform test-wind acceptance" begin
    for value in (0.0, 1e-8, TEST_WIND_TOL_MW)
        row = apply_test_wind_gate!(sample_row(value), 100.0)
        @test row["accepted"] && !row["test_wind_rejected"]
        @test row["shed_mw"] == 2.0
    end
    for model in ("M0", "M1", "M2", "M3", "M3-ClassMax", "M4", "M5",
                  "Proposed-Tradeoff", "Proposed-Safety")
        for value in (nextfloat(TEST_WIND_TOL_MW), 13.5, Inf, NaN, missing)
            row = apply_test_wind_gate!(sample_row(value; model), 100.0)
            @test !row["accepted"] && row["test_wind_rejected"]
            @test row["shed_mw"] == 100.0 && row["curtail_mw"] == 0.0
            @test row["pre_test_wind_gate_shed_mw"] == 2.0
            @test all(row["$(d)_event"] for d in TestWindAcceptance.DIRECTIONS)
            @test row["status"] == "ALMOST_OPTIMAL" && row["T_on_s"] == 7.0
            @test isequal(row["wind_curtail_excess_mw"], value)
            original = deepcopy(row)
            @test isequal(apply_test_wind_gate!(row, 100.0), original)
            @test_throws ErrorException apply_test_wind_gate!(row, 101.0)
        end
    end
    row = apply_test_wind_gate!(sample_row(Inf; accepted = false), 100.0)
    @test !row["accepted"] && !row["test_wind_rejected"]
    @test row["shed_mw"] == 100.0
    @test_throws ErrorException apply_test_wind_gate!(sample_row(0.0), -1.0)
end
