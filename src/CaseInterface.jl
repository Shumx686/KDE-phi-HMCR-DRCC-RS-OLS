# =====================================================================
#  CaseInterface.jl  —  模块 2: 样本数据 -> 模型接口
#
#  产出两个不可变结构体, 所有公开模型(M0-M6)共享 => 公平性由构造保证:
#    CaseData      网络与机组的确定性部分 (PTDF, M^G/M^D/M^W, 界, 成本, β̄)
#    ScenarioData  压缩后的误差分布 (δ̄, e_L, e_W, Ω, π) + 快照预测点
# =====================================================================
module CaseInterface

using LinearAlgebra
using ..Cases

export CaseData, ScenarioData, build_casedata, build_scenariodata, Snapshot
export apply_generator_availability

# ---------------------------------------------------------------------
struct CaseData
    name::String
    nbus::Int
    busidx::Dict{Int,Int}
    # 机组 (仅在线)
    nG::Int
    gen_bus::Vector{Int}
    pmax::Vector{Float64}
    pmin::Vector{Float64}
    cgen::Vector{Float64}
    agc_mask::Vector{Bool}
    βbar::Vector{Float64}        # AGC 方向, sum=1, 非AGC=0
    # 负荷
    nD::Int
    load_buses::Vector{Int}
    cshed::Vector{Float64}       # VOLL
    # 风电
    nW::Int
    wind_buses::Vector{Int}
    wcap::Vector{Float64}
    # 网络
    nE::Int
    F_max::Vector{Float64}
    PTDF::Matrix{Float64}        # nE × nbus
    MG::Matrix{Float64}          # nE × nG
    MD::Matrix{Float64}          # nE × nD
    MW::Matrix{Float64}          # nE × nW
end

"快照: 第 t 个测试小时的预测点 (与 ScenarioData 的误差分布共享)。"
struct Snapshot
    lf::Vector{Float64}          # nD
    wf::Vector{Float64}          # nW
    lact::Vector{Float64}        # nD (审计用)
    wact::Vector{Float64}        # nW (审计用)
    δ::Vector{Float64}           # nbus (审计用, 该快照实际净注入误差)
end

struct ScenarioData
    K::Int
    π::Vector{Float64}           # K, sum=1
    δ::Matrix{Float64}           # K × nbus
    Ω::Vector{Float64}           # K
    eL::Matrix{Float64}          # K × nD  负荷误差分量
    eW::Matrix{Float64}          # K × nW  风电误差分量
    is_tail::Vector{Bool}
    raw_n::Int                   # 压缩前训练样本数
    raw_mean::Vector{Float64}    # 原始训练误差均值
    raw_cov::Matrix{Float64}     # 原始训练经验协方差 (1/N 口径)
    raw_lower::Vector{Float64}   # 原始训练坐标下界
    raw_upper::Vector{Float64}   # 原始训练坐标上界
    compression_l1::Float64     # 实际聚类耦合的 L1 成本；指定中心权重下 W1 的可证上界
end

"""
    sample_safe_curtailment_cap(cd, snap, scen)

Return the componentwise here-and-now wind-curtailment upper bound that is
feasible for every representative wind-error sample.  Representative
available wind is reconstructed as `snap.wf + scen.eW` and clipped to the
physical interval `[0, wcap]`; the common curtailment decision cannot exceed
the smallest reconstructed availability.  This prevents negative utilized
wind in low-wind samples.
"""
function sample_safe_curtailment_cap(cd::CaseData, snap::Snapshot,
                                     scen::ScenarioData)
    size(scen.eW) == (scen.K, cd.nW) ||
        throw(DimensionMismatch("scenario wind-error components do not match nW"))
    length(snap.wf) == cd.nW ||
        throw(DimensionMismatch("forecast wind vector does not match nW"))
    cap = Vector{Float64}(undef, cd.nW)
    for w in 1:cd.nW
        available = clamp.(snap.wf[w] .+ view(scen.eW, :, w), 0.0, cd.wcap[w])
        cap[w] = min(snap.wf[w], minimum(available))
    end
    all(isfinite, cap) || error("sample-safe curtailment cap is nonfinite")
    all(cap .>= -1e-12) || error("sample-safe curtailment cap is negative")
    return max.(cap, 0.0)
end

# ---------------------------------------------------------------------
function build_casedata(rc::RawCase, load_buses::Vector{Int},
                        wind_buses::Vector{Int}, wcap::Vector{Float64}, cfg::Dict;
                        gen_keep::Union{Nothing,Vector{Bool}} = nothing)
    PTDF, busids, busidx, F_max, _, _, _ = build_ptdf(rc)
    nbus = length(busids); nE = size(PTDF, 1)

    gd = Cases.gendata(rc)
    on = gen_keep === nothing ? gd.status : (gd.status .& gen_keep)   # GMLC: 仅常规机组
    gen_bus = gd.bus[on]; pmax = gd.pmax[on]; pmin = gd.pmin[on]; cgen = gd.cost[on]
    nG = length(gen_bus)

    # AGC 方向 β̄
    # Only online units with a positive regulating range can participate in
    # AGC. Some test cases retain zero-capacity records with status=1; treating
    # them as AGC units inserts an identically zero reserve event and pins the
    # grouped reserve target at zero.
    agc_mask = (pmax .> 0.0) .& (pmax .> pmin .+ 1e-9)
    βbar = zeros(nG)
    if cfg["network"]["agc"]["mode"] == "by_pmax"
        w = [agc_mask[j] ? max(pmax[j], 0.0) : 0.0 for j in 1:nG]
        βbar = w ./ sum(w)
    else                                       # uniform
        na = count(agc_mask)
        βbar = [agc_mask[j] ? 1/na : 0.0 for j in 1:nG]
    end

    Sg = selection_matrix(busidx, nbus, gen_bus)
    Sd = selection_matrix(busidx, nbus, load_buses)
    Sw = selection_matrix(busidx, nbus, wind_buses)
    MG = PTDF * Matrix(Sg); MD = PTDF * Matrix(Sd); MW = PTDF * Matrix(Sw)

    voll = get(cfg["model"], "voll", 1000.0)
    cshed = fill(float(voll), length(load_buses))

    return CaseData(rc.name, nbus, busidx, nG, gen_bus, pmax, pmin, cgen,
                    agc_mask, βbar, length(load_buses), load_buses, cshed,
                    length(wind_buses), wind_buses, wcap, nE, F_max,
                    Matrix(PTDF), MG, MD, MW)
end

"返回线路上限按 scale 收紧的 CaseData 副本(高压测试用)。"
function scale_Fmax(cd::CaseData, scale::Float64)
    return CaseData(cd.name, cd.nbus, cd.busidx, cd.nG, cd.gen_bus, cd.pmax, cd.pmin,
                    cd.cgen, cd.agc_mask, cd.βbar, cd.nD, cd.load_buses, cd.cshed,
                    cd.nW, cd.wind_buses, cd.wcap, cd.nE,
                    [isfinite(f) ? f*scale : f for f in cd.F_max],
                    cd.PTDF, cd.MG, cd.MD, cd.MW)
end
export scale_Fmax

"""
    apply_generator_availability(cd, available)

Return a `CaseData` copy for a sampled generator-availability state while
keeping the generator dimension fixed. Unavailable units have `pmin=pmax=0`,
are removed from AGC participation, and receive zero participation weight.
Keeping the dimension fixed makes all models use the same outage state and
avoids model recompilation caused only by changing generator counts.
"""
function apply_generator_availability(cd::CaseData, available::AbstractVector{Bool})
    length(available) == cd.nG ||
        throw(DimensionMismatch("generator availability length $(length(available)) != nG $(cd.nG)"))

    pmax = copy(cd.pmax)
    pmin = copy(cd.pmin)
    pmax[.!available] .= 0.0
    pmin[.!available] .= 0.0

    agc_mask = cd.agc_mask .& available .& (pmax .> 0.0)
    weights = [agc_mask[j] ? max(pmax[j], 0.0) : 0.0 for j in 1:cd.nG]
    βbar = sum(weights) > 0 ? weights ./ sum(weights) : zeros(cd.nG)

    return CaseData(cd.name, cd.nbus, cd.busidx, cd.nG, cd.gen_bus, pmax, pmin,
                    cd.cgen, agc_mask, βbar, cd.nD, cd.load_buses, cd.cshed,
                    cd.nW, cd.wind_buses, cd.wcap, cd.nE, cd.F_max,
                    cd.PTDF, cd.MG, cd.MD, cd.MW)
end

"从数据管线的 scenarios NamedTuple 构造 ScenarioData。"
function build_scenariodata(sc)
    K = size(sc.δbar, 1)
    Ω = vec(sum(sc.δbar, dims=2))
    if hasproperty(sc, :raw_mean)
        return ScenarioData(K, copy(sc.π), copy(sc.δbar), Ω,
                            copy(sc.eLbar), copy(sc.eWbar), copy(sc.is_tail),
                            Int(sc.raw_n), copy(sc.raw_mean), copy(sc.raw_cov),
                            copy(sc.raw_lower), copy(sc.raw_upper),
                            Float64(sc.compression_l1))
    end
    # Backward-compatible fallback for old serialized caches. Versioned pipeline
    # reloads regenerate these summaries before formal experiments.
    mu = vec(sum(sc.δbar .* reshape(sc.π, :, 1), dims = 1))
    centered = sc.δbar .- mu'
    sigma = centered' * (centered .* reshape(sc.π, :, 1))
    return ScenarioData(K, copy(sc.π), copy(sc.δbar), Ω,
                        copy(sc.eLbar), copy(sc.eWbar), copy(sc.is_tail),
                        K, mu, sigma, vec(minimum(sc.δbar, dims = 1)),
                        vec(maximum(sc.δbar, dims = 1)), 0.0)
end

"取面板第 t 小时为快照。panel 来自 DataPipeline。"
function snapshot_at(panel, t::Int, cd::CaseData)
    return Snapshot(panel.Lf[t, :], panel.Wf[t, :],
                    panel.Lact[t, :], panel.Wact[t, :], panel.δ[t, :])
end

export snapshot_at
export sample_safe_curtailment_cap

end # module
