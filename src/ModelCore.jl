# =====================================================================
#  ModelCore.jl  —  模块 3: 统一参数化凸模型 builder + 标定层
#
#  一个 builder 用开关覆盖 M1-M3 和 proposed core (public id M6);
#  M0 单独最简，公平性由共享 CaseData 保证:
#    safety :none | :cvar(h=0,θ=0) | :kde(h>0,θ=0) | :kde_phi(θ>0)
#           | :kde_phi_grouped(θ>0; six-category diagnostic only)
#    cost   :deterministic(M0) | :expected(M1-3) | :unified_rs(M6)
#
#  凸化依据修改稿: 成本与安全目标统一写成 Robust Satisficing 约束:
#    min κ_c + ω_d κ_d
#    Γ_c=κ_c, Γ_m=k_m κ_d,  a∈{c, safety events}
#    - KDE 平滑铰链 Ψ_p 用核积分求积 -> 二阶/幂锥 (h=0 退化为普通铰链)
#    - φ=KL 的共轭透视 Φ̄(s,μ)=μ(e^{s/μ}-1) -> 指数锥
#    - p=2 的 z^p/y^{p-1} -> 旋转二阶锥
# =====================================================================
module ModelCore

using JuMP, Clarabel, LinearAlgebra
import HiGHS
using ..Cases, ..CaseInterface, ..Bandwidth

export build_ols, ModelResult, calibrate_Z0, calibrate_targets,
       retarget_targets, MODEL_CONFIGS

# ---------- 模型配置注册表 (模块 4 的开关表) ----------
const MODEL_CONFIGS = Dict(
    "M0" => (safety=:none,    cost=:deterministic),
    "M1" => (safety=:cvar,    cost=:expected),     # empirical h=0, theta=0 HMCR/SAA-type ablation; CVaR only when p=1
    "M2" => (safety=:kde,     cost=:expected),     # KDE-HMCR, 无 φ-DR
    "M3" => (safety=:kde_phi, cost=:expected),     # DRCC, 无 RS
    "M3G" => (safety=:kde_phi_grouped, cost=:expected), # diagnostic: M3 risk layer on M6's six category losses
    "M6" => (safety=:target_rs, cost=:unified_rs),
)

function _target_rs_mode(P::Dict)
    mode = lowercase(String(get(P, "target_rs_mode", "legacy")))
    mode in ("legacy", "common_dimensionless") ||
        error("unknown target_rs_mode: $mode")
    return mode
end

_common_dimensionless(P::Dict) = _target_rs_mode(P) == "common_dimensionless"

function _target_cost_scale(cd::CaseData, snap::Snapshot, P::Dict)
    floor_value = Float64(get(P, "target_cost_scale_floor", 1.0))
    floor_value > 0 || error("target_cost_scale_floor must be positive")
    return max(dot(cd.cshed, snap.lf), floor_value)
end

function _common_target_gamma(P::Dict)
    gamma = Float64(get(P, "target_gamma",
                        max(Float64(P["gamma_m"]), 1.0 - Float64(P["eps_m"]))))
    0.0 < gamma < 1.0 || error("common target gamma must lie strictly in (0,1)")
    return gamma
end

function _common_target_k_rule(P::Dict)
    rule = lowercase(String(get(P, "target_k_rule",
                                "reference_target_ratio_no_eps")))
    rule in ("equal_normalized", "reference_target_ratio_no_eps",
             "payoff_range_no_eps") ||
        error("unknown common target_k_rule: $rule")
    return rule
end

"""
Construct target-scale coefficients in the normalized common coordinates.

The original no-epsilon rule is

    k_m = abs(Z0m) / abs(Z0c).

The opt-in payoff-range rule instead takes a fixed 7x7 anchor-evaluation
matrix `Pab`: rows are the seven predeclared anchors and columns are the cost
plus six safety payoffs.  Its numerical coordinate scales are

    s_b = maximum(Pab[:, b]) - minimum(Pab[:, b]),
    k_m = s_(m+1) / s_1.

This is an anchor-set range normalization, not a claim that the row maxima
form an exact Pareto nadir.  No epsilon or numerical floor is inserted in
either rule: a zero or nonfinite scale rejects the state explicitly.
"""
function _payoff_range_scale_coefficients(payoff_matrix)
    values = Float64.(Matrix(payoff_matrix))
    size(values) == (7, 7) || error(
        "payoff-range calibration requires a 7x7 anchor-payoff matrix")
    all(isfinite, values) || error(
        "payoff-range calibration rejects nonfinite anchor payoffs (no epsilon fallback)")
    ideal = vec(minimum(values, dims = 1))
    anchor_maximum = vec(maximum(values, dims = 1))
    ranges = anchor_maximum .- ideal
    all(v -> isfinite(v) && v > 0.0, ranges) || error(
        "payoff-range calibration rejects zero or nonfinite objective ranges (no epsilon fallback)")
    kvals = ranges[2:end] ./ ranges[1]
    all(v -> isfinite(v) && v > 0.0, kvals) || error(
        "payoff-range calibration rejects nonfinite induced k coefficients (no epsilon fallback)")
    return (
        k = kvals,
        rule = "payoff_range_no_eps",
        priorities = ones(6),
        cost_scale_reference = ranges[1],
        safety_scale_reference = copy(ranges[2:end]),
        cost_sigma_proxy = NaN,
        safety_sigma_proxy = fill(NaN, 6),
        cost_resolution = NaN,
        safety_resolution = fill(NaN, 6),
        payoff_ideal = ideal,
        payoff_anchor_maximum = anchor_maximum,
        payoff_ranges = ranges,
        payoff_anchor_count = size(values, 1),
        payoff_scale_source = "predeclared_anchor_payoff_range",
        payoff_zero_range_policy = "reject_no_epsilon",
    )
end

function _common_target_scale_coefficients(cd::CaseData, snap::Snapshot,
                                           profile::FixedBandwidthProfile,
                                           Z0c::Real, Z0m,
                                           P::Dict;
                                           payoff_matrix = nothing)
    rule = _common_target_k_rule(P)
    if rule == "equal_normalized"
        return (
            k = ones(6), rule = rule, priorities = ones(6),
            cost_scale_reference = 1.0,
            safety_scale_reference = ones(6),
            cost_sigma_proxy = NaN,
            safety_sigma_proxy = fill(NaN, 6),
            cost_resolution = NaN,
            safety_resolution = fill(NaN, 6),
        )
    end
    if rule == "payoff_range_no_eps"
        payoff_matrix === nothing && error(
            "payoff_range_no_eps requires an explicit 7x7 anchor-payoff matrix")
        return _payoff_range_scale_coefficients(payoff_matrix)
    end
    denominator = abs(Float64(Z0c))
    isfinite(denominator) && denominator > 0.0 ||
        error("common target ratio k_m=|Z0m|/|Z0c| is undefined because Z0c is zero or nonfinite")
    safety_reference = abs.(Float64.(collect(Z0m)))
    length(safety_reference) == 6 ||
        error("common target calibration must contain six safety references")
    all(isfinite, safety_reference) ||
        error("common safety reference scales must be finite")
    kvals = safety_reference ./ denominator
    return (
        k = kvals, rule = rule, priorities = ones(6),
        cost_scale_reference = denominator,
        safety_scale_reference = safety_reference,
        cost_sigma_proxy = NaN,
        safety_sigma_proxy = fill(NaN, 6),
        cost_resolution = NaN,
        safety_resolution = fill(NaN, 6),
    )
end

function _validate_common_settings(P::Dict)
    String(get(P, "target_tau_rule", "signed_relative")) == "signed_relative" ||
        error("common_dimensionless currently supports only target_tau_rule=signed_relative")
    String(get(P, "target_cost_scale_rule", "forecast_full_shed")) ==
        "forecast_full_shed" ||
        error("common_dimensionless currently supports only forecast_full_shed cost scaling")
    String(get(P, "target_bandwidth_pilot", "common_anchor")) == "common_anchor" ||
        error("common_dimensionless currently supports only target_bandwidth_pilot=common_anchor")
    Float64(get(P, "target_kde_h_min_rel", 1e-6)) > 0 ||
        error("target_kde_h_min_rel must be strictly positive")
    Float64(get(P, "target_kde_multiplier", 1.0)) > 0 ||
        error("target_kde_multiplier must be strictly positive")
    _common_target_k_rule(P)
    return nothing
end

function _validate_common_profile(profile, cd::CaseData, snap::Snapshot, P::Dict)
    _validate_common_settings(P)
    profile isa FixedBandwidthProfile ||
        error("common_dimensionless target RS requires a FixedBandwidthProfile")
    length(profile.target_h) == 6 ||
        error("common target bandwidth profile must contain six safety bandwidths")
    profile.cost_h > 0 || error("common target cost bandwidth must be strictly positive")
    all(>(0.0), profile.target_h) ||
        error("all common target safety bandwidths must be strictly positive")
    expected_multiplier = Float64(get(P, "target_kde_multiplier", 1.0))
    isapprox(profile.multiplier, expected_multiplier; rtol = 1e-12, atol = 1e-12) ||
        error("common target bandwidth multiplier does not match the configured value")
    hfloor = Float64(get(P, "target_kde_h_min_rel", 1e-6))
    cost_scale = _target_cost_scale(cd, snap, P)
    minimum(profile.target_h) + 1e-14 >= hfloor ||
        error("common target safety bandwidth is below the configured positive floor")
    profile.cost_h / cost_scale + 1e-14 >= hfloor ||
        error("common target cost bandwidth is below the configured relative floor")
    return nothing
end

struct ModelResult
    model
    status
    x::NamedTuple                # g, rU, rD, snom, cW, β, ρG, αS
    κc::Float64
    κd::Float64
    shed_cost::Float64           # 期望切负荷成本 Σπ_k C^(k)
    gen_cost::Float64            # cgen' g
    solve_time::Float64          # optimize! 墙钟时间 (s)
    build_time::Float64          # 建模(构造 JuMP 模型)时间 (s)
    solver_time::Float64         # 求解器内部报告时间 (s)
    iters::Int                   # 求解器迭代数 (锥内点)
    obj::Float64                 # 目标函数最优值 (供精确度对比)
    n_var::Int                   # 变量数
    n_con::Int                   # 约束数 (含锥)
    target_meta::Any             # M6 的 Z0/τ/k 标定信息; 其他模型为 nothing
end

# ---------- KDE 平滑铰链: 返回 z 变量, z ≥ Ψ_p(expr - α, h) ----------
# 高斯核积分用 (ζ_q, ω_q) 求积近似: Ψ_p^p ≈ Σ_q ω_q [expr-α - hζ_q]_+^p
# h 可为常数(固定带宽)或 JuMP 变量(规则带宽, Ψ_p 对 (x,h) 联合凸); h*ζ_q 仍仿射。
function kde_hinge!(m, expr, α, h, p::Int, ζ::Vector{Float64}, ω::Vector{Float64})
    z = @variable(m, lower_bound = 0.0)
    Q = length(ζ)
    t = @variable(m, [1:Q], lower_bound = 0.0)
    for q in 1:Q
        @constraint(m, t[q] >= expr - α - h * ζ[q])
    end
    if p == 1
        @constraint(m, z >= sum(ω[q] * t[q] for q in 1:Q))
    elseif p == 2
        # z ≥ sqrt(Σ ω_q t_q^2)
        @constraint(m, [z; [sqrt(ω[q]) * t[q] for q in 1:Q]] in SecondOrderCone())
    else
        error("p=$p 暂不支持 (需幂锥)")
    end
    return z
end

# 标准正态核求积节点。Gauss-Hermite 由正态正交多项式的 Jacobi 矩阵生成；
# legacy_grid 仅用于复现早期 [-2,2] 离散核实验。
function _normal_nodes(Q::Int, rule::AbstractString = "gauss_hermite";
                       weight_floor::Real = 0.0)
    Q >= 1 || error("kde_quad_nodes must be positive")
    0 <= weight_floor < 1 || error("kde_quad_weight_floor must be in [0,1)")
    quad_rule = lowercase(strip(rule))
    if quad_rule in ("gauss_hermite", "gh", "normal")
        if Q == 1
            return ([0.0], [1.0])
        end
        jacobi = SymTridiagonal(zeros(Q), sqrt.(Float64.(1:Q-1)))
        eig = eigen(jacobi)
        nodes = collect(eig.values)
        weights = vec(eig.vectors[1, :] .^ 2)
        weights ./= sum(weights)
        if weight_floor > 0
            keep = weights .>= weight_floor
            any(keep) || error("quadrature weight floor removes every node")
            nodes = nodes[keep]
            weights = weights[keep]
            weights ./= sum(weights)
        end
        return (nodes, weights)
    elseif quad_rule in ("legacy_grid", "legacy")
        nodes = range(-2.0, 2.0; length = Q) |> collect
        weights = exp.(-nodes .^ 2 ./ 2)
        weights ./= sum(weights)
        return (nodes, weights)
    end
    error("unknown kde_quad_rule: $rule")
end

# Exact closed perspective for the modified chi-square generator
# phi(t) = (t - 1)^2 on t >= 0. Its conjugate is piecewise:
# phi*(s) = ((s + 2)_+)^2 / 4 - 1. The positive part is essential because
# KDE mixture weights remain nonnegative probabilities.
function _add_modified_chi2_perspective!(m, s, mu)
    t = @variable(m, lower_bound = 0.0)
    g = @variable(m, lower_bound = 0.0)
    @constraint(m, t >= s + 2 * mu)
    @constraint(m, [2 * mu, g, t] in RotatedSecondOrderCone())
    return g - mu
end

# ---------- 安全侧风险块 (定理2 / θ=0 直接 HMCR) ----------
# qk: 长度 K 的样本违约仿射表达式向量; 施加 HMCR_{γ̄,p}(q) ≤ 0 (可含 φ-DR)
# bw_mode: "off"(h=0,样本HMCR/CVaR) | "fixed"(固定 h) | "rule"(规则带宽 h=c‖M·q‖)
function add_safety!(m, qk::Vector, π::Vector{Float64}, p::Int, γ̄::Float64,
                     h::Float64, θ::Float64, ζ, ω, εy::Float64;
                     phi::String = "chi2", bw_mode::String = "off",
                     rule_scale::Float64 = 1.0)
    K = length(qk)
    α = @variable(m)
    # --- 带宽 ---
    if bw_mode == "rule"
        # Silverman 规则带宽(决策相关, 加权): hb ≥ 1.06 K^{-1/5} · sqrt(Σπ_k(q_k-q̄)²)
        qbar = sum(π[k]*qk[k] for k in 1:K)
        hb = @variable(m, lower_bound = 0.0)
        rule_scale > 0 || error("rule_scale must be positive in rule bandwidth mode")
        cprime = rule_scale * 1.06 * K^(-1/5)
        @constraint(m, [hb / cprime; [sqrt(π[k])*(qk[k] - qbar) for k in 1:K]]
                       in SecondOrderCone())
        z = [kde_hinge!(m, qk[k], α, hb, p, ζ, ω) for k in 1:K]
    else
        z = [kde_hinge!(m, qk[k], α, h, p, ζ, ω) for k in 1:K]
    end
    if θ <= 0
        # 直接 HMCR: α + (Σ π_k z_k^p)^{1/p}/(1-γ̄) ≤ 0
        ζm = @variable(m, lower_bound = 0.0)
        if p == 1
            @constraint(m, ζm >= sum(π[k] * z[k] for k in 1:K))
        else
            @constraint(m, [ζm; [sqrt(π[k]) * z[k] for k in 1:K]] in SecondOrderCone())
        end
        @constraint(m, α + ζm / (1 - γ̄) <= 0)
        return nothing
    end
    # φ-DR 对偶 (定理2)
    y = @variable(m, lower_bound = εy)
    η = @variable(m)
    μ = @variable(m, lower_bound = 0.0)
    v = @variable(m, [1:K], lower_bound = 0.0)
    for k in 1:K
        if p == 1
            @constraint(m, v[k] >= z[k] / (1 - γ̄))
        else
            @constraint(m, [(1 - γ̄) * y, v[k], z[k]] in RotatedSecondOrderCone())
        end
    end
    coef = p == 1 ? 0.0 : (p - 1) / (p * (1 - γ̄))
    if phi == "KL"
        ψ = @variable(m, [1:K], lower_bound = 0.0)        # ψ_k ≥ μ e^{(v_k-η)/μ}
        for k in 1:K
            @constraint(m, [v[k] - η, μ, ψ[k]] in MOI.ExponentialCone())
        end
        @constraint(m, α + coef*y + η + θ*μ + sum(π[k]*ψ[k] for k in 1:K) - μ <= 0)
    elseif phi == "chi2"
        phik = [_add_modified_chi2_perspective!(m, v[k] - η, μ) for k in 1:K]
        @constraint(m, α + coef*y + η + θ*μ +
                       sum(π[k]*phik[k] for k in 1:K) <= 0)
    else
        error("未知 φ 类型 $phi")
    end
    return nothing
end


# ---------- 成本侧 Robust Satisficing 块 (原模型定理1, eq.cost_dual1-4: KDE-HMCR_{γc} 代理) ----------
# sup_{λ∈Δ_T}{ α_c + (1-γc)^{-1}(Σλ_i Ψ_p(C_i-α_c,h_c)^p)^{1/p} − κ_c d_φ(λ,p̂) } ≤ τ
# 有限维凸充分条件(定理1, 与安全侧定理2 同构, 仅: 散度对偶用主变量 κc 代 μ、无 θμ 项、RHS=τ):
#   α_c + (p-1)/(p(1-γc))y_c + η_c + Σp̂_i Φ̄(v_i-η_c, κc) ≤ τ      (cost_dual1)
#   z_i ≥ Ψ_p(C_i-α_c, h_c);  z_i^p/(p(1-γc)y_c^{p-1}) ≤ v_i;  y_c≥εy  (cost_dual2-4)
# γc=0 时 HMCR_{0}=期望, 退化为期望-RS(旧 P4 口径)。bw_mode/h_c 给成本侧 KDE 带宽。
# hc: 成本侧 KDE 带宽, 可为标量 0(无平滑)/固定值, 或 JuMP 变量(规则带宽, 由 build_ols 按
#     PDF §6.1 Ω 基公式 h_c=1.06K^{-1/5}·std_π(Ω)·c_shed'αS 在外部装配后传入)。
function add_rs_cost!(m, Ck::Vector, π::Vector{Float64}, τcost, κc, γc::Float64,
                      hc, p::Int, ζ, ω, εy::Float64; phi::String = "chi2")
    K = length(Ck)
    # γc=0 且无带宽(hc≡0)快路径: HMCR_{0}=期望, 直接期望-RS 对偶, 避免退化 αc/yc/v 层致病态。
    if γc <= 1e-9 && (hc isa Real) && hc == 0.0
        η = @variable(m)
        if phi == "chi2"
            phik = [_add_modified_chi2_perspective!(m, Ck[k] - η, κc) for k in 1:K]
            @constraint(m, η + sum(π[k]*phik[k] for k in 1:K) <= τcost)
        else  # KL
            ψ = @variable(m, [1:K], lower_bound = 0.0)
            for k in 1:K
                @constraint(m, [Ck[k] - η, κc, ψ[k]] in MOI.ExponentialCone())
            end
            @constraint(m, η + sum(π[k]*ψ[k] for k in 1:K) - κc <= τcost)
        end
        return nothing
    end
    αc = @variable(m)
    z = [kde_hinge!(m, Ck[k], αc, hc, p, ζ, ω) for k in 1:K]   # hc 标量或变量均可
    yc = @variable(m, lower_bound = εy)
    ηc = @variable(m)
    v = @variable(m, [1:K], lower_bound = 0.0)
    for k in 1:K
        if p == 1
            @constraint(m, v[k] >= z[k] / (1 - γc))
        else
            @constraint(m, [(1 - γc) * yc, v[k], z[k]] in RotatedSecondOrderCone())
        end
    end
    coef = p == 1 ? 0.0 : (p - 1) / (p * (1 - γc))
    if phi == "chi2"
        phik = [_add_modified_chi2_perspective!(m, v[k] - ηc, κc) for k in 1:K]
        @constraint(m, αc + coef*yc + ηc +
                       sum(π[k]*phik[k] for k in 1:K) <= τcost)
    elseif phi == "KL"
        ψ = @variable(m, [1:K], lower_bound = 0.0)
        for k in 1:K
            @constraint(m, [v[k] - ηc, κc, ψ[k]] in MOI.ExponentialCone())
        end
        @constraint(m, αc + coef*yc + ηc + sum(π[k]*ψ[k] for k in 1:K) - κc <= τcost)
    else
        error("未知 φ 类型 $phi")
    end
    return nothing
end

# ---------- 参考分布 KDE-HMCR 风险代理 ----------
# 返回一个仿射表达式 R, 并添加 R ≥ HMCR_{γ,p}(L) 的锥约束。用于七个 Z0 参考目标定标。
function add_hmcr_risk!(m, Lk::Vector, π::Vector{Float64}, p::Int, γ::Float64,
                        h::Float64, ζ, ω; bw_mode::String = "off",
                        rule_scale::Float64 = 1.0)
    K = length(Lk)
    α = @variable(m)
    if bw_mode == "rule"
        Lbar = sum(π[k]*Lk[k] for k in 1:K)
        hb = @variable(m, lower_bound = 0.0)
        rule_scale > 0 || error("rule_scale must be positive in rule bandwidth mode")
        cprime = rule_scale * 1.06 * K^(-1/5)
        @constraint(m, [hb / cprime; [sqrt(π[k])*(Lk[k] - Lbar) for k in 1:K]]
                       in SecondOrderCone())
        z = [kde_hinge!(m, Lk[k], α, hb, p, ζ, ω) for k in 1:K]
    else
        z = [kde_hinge!(m, Lk[k], α, h, p, ζ, ω) for k in 1:K]
    end
    tail = @variable(m, lower_bound = 0.0)
    if p == 1
        @constraint(m, tail >= sum(π[k] * z[k] for k in 1:K))
    else
        @constraint(m, [tail; [sqrt(π[k]) * z[k] for k in 1:K]] in SecondOrderCone())
    end
    return α + tail / (1 - γ)
end

# ---------- 统一 Robust Satisficing 块 ----------
# 对任意目标 a 加入
#   sup_λ { HMCR_a(L;λ) - τ_a - Γ_a D_φ(λ,p̂) } ≤ 0
# 其中 Γ_a 为 κc 或 k_m κd。和论文最终模型 eq.final_dual1--3 同构。
function add_target_rs!(m, Lk::Vector, π::Vector{Float64}, τ, Γ, γ::Float64,
                        h, p::Int, ζ, ω, εy::Float64; phi::String = "chi2",
                        bw_mode::String = "off", rule_scale::Float64 = 1.0)
    K = length(Lk)
    if bw_mode != "rule" && γ <= 1e-9 && (h isa Real) && h == 0.0
        η = @variable(m)
        if phi == "chi2"
            phik = [_add_modified_chi2_perspective!(m, Lk[k] - η, Γ) for k in 1:K]
            @constraint(m, η + sum(π[k]*phik[k] for k in 1:K) <= τ)
        elseif phi == "KL"
            ψ = @variable(m, [1:K], lower_bound = 0.0)
            for k in 1:K
                @constraint(m, [Lk[k] - η, Γ, ψ[k]] in MOI.ExponentialCone())
            end
            @constraint(m, η + sum(π[k]*ψ[k] for k in 1:K) - Γ <= τ)
        else
            error("未知 φ 类型 $phi")
        end
        return nothing
    end

    α = @variable(m)
    if bw_mode == "rule"
        Lbar = sum(π[k]*Lk[k] for k in 1:K)
        hb = @variable(m, lower_bound = 0.0)
        rule_scale > 0 || error("rule_scale must be positive in rule bandwidth mode")
        cprime = rule_scale * 1.06 * K^(-1/5)
        @constraint(m, [hb / cprime; [sqrt(π[k])*(Lk[k] - Lbar) for k in 1:K]]
                       in SecondOrderCone())
        z = [kde_hinge!(m, Lk[k], α, hb, p, ζ, ω) for k in 1:K]
    else
        z = [kde_hinge!(m, Lk[k], α, h, p, ζ, ω) for k in 1:K]
    end
    y = @variable(m, lower_bound = εy)
    η = @variable(m)
    v = @variable(m, [1:K], lower_bound = 0.0)
    for k in 1:K
        if p == 1
            @constraint(m, v[k] >= z[k] / (1 - γ))
        else
            @constraint(m, [(1 - γ) * y, v[k], z[k]] in RotatedSecondOrderCone())
        end
    end
    coef = p == 1 ? 0.0 : (p - 1) / (p * (1 - γ))
    if phi == "chi2"
        phik = [_add_modified_chi2_perspective!(m, v[k] - η, Γ) for k in 1:K]
        @constraint(m, α - τ + coef*y + η +
                       sum(π[k]*phik[k] for k in 1:K) <= 0)
    elseif phi == "KL"
        ψ = @variable(m, [1:K], lower_bound = 0.0)
        for k in 1:K
            @constraint(m, [v[k] - η, Γ, ψ[k]] in MOI.ExponentialCone())
        end
        @constraint(m, α - τ + coef*y + η + sum(π[k]*ψ[k] for k in 1:K) - Γ <= 0)
    else
        error("未知 φ 类型 $phi")
    end
    return nothing
end

function _category_epigraph!(m, parts)
    K = length(parts)
    out = Vector{Any}(undef, K)
    for k in 1:K
        if isempty(parts[k])
            out[k] = 0.0
        else
            q = @variable(m)
            for expr in parts[k]
                @constraint(m, q >= expr)
            end
            out[k] = q
        end
    end
    return out
end

# ---------- 主 builder ----------
"""
    build_ols(cd, snap, scen, params; safety, cost, τcost=0.0, target_meta=nothing)

返回求解后的 ModelResult。params 为 cfg["model"] 字典。
"""
function build_ols(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict;
                   safety::Symbol, cost::Symbol, τcost::Float64 = 0.0,
                   target_meta = nothing, bandwidth_profile = nothing)
    tb = time()                              # 建模计时起点
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "max_iter", 2000)
    set_optimizer_attribute(m, "equilibrate_enable", true)
    set_optimizer_attribute(m, "tol_gap_abs", 1e-8)
    set_optimizer_attribute(m, "tol_gap_rel", 1e-8)
    set_optimizer_attribute(m, "tol_feas", 1e-8)
    set_optimizer_attribute(m, "reduced_tol_gap_abs", 5e-5)
    set_optimizer_attribute(m, "reduced_tol_gap_rel", 1e-5)
    set_optimizer_attribute(m, "reduced_tol_feas", 1e-5)
    if haskey(P, "solver_direct_solve_method")
        set_optimizer_attribute(m, "direct_solve_method",
                                Symbol(P["solver_direct_solve_method"]))
    end
    if haskey(P, "solver_equilibrate_max_iter")
        set_optimizer_attribute(m, "equilibrate_max_iter",
                                Int(P["solver_equilibrate_max_iter"]))
    end
    if haskey(P, "solver_iterative_refinement_max_iter")
        set_optimizer_attribute(m, "iterative_refinement_max_iter",
                                Int(P["solver_iterative_refinement_max_iter"]))
    end
    if haskey(P, "solver_dynamic_regularization_delta")
        set_optimizer_attribute(m, "dynamic_regularization_delta",
                                Float64(P["solver_dynamic_regularization_delta"]))
    end
    p   = P["p"]; εy = P["eps_y"]
    γm  = P["gamma_m"]; εm = P["eps_m"]; γ̄ = max(γm, 1 - εm)
    γc  = float(P["gamma_c"]); θ = P["theta_m"]   # γc: legacy 成本侧 HMCR 置信
    hm  = float(P["kde_h_m"]); hc = float(get(P, "kde_h_c", 0.0)); Q = P["kde_quad_nodes"]
    quad_rule = String(get(P, "kde_quad_rule", "gauss_hermite"))
    quad_weight_floor = float(get(P, "kde_quad_weight_floor", 0.0))
    hscale_m = float(get(P, "kde_rule_scale_m", 1.0))
    hscale_c = float(get(P, "kde_rule_scale_c", 1.0))
    # 安全侧带宽模式: cvar -> off(无平滑); kde/kde_phi -> config kde_mode ("rule"|"fixed"|"off")
    kde_mode = get(P, "kde_mode", "rule")
    bw = safety == :cvar ? "off" : kde_mode
    # 成本侧带宽模式: config cost_kde_mode ("off"|"fixed"|"rule"), 默认 off(样本 HMCR, hc=0)
    bwc = get(P, "cost_kde_mode", "off")
    bandwidth_profile === nothing || bandwidth_profile isa FixedBandwidthProfile ||
        error("bandwidth_profile must be a FixedBandwidthProfile")
    common_target = cost == :unified_rs && _common_dimensionless(P)
    common_target && _validate_common_profile(bandwidth_profile, cd, snap, P)
    profile_active = bandwidth_profile !== nothing && safety != :cvar
    bw_safety = common_target ? "fixed" : (profile_active ? "fixed" : bw)
    (ζm, ωm) = bw_safety == "off" ? ([0.0], [1.0]) :
        _normal_nodes(Q, quad_rule; weight_floor = quad_weight_floor)
    θeff = safety in (:kde_phi, :kde_phi_grouped) ? float(θ) : 0.0

    nG, nD, nW, nE, K = cd.nG, cd.nD, cd.nW, cd.nE, scen.K

    # --- 决策变量 x ---
    @variable(m, g[1:nG])
    @variable(m, rU[1:nG] >= 0)
    @variable(m, rD[1:nG] >= 0)
    @variable(m, snom[1:nD] >= 0)
    @variable(m, cW[1:nW] >= 0)

    # --- 物理约束 Xphys ---
    @constraint(m, g .+ rU .<= cd.pmax)
    @constraint(m, g .- rD .>= cd.pmin)
    @constraint(m, snom .<= snap.lf)
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    @constraint(m, cW .<= curtail_cap)
    for j in 1:nG
        if !cd.agc_mask[j]
            @constraint(m, rU[j] == 0); @constraint(m, rD[j] == 0)
        end
    end
    # 发电成本平局项: OLS 主目标只最小化(期望)切负荷成本, 但切负荷=0(及应力退化)时
    # 调度 g(及 cW)高度欠定。以 ε_gen·c_gen'g 作严格字典序次目标钉住"经济调度"(真实
    # 系统会用的运行点); ε_gen=1e-6≪VOLL=1000, 切负荷以 ~10^6:1 碾压, 故发电成本不可能
    # 影响切负荷决策(需 10^6 MW 重调度才抵 1 MW 切负荷), 是纯平局项。弃风需替代发电故被
    # c_gen'g 隐式惩罚, 无需单独弃风项。
    εgen = get(P, "eps_gen_tiebreak", 1e-6)
    # 给定备用资源总量上限(稀缺备用场景; frac 为占机组总容量比例, 默认 Inf=不约束)
    rucapf = get(P, "ru_cap_frac", Inf); rdcapf = get(P, "rd_cap_frac", Inf)
    isfinite(rucapf) && @constraint(m, sum(rU) <= rucapf * sum(cd.pmax))
    isfinite(rdcapf) && @constraint(m, sum(rD) <= rdcapf * sum(cd.pmax))
    # 功率平衡
    @constraint(m, sum(g) == sum(snap.lf) - sum(snap.wf) + sum(cW) - sum(snom))
    # 预测点线路潮流  f0 = MG g + MW(wf - cW) - MD(lf - snom)
    f0 = [ sum(cd.MG[ℓ,j]*g[j] for j in 1:nG) +
           sum(cd.MW[ℓ,r]*(snap.wf[r]-cW[r]) for r in 1:nW) -
           sum(cd.MD[ℓ,d]*(snap.lf[d]-snom[d]) for d in 1:nD)  for ℓ in 1:nE ]
    for ℓ in 1:nE
        isfinite(cd.F_max[ℓ]) || continue
        @constraint(m, f0[ℓ] <= cd.F_max[ℓ]); @constraint(m, -f0[ℓ] <= cd.F_max[ℓ])
    end

    if safety == :none
        # M0 确定性: 不为不确定性预留备用(rU=rD=0), 不设机会约束; 审计时按默认 AGC
        # 响应(ρG=1, αS=0)评估 -> 体现"忽略不确定性"的代价(备用越界)。
        @constraint(m, rU .== 0); @constraint(m, rD .== 0)
        # 主目标 = 切负荷成本 (OLS); + ε_gen 发电成本平局项钉住经济调度 g。
        @objective(m, Min, sum(cd.cshed[d]*snom[d] for d in 1:nD) +
                            εgen * sum(cd.cgen[j]*g[j] for j in 1:nG))
        return _finalize(m, cd, scen, g, rU, rD, snom, cW, nothing, nothing, 0.0;
                         default_recourse = true, build_time = time() - tb)
    end

    # --- recourse 变量与闭合 ---
    # 代码数据管线的 Ω 为内部净注入误差(= - 论文净负荷误差)；
    # 因此 g - Ωβ, s - ΩαS 与论文 g + Ω_paperβ, s + Ω_paperαS 等价。
    @variable(m, β[1:nG] >= 0)
    @variable(m, αS[1:nD] >= 0)
    @constraint(m, sum(β) + sum(αS) == 1)
    for j in 1:nG
        cd.agc_mask[j] || @constraint(m, β[j] == 0)
    end

    # --- 样本违约函数 q_m^{(k)} (均为 x 的仿射) ---
    # 每个事件按正常数 sc 归一到 O(1): HMCR(q/sc)≤0 ⟺ HMCR(q)≤0, φ-球不受影响,
    # 仅改善指数锥/旋转锥的数值条件。
    MGβ = [ sum(cd.MG[ℓ,j]*β[j] for j in 1:nG) for ℓ in 1:nE ]
    safe_events = Vector{Vector{Any}}()
    q_up_parts    = [Any[] for _ in 1:K]
    q_down_parts  = [Any[] for _ in 1:K]
    q_lpos_parts  = [Any[] for _ in 1:K]
    q_lneg_parts  = [Any[] for _ in 1:K]
    q_slow_parts  = [Any[] for _ in 1:K]
    q_shigh_parts = [Any[] for _ in 1:K]

    # 备用 U/D  (j ∈ AGC)
    for j in 1:nG
        cd.agc_mask[j] || continue
        sc = max(cd.pmax[j], 1.0)
        qU = [ (-scen.Ω[k]*β[j] - rU[j])/sc for k in 1:K]
        qD = [ ( scen.Ω[k]*β[j] - rD[j])/sc for k in 1:K]
        push!(safe_events, qU)
        push!(safe_events, qD)
        for k in 1:K
            push!(q_up_parts[k], qU[k])
            push!(q_down_parts[k], qD[k])
        end
    end
    # 线路 L± : 内部净注入口径 f̃_ℓ^{(k)} = f0_ℓ + (PTDF δ^{(k)})_ℓ - Ω^{(k)}(MGβ_ℓ + MD_ℓ·αS)
    PTDFδ = scen.δ * cd.PTDF'                 # K × nE
    for ℓ in 1:nE
        isfinite(cd.F_max[ℓ]) || continue
        sc = max(cd.F_max[ℓ], 1.0)
        ftil = [ f0[ℓ] + PTDFδ[k,ℓ] - scen.Ω[k]*(MGβ[ℓ] +
                  sum(cd.MD[ℓ,d]*αS[d] for d in 1:nD))  for k in 1:K]
        qP = [ ( ftil[k] - cd.F_max[ℓ])/sc for k in 1:K]
        qN = [ (-ftil[k] - cd.F_max[ℓ])/sc for k in 1:K]
        push!(safe_events, qP)
        push!(safe_events, qN)
        for k in 1:K
            push!(q_lpos_parts[k], qP[k])
            push!(q_lneg_parts[k], qN[k])
        end
    end
    # 切负荷 S± : 内部净注入口径 s_d^{(k)} = snom_d - Ω^{(k)} αS_d；等价于论文净负荷口径的加号。
    for d in 1:nD
        sc = max(snap.lf[d], 1.0)
        sdk = [ snom[d] - scen.Ω[k]*αS[d]  for k in 1:K]
        qL = [ (-sdk[k])/sc                               for k in 1:K]  # S-
        qH = [ ( sdk[k] - (snap.lf[d] + scen.eL[k,d]))/sc for k in 1:K]  # S+
        push!(safe_events, qL)
        push!(safe_events, qH)
        for k in 1:K
            push!(q_slow_parts[k], qL[k])
            push!(q_shigh_parts[k], qH[k])
        end
    end

    target_events = [_category_epigraph!(m, q_up_parts),
                     _category_epigraph!(m, q_down_parts),
                     _category_epigraph!(m, q_lpos_parts),
                     _category_epigraph!(m, q_lneg_parts),
                     _category_epigraph!(m, q_slow_parts),
                     _category_epigraph!(m, q_shigh_parts)]

    φ = get(P, "phi_type", "chi2")
    if safety == :kde_phi_grouped
        profile_active && length(bandwidth_profile.target_h) != length(target_events) &&
            error("target bandwidth count does not match grouped safety targets")
        for (mi, qk) in enumerate(target_events)
            h_use = profile_active ? bandwidth_profile.target_h[mi] : hm
            add_safety!(m, qk, scen.π, p, γ̄, h_use, θeff, ζm, ωm, εy;
                        phi = φ, bw_mode = bw_safety, rule_scale = hscale_m)
        end
    elseif safety != :target_rs
        profile_active && length(bandwidth_profile.event_h) != length(safe_events) &&
            error("event bandwidth count does not match elementary safety events")
        for (ei, qk) in enumerate(safe_events)
            h_use = profile_active ? bandwidth_profile.event_h[ei] : hm
            add_safety!(m, qk, scen.π, p, γ̄, h_use, θeff, ζm, ωm, εy;
                        phi = φ, bw_mode = bw_safety, rule_scale = hscale_m)
        end
    end

    # --- 成本: 期望切负荷成本 C^{(k)} = c_shed' χ^{(k)}, χ ≥ s^{(k)}, χ≥0 ---
    @variable(m, χ[1:K, 1:nD] >= 0)
    for k in 1:K, d in 1:nD
        @constraint(m, χ[k,d] >= snom[d] - scen.Ω[k]*αS[d])
    end
    Ck = [ sum(cd.cshed[d]*χ[k,d] for d in 1:nD) for k in 1:K]      # 切负荷成本/样本
    Ck_target = [ sum(cd.cshed[d]*(snom[d] - scen.Ω[k]*αS[d]) for d in 1:nD) for k in 1:K]

    if cost == :expected
        # 主目标 = 期望切负荷成本 (OLS); + ε_gen 发电成本平局项钉住经济调度 g。
        @objective(m, Min, sum(scen.π[k]*Ck[k] for k in 1:K) +
                           εgen * sum(cd.cgen[j]*g[j] for j in 1:nG))
        return _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, 0.0;
                         build_time = time() - tb)
    elseif cost == :rs
        # 成本侧归一: 用 cscale 把 C、τ、h_c 缩放到 O(1), 改善锥条件 (κc 报告时乘回)。
        cscale = max(τcost, get(P, "voll", 1000.0), 1.0)
        @variable(m, κc >= 0)
        # 成本侧 KDE 带宽 hc(归一到 Ck/cscale 坐标):
        #   off -> 0(样本 HMCR); fixed -> kde_h_c/cscale; rule -> PDF §6.1 Ω 基:
        #   h_c = 1.06 K^{-1/5}·std_π(Ω)·(c_shed'αS), 决策相关(线性于 αS), 用变量装配。
        if bwc == "rule"
            Ωbar = sum(scen.π[k]*scen.Ω[k] for k in 1:K)
            cΩ = hscale_c * 1.06 * K^(-1/5) *
                 sqrt(max(sum(scen.π[k]*(scen.Ω[k]-Ωbar)^2 for k in 1:K), 0.0))
            hcv = @variable(m, lower_bound = 0.0)
            @constraint(m, hcv >= cΩ * sum(cd.cshed[d]*αS[d] for d in 1:nD) / cscale)
            hcc = hcv
        elseif bwc == "fixed"
            hcc = hc / cscale
        else
            hcc = 0.0
        end
        # 定理1 KDE-HMCR_{γc} 成本 RS; γc=0 且 hcc=0 走期望-RS 快路径。
        add_rs_cost!(m, Ck ./ cscale, scen.π, τcost/cscale, κc, γc, hcc, p, ζm, ωm, εy;
                     phi = get(P, "phi_type", "chi2"))
        # 次目标(字典序, ≪ κc=O(100s), 不扰动 RS): ε_gen 发电成本平局项钉住经济调度 g,
        # 与 M0--M3 同口径(弃风经 c_gen'g 隐式惩罚)。κc 仍为唯一主目标。
        @objective(m, Min, κc + εgen * sum(cd.cgen[j]*g[j] for j in 1:nG))
        return _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, NaN;
                         κvar = κc, κscale = cscale, build_time = time() - tb)
    elseif cost == :unified_rs
        target_meta === nothing && error("M6 unified_rs needs target_meta = calibrate_targets(...)")
        @variable(m, κc >= 0)
        @variable(m, κd >= 0)

        if common_target
            cost_scale = _target_cost_scale(cd, snap, P)
            if hasproperty(target_meta, :cost_scale)
                meta_scale = Float64(target_meta.cost_scale)
                isapprox(meta_scale, cost_scale; rtol = 1e-10, atol = 1e-10) ||
                    error("target metadata cost scale does not match the current snapshot")
            end
            gamma_target = _common_target_gamma(P)
            cost_h_normalized = bandwidth_profile.cost_h / cost_scale
            cost_h_normalized > 0 || error("normalized cost bandwidth must be positive")

            # The common mode is solved in normalized coordinates.  The
            # target-specific k_m coefficients retain the frozen reference
            # scale (and optional ex-ante priority) of each safety target; they
            # must not be silently replaced by one merely because the residuals
            # are dimensionless.
            add_target_rs!(m, Ck ./ cost_scale, scen.π, target_meta.τc,
                           κc, gamma_target, cost_h_normalized,
                           p, ζm, ωm, εy; phi = φ, bw_mode = "fixed")
            for mi in 1:6
                add_target_rs!(m, target_events[mi], scen.π, target_meta.τm[mi],
                               target_meta.k[mi] * κd, gamma_target,
                               bandwidth_profile.target_h[mi],
                               p, ζm, ωm, εy; phi = φ, bw_mode = "fixed")
            end
            ωd = Float64(get(P, "omega_d", 1.0))
            εshed = Float64(get(P, "target_expected_cost_tiebreak", 0.0))
            εshed >= 0 || error("target_expected_cost_tiebreak must be nonnegative")
            lex_enabled = Bool(get(P, "target_lexicographic_tiebreak", false))
            lex_enabled && εshed > 0 && error(
                "fixed expected-cost regularization and lexicographic tiebreak are mutually exclusive")
            primary_obj = κc + ωd * κd
            expected_cost_normalized =
                sum(scen.π[k] * Ck[k] / cost_scale for k in 1:K)
            generation_tiebreak = (Float64(εgen) / cost_scale) *
                                  sum(cd.cgen[j] * g[j] for j in 1:nG)
            if lex_enabled
                @objective(m, Min, primary_obj)
                return _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, NaN;
                    κvar = κc, κdvar = κd, build_time = time() - tb,
                    target_meta = target_meta,
                    lex_primary = primary_obj,
                    lex_secondary = expected_cost_normalized + generation_tiebreak,
                    lex_abs_tol = Float64(get(P, "target_lexicographic_abs_tol", 1e-7)),
                    lex_rel_tol = Float64(get(P, "target_lexicographic_rel_tol", 0.0)),
                    lex_gap_multiplier = Float64(get(P, "target_lexicographic_gap_multiplier", 10.0)),
                    lex_activation_tol = Float64(get(P, "target_lexicographic_activation_tol", 1e-2)))
            end
            @objective(m, Min, primary_obj +
                               εshed * expected_cost_normalized + generation_tiebreak)
            return _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, NaN;
                             κvar = κc, κdvar = κd, build_time = time() - tb,
                             target_meta = target_meta)
        end

        if bandwidth_profile !== nothing && bwc != "off"
            hcc = bandwidth_profile.cost_h
            bwc_use = "fixed"
        elseif bwc == "rule"
            Ωbar = sum(scen.π[k]*scen.Ω[k] for k in 1:K)
            cΩ = hscale_c * 1.06 * K^(-1/5) *
                 sqrt(max(sum(scen.π[k]*(scen.Ω[k]-Ωbar)^2 for k in 1:K), 0.0))
            hcv = @variable(m, lower_bound = 0.0)
            @constraint(m, hcv >= cΩ * sum(cd.cshed[d]*αS[d] for d in 1:nD))
            hcc = hcv
            bwc_use = "fixed"
        elseif bwc == "fixed"
            hcc = hc
            bwc_use = "fixed"
        else
            hcc = 0.0
            bwc_use = "off"
        end

        add_target_rs!(m, Ck_target, scen.π, target_meta.τc, κc, γc, hcc, p, ζm, ωm, εy;
                       phi = φ, bw_mode = bwc_use)
        profile_active && length(bandwidth_profile.target_h) != 6 &&
            error("target bandwidth count does not match M6 safety targets")
        for mi in 1:6
            h_use = profile_active ? bandwidth_profile.target_h[mi] : hm
            add_target_rs!(m, target_events[mi], scen.π, target_meta.τm[mi],
                           target_meta.k[mi] * κd, γ̄, h_use, p, ζm, ωm, εy;
                           phi = φ, bw_mode = bw_safety, rule_scale = hscale_m)
        end
        ωd = Float64(get(P, "omega_d", 1.0))
        @objective(m, Min, κc + ωd * κd + εgen * sum(cd.cgen[j]*g[j] for j in 1:nG))
        return _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, NaN;
                         κvar = κc, κdvar = κd, build_time = time() - tb,
                         target_meta = target_meta)
    else
        error("未知 cost 模式 $cost")
    end
end

function _finalize(m, cd, scen, g, rU, rD, snom, cW, β, αS, _κ;
                   κvar = nothing, κscale = 1.0, κdvar = nothing,
                   default_recourse = false, build_time = NaN,
                   target_meta = nothing,
                   lex_primary = nothing, lex_secondary = nothing,
                   lex_abs_tol::Real = 0.0, lex_rel_tol::Real = 0.0,
                   lex_gap_multiplier::Real = 0.0,
                   lex_activation_tol::Real = Inf)
    (lex_primary === nothing) == (lex_secondary === nothing) ||
        error("lexicographic primary and secondary objectives must be supplied together")
    lex_abs_tol >= 0 || error("lexicographic absolute tolerance must be nonnegative")
    lex_rel_tol >= 0 || error("lexicographic relative tolerance must be nonnegative")
    lex_gap_multiplier >= 0 || error("lexicographic gap multiplier must be nonnegative")
    lex_activation_tol >= 0 || error("lexicographic activation tolerance must be nonnegative")
    lex_enabled = lex_primary !== nothing
    nvar = num_variables(m)
    ncon = try; num_constraints(m; count_variable_in_set_constraints = true); catch; -1; end
    t0 = time()
    optimize!(m)
    stage1_status = termination_status(m)
    dt = time() - t0
    solvert = try; MOI.get(m, MOI.SolveTimeSec()); catch; NaN; end
    iters = try; Int(MOI.get(m, MOI.BarrierIterations())); catch; -1; end
    if lex_enabled && string(stage1_status) in ("OPTIMAL", "ALMOST_OPTIMAL") &&
       has_values(m) && value(lex_primary) <= Float64(lex_activation_tol)
        stage1_quality = _clarabel_quality_stats(m)
        primary_star = value(lex_primary)
        gap_term = isfinite(stage1_quality.gap_abs) ?
                   Float64(lex_gap_multiplier) * abs(stage1_quality.gap_abs) : 0.0
        primary_tol = max(Float64(lex_abs_tol),
                          Float64(lex_rel_tol) * abs(primary_star), gap_term)
        setup_start = time()
        @constraint(m, lex_primary <= max(primary_star, 0.0) + primary_tol)
        @objective(m, Min, lex_secondary)
        build_time = (isfinite(build_time) ? build_time : 0.0) +
                     (time() - setup_start)
        t2 = time()
        optimize!(m)
        dt += time() - t2
        stage2_status = termination_status(m)
        stage2_solver_time = try; MOI.get(m, MOI.SolveTimeSec()); catch; NaN; end
        stage2_iters = try; Int(MOI.get(m, MOI.BarrierIterations())); catch; -1; end
        solvert = isfinite(solvert) && isfinite(stage2_solver_time) ?
                   solvert + stage2_solver_time : NaN
        iters = iters >= 0 && stage2_iters >= 0 ? iters + stage2_iters : -1
        stage2_quality = _clarabel_quality_stats(m)
        target_meta = merge(target_meta, (
            lexicographic_enabled = true,
            lexicographic_activation_tolerance = Float64(lex_activation_tol),
            lexicographic_activated = true,
            lexicographic_stage1_status = string(stage1_status),
            lexicographic_stage2_status = string(stage2_status),
            lexicographic_primary_star = primary_star,
            lexicographic_primary_tolerance = primary_tol,
            lexicographic_primary_final = has_values(m) ? value(lex_primary) : NaN,
            lexicographic_secondary_final = has_values(m) ? value(lex_secondary) : NaN,
            lexicographic_stage1_gap_abs = stage1_quality.gap_abs,
            lexicographic_stage1_gap_rel = stage1_quality.gap_rel,
            lexicographic_stage2_gap_abs = stage2_quality.gap_abs,
            lexicographic_stage2_gap_rel = stage2_quality.gap_rel,
        ))
        ncon = try; num_constraints(m; count_variable_in_set_constraints = true); catch; -1; end
    elseif lex_enabled && string(stage1_status) in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(m)
        stage1_quality = _clarabel_quality_stats(m)
        primary_star = value(lex_primary)
        target_meta = merge(target_meta, (
            lexicographic_enabled = true,
            lexicographic_activation_tolerance = Float64(lex_activation_tol),
            lexicographic_activated = false,
            lexicographic_stage1_status = string(stage1_status),
            lexicographic_stage2_status = "SKIPPED_ACTIVE_PRIMARY",
            lexicographic_primary_star = primary_star,
            lexicographic_primary_tolerance = 0.0,
            lexicographic_primary_final = primary_star,
            lexicographic_secondary_final = value(lex_secondary),
            lexicographic_stage1_gap_abs = stage1_quality.gap_abs,
            lexicographic_stage1_gap_rel = stage1_quality.gap_rel,
            lexicographic_stage2_gap_abs = NaN,
            lexicographic_stage2_gap_rel = NaN,
        ))
    elseif lex_enabled
        target_meta = merge(target_meta, (
            lexicographic_enabled = true,
            lexicographic_activation_tolerance = Float64(lex_activation_tol),
            lexicographic_activated = false,
            lexicographic_stage1_status = string(stage1_status),
            lexicographic_stage2_status = "NOT_RUN",
            lexicographic_primary_star = NaN,
            lexicographic_primary_tolerance = NaN,
            lexicographic_primary_final = NaN,
            lexicographic_secondary_final = NaN,
            lexicographic_stage1_gap_abs = NaN,
            lexicographic_stage1_gap_rel = NaN,
            lexicographic_stage2_gap_abs = NaN,
            lexicographic_stage2_gap_rel = NaN,
        ))
    end
    st = termination_status(m)
    objv = try
        lex_enabled && has_values(m) ? value(lex_primary) : objective_value(m)
    catch
        NaN
    end
    κval = κvar === nothing ? NaN : κscale * value(κvar)
    κdval = κdvar === nothing ? NaN : value(κdvar)
    # default_recourse: M0 用默认 AGC 方向(ρG=1, αS=0)供审计
    βv = β === nothing ? (default_recourse ? copy(cd.βbar) : Float64[]) : value.(β)
    ρGv = β === nothing ? (default_recourse ? 1.0 : NaN) : 1.0
    αSv = αS === nothing ? (default_recourse ? zeros(cd.nD) : Float64[]) : value.(αS)
    xnt = (g = value.(g), rU = value.(rU), rD = value.(rD),
           snom = value.(snom), cW = value.(cW), β = βv, ρG = ρGv, αS = αSv)
    # 期望切负荷成本
    shed = 0.0
    if β !== nothing
        for k in 1:scen.K, d in 1:cd.nD
            shed += scen.π[k]*cd.cshed[d]*max(xnt.snom[d] - scen.Ω[k]*xnt.αS[d], 0.0)
        end
    else
        shed = sum(cd.cshed[d]*xnt.snom[d] for d in 1:cd.nD)
    end
    gcost = sum(cd.cgen[j]*xnt.g[j] for j in 1:cd.nG)
    return ModelResult(m, st, xnt, κval, κdval, shed, gcost, dt, build_time,
                       solvert, iters, objv, nvar, ncon, target_meta)
end

# ---------- 标定层: 参考成本 Z0 ----------
# 严格按原模型(PDF eq.4.3/7.1): Z0 = 经验参考分布 p̂ 下经验机会约束模型的最优期望切负荷成本。
#   Z0 = min_x Σπ_i c_shed'ζ^(i)  s.t.  x∈Xphys, ζ^(i)≥s^(i)(x)≥0,
#        Σ_i π_i 1{q_m^(i)(x)>0} ≤ ε_m  ∀m∈M  (指示函数, 用 Big-M 二进制重写)
# 口径开关 z0_method: "milp"(默认,精确 eq.7.1) | "cvar"(CVaR-LP 保守凸近似) | "drcc"(旧 P5, :kde_phi)。

# 在给定 model 上装配 x 变量 + Xphys + recourse 闭合, 返回变量与未归一化样本违约事件(供 Z0 复用)。
function _build_xphys_events!(m, cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict)
    nG, nD, nW, nE, K = cd.nG, cd.nD, cd.nW, cd.nE, scen.K
    @variable(m, g[1:nG]); @variable(m, rU[1:nG] >= 0); @variable(m, rD[1:nG] >= 0)
    @variable(m, snom[1:nD] >= 0); @variable(m, cW[1:nW] >= 0)
    @constraint(m, g .+ rU .<= cd.pmax); @constraint(m, g .- rD .>= cd.pmin)
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    @constraint(m, snom .<= snap.lf); @constraint(m, cW .<= curtail_cap)
    for j in 1:nG
        if !cd.agc_mask[j]; @constraint(m, rU[j] == 0); @constraint(m, rD[j] == 0); end
    end
    rucapf = get(P, "ru_cap_frac", Inf); rdcapf = get(P, "rd_cap_frac", Inf)
    isfinite(rucapf) && @constraint(m, sum(rU) <= rucapf * sum(cd.pmax))
    isfinite(rdcapf) && @constraint(m, sum(rD) <= rdcapf * sum(cd.pmax))
    @constraint(m, sum(g) == sum(snap.lf) - sum(snap.wf) + sum(cW) - sum(snom))
    f0 = [ sum(cd.MG[ℓ,j]*g[j] for j in 1:nG) +
           sum(cd.MW[ℓ,r]*(snap.wf[r]-cW[r]) for r in 1:nW) -
           sum(cd.MD[ℓ,d]*(snap.lf[d]-snom[d]) for d in 1:nD)  for ℓ in 1:nE ]
    for ℓ in 1:nE
        isfinite(cd.F_max[ℓ]) || continue
        @constraint(m, f0[ℓ] <= cd.F_max[ℓ]); @constraint(m, -f0[ℓ] <= cd.F_max[ℓ])
    end
    @variable(m, β[1:nG] >= 0); @variable(m, αS[1:nD] >= 0)
    @constraint(m, sum(β) + sum(αS) == 1)
    for j in 1:nG
        cd.agc_mask[j] || @constraint(m, β[j] == 0)
    end
    MGβ = [ sum(cd.MG[ℓ,j]*β[j] for j in 1:nG) for ℓ in 1:nE ]
    PTDFδ = scen.δ * cd.PTDF'
    events = Vector{Vector{Any}}()       # 每个事件: K 维未归一化仿射违约 q_m^(k)(x)
    q_up_parts    = [Any[] for _ in 1:K]
    q_down_parts  = [Any[] for _ in 1:K]
    q_lpos_parts  = [Any[] for _ in 1:K]
    q_lneg_parts  = [Any[] for _ in 1:K]
    q_slow_parts  = [Any[] for _ in 1:K]
    q_shigh_parts = [Any[] for _ in 1:K]
    for j in 1:nG
        cd.agc_mask[j] || continue
        sc = max(cd.pmax[j], 1.0)
        qU = [ -scen.Ω[k]*β[j] - rU[j] for k in 1:K]
        qD = [  scen.Ω[k]*β[j] - rD[j] for k in 1:K]
        push!(events, qU)
        push!(events, qD)
        for k in 1:K
            push!(q_up_parts[k], qU[k] / sc)
            push!(q_down_parts[k], qD[k] / sc)
        end
    end
    for ℓ in 1:nE
        isfinite(cd.F_max[ℓ]) || continue
        sc = max(cd.F_max[ℓ], 1.0)
        ftil = [ f0[ℓ] + PTDFδ[k,ℓ] - scen.Ω[k]*(MGβ[ℓ] +
                  sum(cd.MD[ℓ,d]*αS[d] for d in 1:nD))  for k in 1:K]
        qP = [  ftil[k] - cd.F_max[ℓ] for k in 1:K]
        qN = [ -ftil[k] - cd.F_max[ℓ] for k in 1:K]
        push!(events, qP)
        push!(events, qN)
        for k in 1:K
            push!(q_lpos_parts[k], qP[k] / sc)
            push!(q_lneg_parts[k], qN[k] / sc)
        end
    end
    for d in 1:nD
        sc = max(snap.lf[d], 1.0)
        sdk = [ snom[d] - scen.Ω[k]*αS[d] for k in 1:K ]
        qL = [ -sdk[k] for k in 1:K]
        qH = [  sdk[k] - (snap.lf[d] + scen.eL[k,d]) for k in 1:K]
        push!(events, qL)
        push!(events, qH)
        for k in 1:K
            push!(q_slow_parts[k], qL[k] / sc)
            push!(q_shigh_parts[k], qH[k] / sc)
        end
    end
    target_events = [_category_epigraph!(m, q_up_parts),
                     _category_epigraph!(m, q_down_parts),
                     _category_epigraph!(m, q_lpos_parts),
                     _category_epigraph!(m, q_lneg_parts),
                     _category_epigraph!(m, q_slow_parts),
                     _category_epigraph!(m, q_shigh_parts)]
    Ck_target = [sum(cd.cshed[d]*(snom[d] - scen.Ω[k]*αS[d]) for d in 1:nD) for k in 1:K]
    Ck_positive = Ck_target
    if _common_dimensionless(P)
        @variable(m, χtarget[1:K, 1:nD] >= 0)
        for k in 1:K, d in 1:nD
            @constraint(m, χtarget[k,d] >= snom[d] - scen.Ω[k]*αS[d])
        end
        Ck_positive = [sum(cd.cshed[d]*χtarget[k,d] for d in 1:nD) for k in 1:K]
    end
    # Xphys 隐含的变量盒界(供 Z0 解析 Big-M 用): 由这些边界算 max_box q ≥ max_{Xphys} q, 仍是合法 Big-M。
    vbounds = Dict{Any,Tuple{Float64,Float64}}()
    for j in 1:nG
        vbounds[g[j]]  = (cd.pmin[j], cd.pmax[j])
        rb = max(cd.pmax[j] - cd.pmin[j], 0.0)
        vbounds[rU[j]] = (0.0, rb); vbounds[rD[j]] = (0.0, rb)
        vbounds[β[j]] = (0.0, 1.0)
    end
    for d in 1:nD; vbounds[snom[d]] = (0.0, snap.lf[d]); vbounds[αS[d]] = (0.0, 1.0); end
    for r in 1:nW; vbounds[cW[r]] = (0.0, curtail_cap[r]); end
    return (g=g, rU=rU, rD=rD, snom=snom, cW=cW, β=β, αS=αS,
            events=events, target_events=target_events,
            Ck_target=Ck_target, Ck_positive=Ck_positive,
            vbounds=vbounds)
end

function _fix_reference_decision!(m, v, x)
    for name in (:g, :rU, :rD, :snom, :cW, :β, :αS)
        vars = getproperty(v, name)
        vals = getproperty(x, name)
        length(vars) == length(vals) || error("fixed reference decision dimension mismatch for $name")
        for i in eachindex(vars)
            @constraint(m, vars[i] == vals[i])
        end
    end
    return nothing
end

# 仿射式在变量盒界上的最大值(解析, 无需 LP): max_box (const + Σ c·v) = const + Σ (c>0 ? c·hi : c·lo)。
function _affmax(expr, vbounds::Dict)
    M = JuMP.constant(expr)
    for (c, vr) in JuMP.linear_terms(expr)
        lo, hi = vbounds[vr]
        M += c > 0 ? c * hi : c * lo
    end
    return M
end

# 精确经验机会约束 Z0 (PDF eq.7.1): Big-M = max_{x∈Xphys} q 的合法上界(解析盒界, eq.7.2 的快速版);
# M*≤0 则该样本不会违约, 固定 b=0 删约束。任意合法 Big-M 给出相同 Z0(仅松紧影响 MILP 速度)。
function _z0_empirical_cc_milp(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict)
    tb = time(); K = scen.K; εm = float(P["eps_m"])
    # --- MILP 主体 ---
    mm = Model(HiGHS.Optimizer); set_silent(mm)
    set_optimizer_attribute(mm, "mip_rel_gap", 1e-6)
    v = _build_xphys_events!(mm, cd, snap, scen, P)
    Mn = length(v.events)
    bigM = [[_affmax(v.events[mi][k], v.vbounds) for k in 1:K] for mi in 1:Mn]
    @variable(mm, ζ[1:K, 1:cd.nD] >= 0)
    for k in 1:K, d in 1:cd.nD
        @constraint(mm, ζ[k,d] >= v.snom[d] - scen.Ω[k]*v.αS[d])
    end
    @variable(mm, b[1:Mn, 1:K], Bin)
    for mi in 1:Mn, k in 1:K
        if bigM[mi][k] <= 1e-9
            @constraint(mm, b[mi,k] == 0)            # 物理域内不可能违约 -> 钉 0
        else
            @constraint(mm, v.events[mi][k] <= bigM[mi][k] * b[mi,k])
        end
    end
    for mi in 1:Mn
        @constraint(mm, sum(scen.π[k]*b[mi,k] for k in 1:K) <= εm)
    end
    @objective(mm, Min, sum(scen.π[k]*cd.cshed[d]*ζ[k,d] for k in 1:K, d in 1:cd.nD))
    t0 = time(); optimize!(mm); st = time() - t0
    Z0 = is_solved_and_feasible(mm) ? objective_value(mm) : NaN
    return Z0, (solve_time = st, build_time = time() - tb - st, status = string(termination_status(mm)))
end

function _clarabel_model(P::Dict = Dict{String,Any}())
    m = Model(Clarabel.Optimizer)
    set_silent(m)
    set_optimizer_attribute(m, "max_iter", 2000)
    set_optimizer_attribute(m, "equilibrate_enable", true)
    set_optimizer_attribute(m, "tol_gap_abs", 1e-8)
    set_optimizer_attribute(m, "tol_gap_rel", 1e-8)
    set_optimizer_attribute(m, "tol_feas", 1e-8)
    set_optimizer_attribute(m, "reduced_tol_gap_abs", 5e-5)
    set_optimizer_attribute(m, "reduced_tol_gap_rel", 1e-5)
    set_optimizer_attribute(m, "reduced_tol_feas", 1e-5)
    if haskey(P, "solver_direct_solve_method")
        set_optimizer_attribute(m, "direct_solve_method",
                                Symbol(P["solver_direct_solve_method"]))
    end
    if haskey(P, "solver_equilibrate_max_iter")
        set_optimizer_attribute(m, "equilibrate_max_iter",
                                Int(P["solver_equilibrate_max_iter"]))
    end
    if haskey(P, "solver_iterative_refinement_max_iter")
        set_optimizer_attribute(m, "iterative_refinement_max_iter",
                                Int(P["solver_iterative_refinement_max_iter"]))
    end
    if haskey(P, "solver_dynamic_regularization_delta")
        set_optimizer_attribute(m, "dynamic_regularization_delta",
                                Float64(P["solver_dynamic_regularization_delta"]))
    end
    return m
end

function _clarabel_quality_stats(m)
    try
        info = unsafe_backend(m).solver_info
        return (primal_residual = Float64(info.res_primal),
                dual_residual = Float64(info.res_dual),
                gap_abs = Float64(info.gap_abs),
                gap_rel = Float64(info.gap_rel))
    catch
        return (primal_residual = NaN, dual_residual = NaN,
                gap_abs = NaN, gap_rel = NaN)
    end
end

function _solve_reference_target(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict,
                                 target; bandwidth_profile = nothing,
                                 fixed_x = nothing,
                                 evaluation_only::Bool = false)
    tb = time()
    mm = _clarabel_model(P)
    p   = P["p"]
    γm  = P["gamma_m"]; εm = P["eps_m"]; γ̄ = max(γm, 1 - εm)
    γc  = float(P["gamma_c"])
    hm  = float(P["kde_h_m"]); hc = float(get(P, "kde_h_c", 0.0))
    hscale_m = float(get(P, "kde_rule_scale_m", 1.0))
    hscale_c = float(get(P, "kde_rule_scale_c", 1.0))
    Q   = P["kde_quad_nodes"]
    quad_rule = String(get(P, "kde_quad_rule", "gauss_hermite"))
    quad_weight_floor = float(get(P, "kde_quad_weight_floor", 0.0))
    bw  = get(P, "kde_mode", "rule")
    bwc = get(P, "cost_kde_mode", "off")
    bandwidth_profile === nothing || bandwidth_profile isa FixedBandwidthProfile ||
        error("bandwidth_profile must be a FixedBandwidthProfile")
    common_target = _common_dimensionless(P)
    common_target && _validate_common_profile(bandwidth_profile, cd, snap, P)
    gamma_target = common_target ? _common_target_gamma(P) : γ̄
    gamma_cost = common_target ? gamma_target : γc
    bw_safety = common_target ? "fixed" :
        (bandwidth_profile === nothing ? bw : "fixed")
    bw_cost = common_target ? "fixed" :
        (bandwidth_profile === nothing ? bwc : (bwc == "off" ? "off" : "fixed"))
    (ζm, ωm) = bw_safety == "off" ? ([0.0], [1.0]) :
        _normal_nodes(Q, quad_rule; weight_floor = quad_weight_floor)
    (ζc, ωc) = bw_cost == "off" ? ([0.0], [1.0]) :
        _normal_nodes(Q, quad_rule; weight_floor = quad_weight_floor)

    v = _build_xphys_events!(mm, cd, snap, scen, P)
    fixed_x === nothing || _fix_reference_decision!(mm, v, fixed_x)
    bandwidth_profile !== nothing && length(bandwidth_profile.target_h) != 6 &&
        error("target bandwidth count does not match reference safety targets")
    safety_risks = [add_hmcr_risk!(mm, v.target_events[mi], scen.π, p, gamma_target,
                                   bandwidth_profile === nothing ? hm : bandwidth_profile.target_h[mi],
                                   ζm, ωm; bw_mode = bw_safety,
                                   rule_scale = hscale_m) for mi in 1:6]
    cost_scale = common_target ? _target_cost_scale(cd, snap, P) : 1.0
    cost_losses = common_target ? v.Ck_positive ./ cost_scale : v.Ck_target
    cost_bandwidth = if common_target
        bandwidth_profile.cost_h / cost_scale
    elseif bandwidth_profile === nothing
        hc
    else
        bw_cost == "off" ? 0.0 : bandwidth_profile.cost_h
    end
    cost_risk = add_hmcr_risk!(mm, cost_losses, scen.π, p, gamma_cost,
                               cost_bandwidth,
                               ζc, ωc; bw_mode = bw_cost,
                               rule_scale = hscale_c)

    target_scale = 1.0
    if target == :cost
        if !evaluation_only
            for mi in 1:6
                @constraint(mm, safety_risks[mi] <= 0)
            end
        end
        # Legacy cost is numerically divided by a physical cost scale and then
        # multiplied back.  Common-mode cost is already dimensionless.
        target_scale = common_target ? 1.0 :
            max(Float64(get(P, "voll", 1000.0)) * sum(snap.lf), 1.0)
        @objective(mm, Min, cost_risk / target_scale)
    else
        if !evaluation_only
            for mi in 1:6
                mi == target && continue
                @constraint(mm, safety_risks[mi] <= 0)
            end
        end
        @objective(mm, Min, safety_risks[target])
    end

    buildt = time() - tb
    t0 = time()
    optimize!(mm)
    solvet = time() - t0
    status = string(termination_status(mm))
    ok = status in ("OPTIMAL", "ALMOST_OPTIMAL")
    z = ok ? target_scale * objective_value(mm) : NaN
    xsol = ok ? (g = value.(v.g), rU = value.(v.rU), rD = value.(v.rD),
                 snom = value.(v.snom), cW = value.(v.cW),
                 β = value.(v.β), αS = value.(v.αS)) : nothing
    quality = _clarabel_quality_stats(mm)
    return z, (solve_time = solvet, build_time = buildt,
               status = status, x = xsol,
               primal_residual = quality.primal_residual,
               dual_residual = quality.dual_residual,
               gap_abs = quality.gap_abs, gap_rel = quality.gap_rel)
end

"""
Evaluate the cost plus six safety HMCR payoffs at one fixed reference decision.

Once the physical decision is fixed, the seven risk epigraph blocks have no
remaining shared optimization variables.  Minimizing their sum therefore
obtains the same componentwise payoffs as seven separate evaluations while
using one conic solve.  This routine is deliberately restricted to the common
positive-bandwidth protocol used by the payoff-range calibration.
"""
function _evaluate_reference_payoff_row(cd::CaseData, snap::Snapshot,
                                        scen::ScenarioData, P::Dict, fixed_x;
                                        bandwidth_profile = nothing)
    fixed_x === nothing && error("anchor-payoff evaluation requires a fixed reference decision")
    _common_dimensionless(P) || error(
        "anchor-payoff evaluation is defined only for common_dimensionless targets")
    _validate_common_profile(bandwidth_profile, cd, snap, P)

    tb = time()
    mm = _clarabel_model(P)
    p = P["p"]
    gamma_target = _common_target_gamma(P)
    Q = P["kde_quad_nodes"]
    quad_rule = String(get(P, "kde_quad_rule", "gauss_hermite"))
    quad_weight_floor = float(get(P, "kde_quad_weight_floor", 0.0))
    hscale_m = float(get(P, "kde_rule_scale_m", 1.0))
    hscale_c = float(get(P, "kde_rule_scale_c", 1.0))
    (xi_m, w_m) = _normal_nodes(Q, quad_rule; weight_floor = quad_weight_floor)
    (xi_c, w_c) = _normal_nodes(Q, quad_rule; weight_floor = quad_weight_floor)

    v = _build_xphys_events!(mm, cd, snap, scen, P)
    _fix_reference_decision!(mm, v, fixed_x)
    safety_risks = [add_hmcr_risk!(mm, v.target_events[mi], scen.π, p,
                                   gamma_target, bandwidth_profile.target_h[mi],
                                   xi_m, w_m; bw_mode = "fixed",
                                   rule_scale = hscale_m) for mi in 1:6]
    cost_scale = _target_cost_scale(cd, snap, P)
    cost_risk = add_hmcr_risk!(mm, v.Ck_positive ./ cost_scale, scen.π, p,
                               gamma_target,
                               bandwidth_profile.cost_h / cost_scale,
                               xi_c, w_c; bw_mode = "fixed",
                               rule_scale = hscale_c)
    @objective(mm, Min, cost_risk + sum(safety_risks))

    buildt = time() - tb
    t0 = time()
    optimize!(mm)
    solvet = time() - t0
    status = string(termination_status(mm))
    ok = status in ("OPTIMAL", "ALMOST_OPTIMAL") && has_values(mm)
    payoffs = ok ? Float64[value(cost_risk); value.(safety_risks)] : fill(NaN, 7)
    quality = _clarabel_quality_stats(mm)
    return payoffs, (solve_time = solvet, build_time = buildt,
                     status = status,
                     primal_residual = quality.primal_residual,
                     dual_residual = quality.dual_residual,
                     gap_abs = quality.gap_abs, gap_rel = quality.gap_rel)
end

function retarget_targets(meta, rho_tau::Real, omega_d::Real;
                          rule = nothing,
                          eps_c::Real = 1000.0)
    rho = Float64(rho_tau)
    omega = Float64(omega_d)
    0.0 <= rho <= 1.0 || error("rho_tau must lie in [0,1]")
    omega >= 0 || error("omega_d must be nonnegative")
    mode = rule === nothing ?
        (hasproperty(meta, :rule_mode) ? String(meta.rule_mode) : "legacy") :
        String(rule)
    if mode == "common_dimensionless"
        tau_c = meta.Z0c + rho * abs(meta.Z0c)
        tau_m = [z + rho * abs(z) for z in meta.Z0m]
        kvals = hasproperty(meta, :k) ? Float64.(collect(meta.k)) : ones(6)
        length(kvals) == 6 || error("common target metadata must contain six k coefficients")
        all(v -> isfinite(v) && v >= 0.0, kvals) ||
            error("common target k coefficients must be finite and nonnegative")
        return merge(meta, (τc = tau_c, τm = tau_m, k = kvals,
                            rho_tau = rho, omega_d = omega,
                            tau_rule = "signed_relative"))
    elseif mode == "legacy"
        tau_c = (1.0 + rho) * meta.Z0c
        tau_m = [z < 0 ? 0.0 : (1.0 + rho) * z for z in meta.Z0m]
        den = max(abs(meta.Z0c), Float64(eps_c))
        kvals = abs.(meta.Z0m) ./ den
        return merge(meta, (τc = tau_c, τm = tau_m, k = kvals,
                            rho_tau = rho, omega_d = omega,
                            tau_rule = "legacy_piecewise_zero"))
    end
    error("unknown target retargeting rule: $mode")
end

function calibrate_targets(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict;
                           bandwidth_profile = nothing)
    common_target = _common_dimensionless(P)
    common_target && _validate_common_profile(bandwidth_profile, cd, snap, P)

    Z0c, rc = _solve_reference_target(cd, snap, scen, P, :cost;
                                      bandwidth_profile = bandwidth_profile)
    Z0m = fill(NaN, 6)
    stats = Vector{Any}(undef, 7)
    stats[1] = rc

    if common_target
        # This opt-in branch changes only the safety-scale coefficients.  The
        # targets themselves remain the seven payoffs at the cost anchor (row
        # one), so the anchor table is used as a reproducible numerical scale
        # estimate rather than as a replacement target construction.
        if _common_target_k_rule(P) == "payoff_range_no_eps"
            anchor_stats = Vector{Any}(undef, 7)
            anchor_stats[1] = rc
            if rc.x === nothing
                for mi in 1:6
                    anchor_stats[mi + 1] = rc
                end
            else
                for mi in 1:6
                    _, anchor_stats[mi + 1] = _solve_reference_target(
                        cd, snap, scen, P, mi;
                        bandwidth_profile = bandwidth_profile)
                end
            end

            payoff_matrix = fill(NaN, 7, 7)
            payoff_stats = Vector{Any}(undef, 7)
            for ai in 1:7
                anchor = anchor_stats[ai]
                if anchor.x === nothing
                    payoff_stats[ai] = (
                        solve_time = 0.0, build_time = 0.0,
                        status = string(anchor.status),
                        primal_residual = Float64(anchor.primal_residual),
                        dual_residual = Float64(anchor.dual_residual),
                        gap_abs = Float64(anchor.gap_abs),
                        gap_rel = Float64(anchor.gap_rel),
                    )
                else
                    payoff_matrix[ai, :], payoff_stats[ai] =
                        _evaluate_reference_payoff_row(
                            cd, snap, scen, P, anchor.x;
                            bandwidth_profile = bandwidth_profile)
                end
            end

            # Row one is the cost-anchor evaluation under exactly the same
            # risk representation as every other table entry.  It is the sole
            # source for the unchanged target levels tau; only k comes from
            # the column ranges below.
            payoff_Z0c = payoff_matrix[1, 1]
            payoff_Z0m = vec(copy(payoff_matrix[1, 2:end]))
            cost_scale = _target_cost_scale(cd, snap, P)
            gamma_target = _common_target_gamma(P)
            kscale = _common_target_scale_coefficients(
                cd, snap, bandwidth_profile, payoff_Z0c, payoff_Z0m, P;
                payoff_matrix = payoff_matrix)
            base = (
                Z0c = payoff_Z0c, Z0m = payoff_Z0m,
                蟿c = payoff_Z0c, 蟿m = copy(payoff_Z0m),
                k = kscale.k, rho_tau = 0.0,
                omega_d = Float64(get(P, "omega_d", 1.0)),
                rule_mode = "common_dimensionless",
                tau_rule = "signed_relative",
                k_rule = kscale.rule,
                k_priority = kscale.priorities,
                k_cost_scale_reference = kscale.cost_scale_reference,
                k_safety_scale_reference = kscale.safety_scale_reference,
                k_cost_sigma_proxy = kscale.cost_sigma_proxy,
                k_safety_sigma_proxy = kscale.safety_sigma_proxy,
                k_cost_resolution = kscale.cost_resolution,
                k_safety_resolution = kscale.safety_resolution,
                payoff_matrix = copy(payoff_matrix),
                payoff_ideal = kscale.payoff_ideal,
                payoff_anchor_maximum = kscale.payoff_anchor_maximum,
                payoff_ranges = kscale.payoff_ranges,
                payoff_anchor_count = kscale.payoff_anchor_count,
                payoff_scale_source = kscale.payoff_scale_source,
                payoff_zero_range_policy = kscale.payoff_zero_range_policy,
                payoff_anchor_labels = ["cost", "reserve_up", "reserve_down",
                                        "line_pos", "line_neg", "shed_low",
                                        "shed_high"],
                payoff_objective_labels = ["cost", "reserve_up", "reserve_down",
                                           "line_pos", "line_neg", "shed_low",
                                           "shed_high"],
                payoff_evaluation_method = "joint_fixed_x_sum_epigraph",
                payoff_anchor_build_times = [s.build_time for s in anchor_stats],
                payoff_anchor_solve_times = [s.solve_time for s in anchor_stats],
                payoff_evaluation_statuses = [s.status for s in payoff_stats],
                payoff_evaluation_primal_residuals = [s.primal_residual for s in payoff_stats],
                payoff_evaluation_dual_residuals = [s.dual_residual for s in payoff_stats],
                payoff_evaluation_gap_abs_values = [s.gap_abs for s in payoff_stats],
                payoff_evaluation_gap_rel_values = [s.gap_rel for s in payoff_stats],
                payoff_evaluation_build_times = [s.build_time for s in payoff_stats],
                payoff_evaluation_solve_times = [s.solve_time for s in payoff_stats],
                calibration_solve_count = 14,
                cost_scale = cost_scale,
                target_gamma = gamma_target,
                cost_h = bandwidth_profile.cost_h,
                cost_h_normalized = bandwidth_profile.cost_h / cost_scale,
                target_h = copy(bandwidth_profile.target_h),
                bandwidth_raw_n = bandwidth_profile.raw_n,
                bandwidth_multiplier = bandwidth_profile.multiplier,
                bandwidth_floor_active = any(h <=
                    Float64(get(P, "target_kde_h_min_rel", 1e-6)) * (1 + 1e-10)
                    for h in bandwidth_profile.target_h) ||
                    bandwidth_profile.cost_h / cost_scale <=
                    Float64(get(P, "target_kde_h_min_rel", 1e-6)) * (1 + 1e-10),
                anchor_safety_max = maximum(payoff_Z0m),
                solve_time = sum(s.solve_time for s in anchor_stats) +
                             sum(s.solve_time for s in payoff_stats),
                build_time = sum(s.build_time for s in anchor_stats) +
                             sum(s.build_time for s in payoff_stats),
                statuses = [s.status for s in anchor_stats],
                primal_residuals = [s.primal_residual for s in anchor_stats],
                dual_residuals = [s.dual_residual for s in anchor_stats],
                gap_abs_values = [s.gap_abs for s in anchor_stats],
                gap_rel_values = [s.gap_rel for s in anchor_stats],
                reference_solutions = [s.x for s in anchor_stats],
            )
            rho = Float64(get(P, "rho_tau", get(P, "rho_cost", 0.0)))
            return retarget_targets(base, rho, Float64(get(P, "omega_d", 1.0));
                                    rule = "common_dimensionless")
        end
        if rc.x === nothing
            for mi in 1:6
                stats[mi + 1] = rc
            end
        else
            # All seven targets are evaluated at the same jointly feasible
            # positive-bandwidth anchor.  This avoids combining six mutually
            # incompatible rotating ideal points into one target vector.
            for mi in 1:6
                Z0m[mi], stats[mi + 1] = _solve_reference_target(
                    cd, snap, scen, P, mi;
                    bandwidth_profile = bandwidth_profile,
                    fixed_x = rc.x, evaluation_only = true)
            end
        end
        cost_scale = _target_cost_scale(cd, snap, P)
        gamma_target = _common_target_gamma(P)
        kscale = _common_target_scale_coefficients(
            cd, snap, bandwidth_profile, Z0c, Z0m, P)
        base = (Z0c = Z0c, Z0m = Z0m, τc = Z0c, τm = copy(Z0m),
                k = kscale.k, rho_tau = 0.0,
                omega_d = Float64(get(P, "omega_d", 1.0)),
                rule_mode = "common_dimensionless",
                tau_rule = "signed_relative",
                k_rule = kscale.rule,
                k_priority = kscale.priorities,
                k_cost_scale_reference = kscale.cost_scale_reference,
                k_safety_scale_reference = kscale.safety_scale_reference,
                k_cost_sigma_proxy = kscale.cost_sigma_proxy,
                k_safety_sigma_proxy = kscale.safety_sigma_proxy,
                k_cost_resolution = kscale.cost_resolution,
                k_safety_resolution = kscale.safety_resolution,
                cost_scale = cost_scale,
                target_gamma = gamma_target,
                cost_h = bandwidth_profile.cost_h,
                cost_h_normalized = bandwidth_profile.cost_h / cost_scale,
                target_h = copy(bandwidth_profile.target_h),
                bandwidth_raw_n = bandwidth_profile.raw_n,
                bandwidth_multiplier = bandwidth_profile.multiplier,
                bandwidth_floor_active = any(h <=
                    Float64(get(P, "target_kde_h_min_rel", 1e-6)) * (1 + 1e-10)
                    for h in bandwidth_profile.target_h) ||
                    bandwidth_profile.cost_h / cost_scale <=
                    Float64(get(P, "target_kde_h_min_rel", 1e-6)) * (1 + 1e-10),
                anchor_safety_max = maximum(Z0m),
                solve_time = sum(s.solve_time for s in stats),
                build_time = sum(s.build_time for s in stats),
                statuses = [s.status for s in stats],
                primal_residuals = [s.primal_residual for s in stats],
                dual_residuals = [s.dual_residual for s in stats],
                gap_abs_values = [s.gap_abs for s in stats],
                gap_rel_values = [s.gap_rel for s in stats],
                reference_solutions = fill(rc.x, 7))
        rho = Float64(get(P, "rho_tau", get(P, "rho_cost", 0.0)))
        return retarget_targets(base, rho, Float64(get(P, "omega_d", 1.0));
                                rule = "common_dimensionless")
    end

    for mi in 1:6
        Z0m[mi], stats[mi + 1] = _solve_reference_target(cd, snap, scen, P, mi;
                                                         bandwidth_profile = bandwidth_profile)
    end
    base = (Z0c = Z0c, Z0m = Z0m, τc = Z0c, τm = copy(Z0m), k = ones(6),
            rho_tau = 0.0, omega_d = Float64(get(P, "omega_d", 1.0)),
            rule_mode = "legacy", tau_rule = "legacy_piecewise_zero",
            cost_scale = 1.0, target_gamma = Float64(P["gamma_c"]),
            solve_time = sum(s.solve_time for s in stats),
            build_time = sum(s.build_time for s in stats),
            statuses = [s.status for s in stats],
            primal_residuals = [s.primal_residual for s in stats],
            dual_residuals = [s.dual_residual for s in stats],
            gap_abs_values = [s.gap_abs for s in stats],
            gap_rel_values = [s.gap_rel for s in stats],
            reference_solutions = [s.x for s in stats])
    rho = Float64(get(P, "rho_tau", get(P, "rho_cost", 0.0)))
    eps_value = Float64(get(P, "eps_c", get(P, "voll", 1000.0)))
    return retarget_targets(base, rho, Float64(get(P, "omega_d", 1.0));
                            rule = "legacy", eps_c = eps_value)
end

function calibrate_Z0(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict)
    method = get(P, "z0_method", "milp")
    if method == "milp"
        return _z0_empirical_cc_milp(cd, snap, scen, P)
    elseif method == "cvar"          # 经验机会约束的 CVaR-LP 保守凸近似 (M1 口径)
        r = build_ols(cd, snap, scen, P; safety = :cvar, cost = :expected)
        return r.shed_cost, r
    elseif method == "drcc"          # 旧 P5 口径 (DRCC 可行集), 保留作对照
        r = build_ols(cd, snap, scen, P; safety = :kde_phi, cost = :expected)
        return r.shed_cost, r
    else
        error("未知 z0_method=$method (应为 milp|cvar|drcc)")
    end
end

end # module
