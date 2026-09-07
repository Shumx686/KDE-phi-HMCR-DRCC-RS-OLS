# =====================================================================
#  Baselines.jl — 文献基线 (模块 4)
#
#  Public labels:
#    M4 = Wasserstein-CVaR-DRCC-DC-OLS (W-DRO-SP-OLS.tex 的精确锥/线性重写)。
#    M5  = Moment-DRCC-DC-OLS based on empirical first and second moments.
#  M4 with ρ_W=0 退化为经验随机规划/SAA-CVaR 基线 (SP 文献口径)。
#
#  与 M2-M3/M6 的差异(文献基线自身结构, 非我们模型):
#    - recourse: 逐机组 AGC 参与因子 β(变量, 1ᵀβ=1), 无切负荷仿射响应;
#    - 切负荷 ΔL 为 here-and-now(不随 Ω), 风电弃电 c 同;
#    - 联合 CVaR 作用于"缺额侧"事件 M_def = 上备用 + 线路上下限;
#    - 目标 min 1ᵀΔL + ε_c 1ᵀc (最小切负荷, 微弱弃电惩罚)。
#  共享: 网络/PTDF、机组上下界、线路上限、压缩误差样本 ω̂_i=δ̄_i 与权重 π_i、ε。
# =====================================================================
module Baselines

using JuMP, Clarabel, LinearAlgebra
using ..CaseInterface

export build_wdro, build_moment_dro, WdroResult, m4_active_event_indices

const M4_NATIVE_EVENT_SET = "deficit_side_up_reserve_and_bidirectional_lines"

"Native W-DRO-SP-OLS event indices in the implementation order RU|RD|F+|F-."
function m4_active_event_indices(nG::Int, finite_lines::AbstractVector{Bool})
    nE = length(finite_lines)
    active = collect(1:nG)
    append!(active, (2nG + e for e in 1:nE if finite_lines[e]))
    append!(active, (2nG + nE + e for e in 1:nE if finite_lines[e]))
    return active
end

struct WdroResult
    status
    x::NamedTuple          # g, rU, rD, snom(=ΔL), cW(=c), β, ρG=1, αS=[]
    obj::Float64           # 1ᵀΔL + ε_c 1ᵀc
    shed::Float64          # 1ᵀΔL (计划切负荷)
    solve_time::Float64
    build_time::Float64
    solver_time::Float64
    iters::Int
    n_var::Int
    n_con::Int
    jump_report_distance::Float64
    primal_residual::Float64
    dual_residual::Float64
    gap_abs::Float64
    gap_rel::Float64
end

function _max_primal_violation(m)
    has_values(m) || return Inf
    try
        report = primal_feasibility_report(m; atol = 0.0)
        return isempty(report) ? 0.0 : maximum(values(report))
    catch
        return NaN
    end
end

function _clarabel_quality(m)
    try
        info = unsafe_backend(m).solver_info
        return (primal = Float64(info.res_primal),
                dual = Float64(info.res_dual),
                gap_abs = Float64(info.gap_abs),
                gap_rel = Float64(info.gap_rel))
    catch
        return (primal = NaN, dual = NaN, gap_abs = NaN, gap_rel = NaN)
    end
end

function _apply_optional_clarabel_settings!(m, P::Dict)
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
    return m
end

"""
    build_wdro(cd, snap, scen, P; rho_wass)

求解 Wasserstein-CVaR-DRCC-OLS (rho_wass=0 → SP/SAA-CVaR)。
P 为 cfg["model"]; 读 eps_m(风险水平 ε)、baseline 段的 eps_curtail、support_margin。
"""
function build_wdro(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict;
                    rho_wass::Float64)
    tb = time()
    nG, nD, nW, nE = cd.nG, cd.nD, cd.nW, cd.nE
    nb = size(scen.δ, 2); K = scen.K
    ε   = P["eps_m"]
    εc  = get(P, "eps_curtail", 1e-3)
    mgn = get(P, "support_margin", 0.25)

    # 误差样本与盒型支撑 Ξ=[ω_lo,ω_hi]
    Ω̂ = scen.δ                                   # K × nb
    support_source = lowercase(String(get(P, "wasserstein_support_source", "raw_training")))
    if support_source == "raw_training"
        ωlo = copy(scen.raw_lower)
        ωhi = copy(scen.raw_upper)
    elseif support_source in ("representatives", "compressed")
        ωlo = vec(minimum(Ω̂, dims = 1))
        ωhi = vec(maximum(Ω̂, dims = 1))
    else
        error("unknown Wasserstein support source: $support_source")
    end
    half = (ωhi .- ωlo) ./ 2 .+ 1e-9
    ωlo .-= mgn .* half; ωhi .+= mgn .* half
    cΞ = (ωhi .+ ωlo) ./ 2; dΞ = (ωhi .- ωlo) ./ 2

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
    _apply_optional_clarabel_settings!(m, P)

    @variable(m, g[1:nG]); @variable(m, β[1:nG] >= 0)
    @variable(m, rD[1:nG] >= 0); @variable(m, rU[1:nG] >= 0)
    @variable(m, 0.0 <= ΔL[d=1:nD] <= snap.lf[d])
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    @variable(m, 0.0 <= c[r=1:nW] <= curtail_cap[r])
    @variable(m, τ)

    # 物理可行域 X_t
    @constraint(m, g .- rD .>= cd.pmin)
    @constraint(m, g .+ rU .<= cd.pmax)
    @constraint(m, sum(g) + sum(snap.wf[r]-c[r] for r in 1:nW) ==
                   sum(snap.lf[d]-ΔL[d] for d in 1:nD))
    @constraint(m, sum(β) == 1)
    for j in 1:nG
        if !cd.agc_mask[j]
            @constraint(m, β[j] == 0)
            @constraint(m, rU[j] == 0)
            @constraint(m, rD[j] == 0)
        end
    end
    # Keep the deterministic physical domain identical to ModelCore.  These
    # optional aggregate reserve caps are inactive unless explicitly enabled.
    rucapf = get(P, "ru_cap_frac", Inf); rdcapf = get(P, "rd_cap_frac", Inf)
    isfinite(rucapf) && @constraint(m, sum(rU) <= rucapf * sum(cd.pmax))
    isfinite(rdcapf) && @constraint(m, sum(rD) <= rdcapf * sum(cd.pmax))

    # 缺额侧事件 M_def 的仿射系数 ã_m(x)∈R^nb, b̃_m(x,τ)。m=0: ã=0,b̃=0。
    # 基准潮流 ℓ_e = MG g + MW(wf-c) - MD(lf-ΔL)
    ℓ = [ sum(cd.MG[e,j]*g[j] for j in 1:nG) +
          sum(cd.MW[e,r]*(snap.wf[r]-c[r]) for r in 1:nW) -
          sum(cd.MD[e,d]*(snap.lf[d]-ΔL[d]) for d in 1:nD)  for e in 1:nE ]
    for e in 1:nE
        isfinite(cd.F_max[e]) || continue
        @constraint(m, ℓ[e] <= cd.F_max[e])
        @constraint(m, -ℓ[e] <= cd.F_max[e])
    end
    # MG_e·β
    MGβ = [ sum(cd.MG[e,j]*β[j] for j in 1:nG) for e in 1:nE ]

    # ã_m[b] 与 b̃_m: 实现顺序为上备用(nG) | 下备用(nG) |
    # 线路上(nE) | 线路下(nE)。W-DRO-SP-OLS 原生缺额侧集合只包含
    # 上备用与双向线路；下备用块不得为了“公平”或结果表现而加入。
    event_set = String(get(P, "m4_event_set", ""))
    event_set == M4_NATIVE_EVENT_SET ||
        error("M4 requires frozen native event set $M4_NATIVE_EVENT_SET; got $(repr(event_set))")
    nMdef = 2*nG + 2*nE
    ã(mi) = begin
        if mi <= nG                                       # 上备用 -β_j
            [ -β[mi] for _ in 1:nb ]
        elseif mi <= 2nG                                  # 下备用 +β_j
            [  β[mi-nG] for _ in 1:nb ]
        elseif mi <= 2nG + nE                             # 线路上 q_e
            e = mi - 2nG; [ cd.PTDF[e,b] - MGβ[e] for b in 1:nb ]
        else                                              # 线路下 -q_e
            e = mi - 2nG - nE; [ -(cd.PTDF[e,b] - MGβ[e]) for b in 1:nb ]
        end
    end
    b̃(mi) = begin
        if mi <= nG;        -rU[mi] - τ
        elseif mi <= 2nG;   -rD[mi-nG] - τ
        elseif mi <= 2nG+nE; (e=mi-2nG;  ℓ[e] - cd.F_max[e] - τ)
        else                (e=mi-2nG-nE; -ℓ[e] - cd.F_max[e] - τ)
        end
    end
    line_finite(mi) = (mi <= 2nG) ? true :
        isfinite(cd.F_max[mi <= 2nG+nE ? mi-2nG : mi-2nG-nE])
    # 预计算原生缺额侧事件系数(与样本 i 无关)，只保留有限线路。
    finite_lines = Bool[isfinite(cd.F_max[e]) for e in 1:nE]
    active = m4_active_event_indices(nG, finite_lines)
    all(line_finite, active) || error("M4 active event set contains a non-finite line")
    amL = Dict(mi => ã(mi) for mi in active)
    bbL = Dict(mi => b̃(mi) for mi in active)

    @variable(m, s[1:K])

    if rho_wass <= 0
        # SP / SAA-CVaR: s_i ≥ ã_m^T ω̂_i + b̃_m ; 且 s_i≥0 (m=0)
        for i in 1:K
            @constraint(m, s[i] >= 0)
            for mi in active
                am = amL[mi]
                @constraint(m, s[i] >= sum(am[b]*Ω̂[i,b] for b in 1:nb) + bbL[mi])
            end
        end
        @constraint(m, τ + (1/ε)*sum(scen.π[i]*s[i] for i in 1:K) <= 0)
    else
        # 完整 W-DRO 对偶: ζ_im, θ_im≥0, λ≥0
        @variable(m, λ >= 0)
        @variable(m, ζ[1:K, active, 1:nb])
        @variable(m, θ[1:K, active, 1:nb] >= 0)
        for i in 1:K
            @constraint(m, s[i] >= 0)
            for mi in active
                am = amL[mi]
                @constraint(m, s[i] >= bbL[mi] +
                    sum((am[b]+ζ[i,mi,b])*cΞ[b] for b in 1:nb) +
                    sum(θ[i,mi,b] for b in 1:nb) -
                    sum(ζ[i,mi,b]*Ω̂[i,b] for b in 1:nb))
                for b in 1:nb
                    @constraint(m, θ[i,mi,b] >=  dΞ[b]*(am[b]+ζ[i,mi,b]))
                    @constraint(m, θ[i,mi,b] >= -dΞ[b]*(am[b]+ζ[i,mi,b]))
                    @constraint(m, ζ[i,mi,b] <=  λ)
                    @constraint(m, ζ[i,mi,b] >= -λ)
                end
            end
        end
        @constraint(m, τ + (1/ε)*(λ*rho_wass + sum(scen.π[i]*s[i] for i in 1:K)) <= 0)
    end

    # 原生主目标仍为 min 1ᵀΔL + ε_c 1ᵀc (W-DRO-SP-OLS 原文)。
    # 整个原生主项乘以公共 VOLL 是正数缩放，不改变其最优集；随后再加
    # 与 ModelCore 同量纲的 ε_gen c_gen'g，避免 baseline 中平局项相对放大 VOLL 倍。
    primary_scale = Float64(get(P, "voll", maximum(cd.cshed)))
    isfinite(primary_scale) && primary_scale > 0 ||
        error("baseline primary objective scale must be finite and positive")
    εgen = get(P, "eps_gen_tiebreak", 1e-6)
    @objective(m, Min, primary_scale *
                       (sum(ΔL[d] for d in 1:nD) + εc*sum(c[r] for r in 1:nW)) +
                       εgen*sum(cd.cgen[j]*g[j] for j in 1:nG))

    nvar = num_variables(m)
    ncon = try; num_constraints(m; count_variable_in_set_constraints = true); catch; -1; end
    buildt = time() - tb
    t0 = time(); optimize!(m); st = termination_status(m); dt = time()-t0
    solvert = try; MOI.get(m, MOI.SolveTimeSec()); catch; NaN; end
    iters = try; Int(MOI.get(m, MOI.BarrierIterations())); catch; -1; end
    objv = try; objective_value(m); catch; NaN; end
    pviol = _max_primal_violation(m)
    quality = _clarabel_quality(m)

    if !has_values(m)
        x = _baseline_x(fill(NaN, nG), fill(NaN, nG), fill(NaN, nG),
                        fill(NaN, nD), fill(NaN, nW), fill(NaN, nG))
        return WdroResult(st, x, objv, NaN, dt, buildt, solvert, iters, nvar, ncon,
                          pviol, quality.primal, quality.dual,
                          quality.gap_abs, quality.gap_rel)
    end
    gv = value.(g); βv = value.(β)
    x = _baseline_x(gv, value.(rU), value.(rD), value.(ΔL), value.(c), βv)
    shed = sum(value.(ΔL))
    return WdroResult(st, x, objv, shed, dt, buildt, solvert, iters, nvar, ncon,
                      pviol, quality.primal, quality.dual,
                      quality.gap_abs, quality.gap_rel)
end

function _scenario_moments(scen::ScenarioData; cov_scale::Float64 = 1.0,
                           source::AbstractString = "raw_training")
    delta = getproperty(scen, Symbol("\u03b4"))
    prob = getproperty(scen, Symbol("\u03c0"))
    K, nb = size(delta)
    source = lowercase(String(source))
    if source == "raw_training"
        mu = copy(scen.raw_mean)
        sigma = copy(scen.raw_cov)
    elseif source in ("representatives", "compressed")
        mu = zeros(nb)
        for i in 1:K, b in 1:nb
            mu[b] += prob[i] * delta[i, b]
        end
        sigma = zeros(nb, nb)
        for i in 1:K
            d = vec(delta[i, :]) .- mu
            sigma .+= prob[i] .* (d * d')
        end
    else
        error("unknown moment source: $source")
    end
    sigma .*= cov_scale
    sigma = (sigma + sigma') ./ 2
    for b in 1:nb
        sigma[b, b] += 1e-9
    end
    sigma = Symmetric(sigma)
    ev = eigen(sigma)
    vals = sqrt.(max.(ev.values, 0.0))
    return mu, Diagonal(vals) * ev.vectors'
end

function _add_moment_drcc!(m, a, b_expr, mu::Vector{Float64},
                           sqrt_sigma::AbstractMatrix{Float64},
                           eps::Float64)
    0.0 < eps < 1.0 || error("moment DRCC epsilon must be in (0, 1), got $eps")
    nb = length(mu)
    length(a) == nb ||
        throw(DimensionMismatch("moment DRCC affine vector has length $(length(a)) != $nb"))
    kappa = sqrt((1.0 - eps) / eps)
    t = @variable(m, lower_bound = 0.0)
    v = [sum(sqrt_sigma[row, col] * a[col] for col in 1:nb) for row in 1:nb]
    @constraint(m, [t; v] in SecondOrderCone())
    @constraint(m, b_expr + sum(mu[col] * a[col] for col in 1:nb) + kappa * t <= 0.0)
    return nothing
end

function _baseline_x(g, rU, rD, snom, cW, beta)
    names = (:g, :rU, :rD, :snom, :cW, Symbol("\u03c1G"), Symbol("\u03b1S"), Symbol("\u03b2"))
    vals = (g, rU, rD, snom, cW, 1.0, Float64[], beta)
    return NamedTuple{names}(vals)
end

"""
    build_moment_dro(cd, snap, scen, P)

Solve an individual moment-ambiguity DRCC baseline. Each affine safety event is
protected against all distributions with the empirical first two moments via
the one-sided Chebyshev/Cantelli SOC reformulation. This is a literature
baseline, separate from the KDE-phi-HMCR robust-satisficing model.
"""
function build_moment_dro(cd::CaseData, snap::Snapshot, scen::ScenarioData, P::Dict)
    tb = time()
    nG, nD, nW, nE = cd.nG, cd.nD, cd.nW, cd.nE
    nb = size(getproperty(scen, Symbol("\u03b4")), 2)
    eps = Float64(get(P, "moment_eps", P["eps_m"]))
    cov_scale = Float64(get(P, "moment_cov_scale", 1.0))
    eps_c = get(P, "eps_curtail", 1e-3)
    eps_gen = get(P, "eps_gen_tiebreak", 1e-6)
    moment_source = String(get(P, "moment_source", "raw_training"))
    mu, sqrt_sigma = _scenario_moments(scen; cov_scale = cov_scale,
                                       source = moment_source)

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
    _apply_optional_clarabel_settings!(m, P)

    @variable(m, g[1:nG])
    @variable(m, beta[1:nG] >= 0.0)
    @variable(m, rD[1:nG] >= 0.0)
    @variable(m, rU[1:nG] >= 0.0)
    @variable(m, 0.0 <= shed[d=1:nD] <= snap.lf[d])
    curtail_cap = sample_safe_curtailment_cap(cd, snap, scen)
    @variable(m, 0.0 <= curt[r=1:nW] <= curtail_cap[r])

    @constraint(m, g .- rD .>= cd.pmin)
    @constraint(m, g .+ rU .<= cd.pmax)
    @constraint(m, sum(g) + sum(snap.wf[r] - curt[r] for r in 1:nW) ==
                   sum(snap.lf[d] - shed[d] for d in 1:nD))
    @constraint(m, sum(beta) == 1.0)
    for j in 1:nG
        if !cd.agc_mask[j]
            @constraint(m, beta[j] == 0.0)
            @constraint(m, rU[j] == 0.0)
            @constraint(m, rD[j] == 0.0)
        end
    end
    # Match ModelCore's optional aggregate reserve limits when configured.
    rucapf = get(P, "ru_cap_frac", Inf); rdcapf = get(P, "rd_cap_frac", Inf)
    isfinite(rucapf) && @constraint(m, sum(rU) <= rucapf * sum(cd.pmax))
    isfinite(rdcapf) && @constraint(m, sum(rD) <= rdcapf * sum(cd.pmax))

    flow0 = [sum(cd.MG[e, j] * g[j] for j in 1:nG) +
             sum(cd.MW[e, r] * (snap.wf[r] - curt[r]) for r in 1:nW) -
             sum(cd.MD[e, d] * (snap.lf[d] - shed[d]) for d in 1:nD) for e in 1:nE]
    for e in 1:nE
        isfinite(cd.F_max[e]) || continue
        @constraint(m, flow0[e] <= cd.F_max[e])
        @constraint(m, -flow0[e] <= cd.F_max[e])
    end
    mgbeta = [sum(cd.MG[e, j] * beta[j] for j in 1:nG) for e in 1:nE]

    for j in 1:nG
        _add_moment_drcc!(m, [-beta[j] for _ in 1:nb], -rU[j], mu, sqrt_sigma, eps)
        _add_moment_drcc!(m, [ beta[j] for _ in 1:nb], -rD[j], mu, sqrt_sigma, eps)
    end
    for e in 1:nE
        isfinite(cd.F_max[e]) || continue
        a = [cd.PTDF[e, b] - mgbeta[e] for b in 1:nb]
        _add_moment_drcc!(m, a, flow0[e] - cd.F_max[e], mu, sqrt_sigma, eps)
        _add_moment_drcc!(m, [-a[b] for b in 1:nb], -flow0[e] - cd.F_max[e], mu, sqrt_sigma, eps)
    end

    # A positive VOLL scaling preserves the native moment baseline's primary
    # shed/curtail optimum while putting the shared generation tie-break on the
    # same relative cost scale as ModelCore.
    primary_scale = Float64(get(P, "voll", maximum(cd.cshed)))
    isfinite(primary_scale) && primary_scale > 0 ||
        error("baseline primary objective scale must be finite and positive")
    @objective(m, Min, primary_scale *
                       (sum(shed[d] for d in 1:nD) + eps_c * sum(curt[r] for r in 1:nW)) +
                       eps_gen * sum(cd.cgen[j] * g[j] for j in 1:nG))

    nvar = num_variables(m)
    ncon = try
        num_constraints(m; count_variable_in_set_constraints = true)
    catch
        -1
    end
    buildt = time() - tb
    t0 = time()
    optimize!(m)
    st = termination_status(m)
    dt = time() - t0
    solvert = try
        MOI.get(m, MOI.SolveTimeSec())
    catch
        NaN
    end
    iters = try
        Int(MOI.get(m, MOI.BarrierIterations()))
    catch
        -1
    end
    objv = try
        objective_value(m)
    catch
        NaN
    end
    pviol = _max_primal_violation(m)
    quality = _clarabel_quality(m)

    if !has_values(m)
        x = _baseline_x(fill(NaN, nG), fill(NaN, nG), fill(NaN, nG),
                        fill(NaN, nD), fill(NaN, nW), fill(NaN, nG))
        return WdroResult(st, x, objv, NaN, dt, buildt, solvert, iters, nvar, ncon,
                          pviol, quality.primal, quality.dual,
                          quality.gap_abs, quality.gap_rel)
    end

    x = _baseline_x(value.(g), value.(rU), value.(rD),
                    value.(shed), value.(curt), value.(beta))
    return WdroResult(st, x, objv, sum(value.(shed)), dt, buildt,
                      solvert, iters, nvar, ncon, pviol,
                      quality.primal, quality.dual, quality.gap_abs, quality.gap_rel)
end

end # module
