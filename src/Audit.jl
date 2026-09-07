# =====================================================================
#  Audit.jl  —  模块 5a: 样本外审计器 (模型无关)
#
#  只吃 解x + 全测试快照, 按代码内部净注入误差口径计算实际实现量并统计可靠性指标。
#  内部 δ_code=(w_act-w_f)-(l_act-l_f)=-δ_paper, 所以下式与论文净负荷误差加号口径等价:
#    s_{d,t}=snom_d - Ω_t αS_d ;  f_{ℓ,t}=f0_ℓ + PTDFδ_t - Ω_t(MGβ_ℓ ρG + MD_ℓ αS)
#    备用越界: -Ω_t ρG β_j > rU_j 或 Ω_t ρG β_j > rD_j
#  审计一律用 *该快照实际* δ_t (单次实现), 跨全部测试快照统计频率。
# =====================================================================
module Audit

using LinearAlgebra
using ..CaseInterface

export audit_snapshot, AuditAgg, aggregate_audit, DEFAULT_AUDIT_TOL_MW

# One common physical deadband for MW-valued reliability quantities. Using a
# tighter (1 W) threshold only for the binary shed-bound flag made a 46.8 W
# numerical/sign excursion count as 100% risk while EDNS treated it as zero.
# The 1 kW deadband is applied identically to every model.
const DEFAULT_AUDIT_TOL_MW = 1e-3

# 单快照审计指标。tol_mw: 失负荷判定阈值, 滤除求解器数值微量(默认 1e-3 MW)。
# βbar 参数名为历史兼容；调用方可传入模型自身的 β。M0 默认用案例 β̄，
# M1--M3/M6 使用优化得到的 β，M4/M5 使用各自文献基线 β。
function audit_snapshot(cd::CaseData, x::NamedTuple, snap::Snapshot;
                        tol_mw::Float64 = DEFAULT_AUDIT_TOL_MW,
                        βbar::AbstractVector = cd.βbar)
    isfinite(tol_mw) && tol_mw >= 0 ||
        error("audit MW tolerance must be finite and nonnegative")
    Ω = sum(snap.δ)
    nG, nD, nE = cd.nG, cd.nD, cd.nE
    gen_rec  = isfinite(x.ρG)        # 机组 AGC 调节(M0-M6 都有, 文献基线用自身 β 覆盖)
    shed_rec = !isempty(x.αS)        # 切负荷仿射响应(M0-M3/M6 有, M4/M5 为 here-and-now 无)

    # 预测点基准潮流 f0 (该快照预测点)
    f0 = [ sum(cd.MG[ℓ,j]*x.g[j] for j in 1:nG) +
           sum(cd.MW[ℓ,r]*(snap.wf[r]-x.cW[r]) for r in 1:cd.nW) -
           sum(cd.MD[ℓ,d]*(snap.lf[d]-x.snom[d]) for d in 1:nD) for ℓ in 1:nE ]
    PTDFδ = cd.PTDF * snap.δ                       # nE
    MGβ = [ sum(cd.MG[ℓ,j]*βbar[j] for j in 1:nG) for ℓ in 1:nE ]
    ftil = [ f0[ℓ] + PTDFδ[ℓ] -
             (gen_rec ? Ω*MGβ[ℓ]*x.ρG : 0.0) -
             (shed_rec ? Ω*sum(cd.MD[ℓ,d]*x.αS[d] for d in 1:nD) : 0.0) for ℓ in 1:nE ]

    # 实际切负荷 s̃_d, 钳正得未供电 (滤除 < tol_mw 的数值微量)
    s̃ = [ x.snom[d] - (shed_rec ? Ω*x.αS[d] : 0.0) for d in 1:nD ]
    shed_mw = sum(s̃[d] > tol_mw ? s̃[d] : 0.0 for d in 1:nD)

    # 线路越限
    ratios = [ isfinite(cd.F_max[ℓ]) ? abs(ftil[ℓ])/cd.F_max[ℓ] : 0.0 for ℓ in 1:nE ]
    maxratio = maximum(ratios)
    line_viol = maxratio > 1.0 + 1e-6

    # 备用越界 (机组实时调节超出预置备用; β_j=0 的机组 reg=0 不计)
    reserve_viol = false
    if gen_rec
        for j in 1:nG
            reg = -Ω * x.ρG * βbar[j]
            if reg > x.rU[j] + tol_mw || -reg > x.rD[j] + tol_mw
                reserve_viol = true; break
            end
        end
    end

    # 切负荷边界越界: s̃_d > 实际负荷 lact_d (过切) 或 s̃_d < 0 (负切)
    shedbound_viol = any(s̃[d] > snap.lact[d] + tol_mw ||
                         s̃[d] < -tol_mw for d in 1:nD)

    # 弃风审计: 调度弃风超出该快照实际可用风电
    curtail = sum(x.cW)
    curtail_audit = sum(max(x.cW[r] - snap.wact[r], 0.0) for r in 1:cd.nW)

    gen_cost = sum(cd.cgen[j]*x.g[j] for j in 1:nG)

    return (shed_mw = shed_mw, lol = shed_mw > tol_mw,
            line_viol = line_viol, max_ratio = maxratio,
            reserve_viol = reserve_viol, shedbound_viol = shedbound_viol,
            curtail = curtail, curtail_audit = curtail_audit,
            gen_cost = gen_cost)
end

struct AuditAgg
    n::Int
    EDNS::Float64; EENS::Float64; LOLP::Float64; LOLP_thr::Float64
    line_prob::Float64; reserve_prob::Float64; shedbound_prob::Float64
    max_flow_ratio::Float64
    AvgCurtail::Float64; WindCurtailAudit::Float64
    Cost::Float64
end

"聚合多快照审计。rows::Vector of audit_snapshot 输出; Δt 小时; voll; gen_costs 每快照发电成本。"
function aggregate_audit(rows::Vector, Δt::Float64, voll::Float64, lolp_thr::Float64)
    n = length(rows)
    g(f) = sum(f(r) for r in rows) / n
    EDNS = g(r -> r.shed_mw)
    EENS = sum(r.shed_mw for r in rows) * Δt
    LOLP = g(r -> r.lol ? 1.0 : 0.0)
    LOLP_thr = g(r -> r.shed_mw > lolp_thr ? 1.0 : 0.0)
    line_prob = g(r -> r.line_viol ? 1.0 : 0.0)
    reserve_prob = g(r -> r.reserve_viol ? 1.0 : 0.0)
    shedbound_prob = g(r -> r.shedbound_viol ? 1.0 : 0.0)
    maxr = maximum(r.max_ratio for r in rows)
    AvgCurtail = g(r -> r.curtail)
    WindCurtailAudit = g(r -> r.curtail_audit)
    Cost = g(r -> r.gen_cost + voll * r.shed_mw)
    return AuditAgg(n, EDNS, EENS, LOLP, LOLP_thr, line_prob, reserve_prob,
                    shedbound_prob, maxr, AvgCurtail, WindCurtailAudit, Cost)
end

end # module
