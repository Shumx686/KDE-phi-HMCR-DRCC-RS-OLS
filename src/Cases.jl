# =====================================================================
#  Cases.jl  —  网络解析与潮流映射 (模块 2 的静态部分)
#
#  统一解析 RBTS / RTS79 / RTS_GMLC 的 MATPOWER 文本算例，构造:
#    - DC-PTDF (nbranch × nbus)
#    - 注入->潮流映射 M^G, M^D, M^W, M^Δ (论文 §符号)
#    - 机组上下界、线路上限、边际成本、固定 AGC 方向 β̄
#  约定: 净注入误差 δ 定义在母线注入坐标 (n = nbus)，故 M^Δ = PTDF，Ω = 1ᵀδ。
# =====================================================================
module Cases

using LinearAlgebra, SparseArrays

export RawCase, parse_matpower, build_ptdf, selection_matrix, GenData

# ---------- 原始算例 ----------
struct RawCase
    name::String
    baseMVA::Float64
    bus::Matrix{Float64}      # 行: 母线; 列同 MATPOWER bus 段
    gen::Matrix{Float64}      # 行: 机组
    branch::Matrix{Float64}   # 行: 支路
    gencost::Matrix{Float64}  # 行: 机组 (与 gen 对齐)
end

"读取一个 MATPOWER `mpc.<field> = [ ... ];` 数值块为 Matrix{Float64}。"
function _read_block(lines::Vector{String}, field::String)
    # 找到 `mpc.field = [` 起始
    start = findfirst(l -> occursin(Regex("mpc\\.$(field)\\s*=\\s*\\["), l), lines)
    start === nothing && error("找不到字段 mpc.$field")
    rows = Vector{Vector{Float64}}()
    i = start + 1
    # 同一行可能已有数据 (少见)；统一从下一行扫到 `];`
    while i <= length(lines)
        raw = lines[i]
        if occursin("]", raw)
            # 处理 `];` 之前可能仍有一行数据
            pre = split(raw, "]")[1]
            _push_numrow!(rows, pre)
            break
        end
        _push_numrow!(rows, raw)
        i += 1
    end
    isempty(rows) && error("字段 mpc.$field 为空")
    ncol = maximum(length, rows)
    M = fill(0.0, length(rows), ncol)
    for (r, v) in enumerate(rows)
        for (c, x) in enumerate(v)
            M[r, c] = x
        end
    end
    return M
end

function _push_numrow!(rows, raw::AbstractString)
    s = strip(raw)
    # 去注释 / 去行尾分号
    s = replace(s, r"%.*$" => "")
    s = replace(s, ";" => " ")
    s = strip(s)
    isempty(s) && return
    toks = split(s)
    vals = Float64[]
    ok = true
    for t in toks
        x = tryparse(Float64, t)
        x === nothing && (ok = false; break)
        push!(vals, x)
    end
    ok && !isempty(vals) && push!(rows, vals)
end

"解析 MATPOWER .txt/.m 算例文件。"
function parse_matpower(path::AbstractString; name::AbstractString="case")
    lines = readlines(path)
    baseline = let
        idx = findfirst(l -> occursin("baseMVA", l), lines)
        idx === nothing ? 100.0 : something(tryparse(Float64,
            strip(replace(split(lines[idx], "=")[2], ";" => ""))), 100.0)
    end
    bus     = _read_block(lines, "bus")
    gen     = _read_block(lines, "gen")
    branch  = _read_block(lines, "branch")
    gencost = _read_block(lines, "gencost")
    return RawCase(name, baseline, bus, gen, branch, gencost)
end

# ---------- 机组 ----------
struct GenData
    bus::Vector{Int}       # 机组所在母线 (原始母线号)
    pmax::Vector{Float64}
    pmin::Vector{Float64}
    cost::Vector{Float64}  # 边际成本 (gencost 线性项)
    status::Vector{Bool}
end

function gendata(rc::RawCase)
    G = size(rc.gen, 1)
    bus   = Int.(rc.gen[:, 1])
    status = rc.gen[:, 8] .> 0
    pmax  = rc.gen[:, 9]
    pmin  = rc.gen[:, 10]
    # gencost 两种 MATPOWER 模型:
    #   model=2 多项式 [2 su sd n c_{n-1}..c0]: 线性取 c1; 二次取一次项。
    #   model=1 分段线性 [1 su sd npts x1 y1 x2 y2 ...]: 取整体平均斜率为线性边际成本。
    cost = fill(0.0, G)
    for g in 1:G
        model = Int(rc.gencost[g, 1])
        if model == 1
            npts = Int(rc.gencost[g, 4])
            x1 = rc.gencost[g, 5]; y1 = rc.gencost[g, 6]
            xn = rc.gencost[g, 5 + 2*(npts-1)]; yn = rc.gencost[g, 6 + 2*(npts-1)]
            cost[g] = xn > x1 ? (yn - y1) / (xn - x1) : 0.0
        else
            ncoef = Int(rc.gencost[g, 4])
            cost[g] = ncoef >= 3 ? rc.gencost[g, 6] : rc.gencost[g, 5]
        end
    end
    return GenData(bus, pmax, pmin, cost, status)
end

# ---------- PTDF ----------
"""
    build_ptdf(rc) -> (PTDF, busids, busidx, F_max, fbus, tbus)

DC-PTDF: 行=支路, 列=母线。flow[MW] = PTDF * Pinj[MW]。
忽略 r、线路电纳，b_ℓ = 1/x_ℓ。slack 取 bus type==3 (无则取第一个母线)。
"""
function build_ptdf(rc::RawCase)
    busids = Int.(rc.bus[:, 1])
    n = length(busids)
    busidx = Dict(busids[i] => i for i in 1:n)

    L = size(rc.branch, 1)
    online = rc.branch[:, 11] .> 0
    fbus = Int.(rc.branch[:, 1])
    tbus = Int.(rc.branch[:, 2])
    x    = rc.branch[:, 4]
    b    = [online[ℓ] && x[ℓ] != 0 ? 1.0 / x[ℓ] : 0.0 for ℓ in 1:L]

    # 关联矩阵 A (L×n): +1 from, -1 to
    A = spzeros(L, n)
    for ℓ in 1:L
        A[ℓ, busidx[fbus[ℓ]]] += 1.0
        A[ℓ, busidx[tbus[ℓ]]] -= 1.0
    end
    Bf   = spdiagm(0 => b) * A          # 支路潮流 = Bf * θ
    Bbus = A' * spdiagm(0 => b) * A     # 母线导纳 (DC)

    # slack
    slack_rows = findall(rc.bus[:, 2] .== 3)
    slack = isempty(slack_rows) ? 1 : slack_rows[1]
    keep = setdiff(1:n, slack)

    PTDF = zeros(L, n)
    Bred = Matrix(Bbus[keep, keep])
    Bfr  = Matrix(Bf[:, keep])
    # PTDF_reduced = Bf_reduced * inv(Bbus_reduced)
    PTDFred = Bfr / Bred
    PTDF[:, keep] .= PTDFred
    # slack 列保持 0

    # 线路上限 (rateA); 0 视为无限制
    F_raw = rc.branch[:, 6]
    F_max = [online[ℓ] && F_raw[ℓ] > 0 ? F_raw[ℓ] : Inf for ℓ in 1:L]

    return PTDF, busids, busidx, F_max, fbus, tbus, online
end

"""
    selection_matrix(busidx, target_buses) -> S (nbus × k)

把长度 k 的某类设备量 (按 target_buses 顺序) 累加到母线注入向量。
列 j 对应 target_buses[j]，在其母线行置 1。同母线多设备各占一列。
"""
function selection_matrix(busidx::Dict{Int,Int}, nbus::Int, target_buses::Vector{Int})
    k = length(target_buses)
    S = spzeros(nbus, k)
    for j in 1:k
        S[busidx[target_buses[j]], j] = 1.0
    end
    return S
end

end # module
