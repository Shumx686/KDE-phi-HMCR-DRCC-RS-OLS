# =====================================================================
#  DataPipeline.jl  —  模块 1: 数据清洗 / 时间戳对齐 / 误差样本 / 划分 / 压缩
#
#  产物 (落盘到 data_cache/<system>/):
#    panel.arrow        全小时面板 (Lf,Wf,Lact,Wact,δ,Ω) —— 审计与逐快照求解用
#    scenarios_K.csv    带权代表样本 (δ̄ 各母线列 + pi + is_tail) —— 求解用
#    split.toml         train/val/test 索引范围 (时间顺序，无泄露)
#
#  口径: δ 定义在母线注入坐标, δ_b = (w_act-w_f)_b - (l_act-l_f)_b, Ω = 1ᵀδ。
#         RBTS/RTS79 无原生预测 -> 合成 AR(1) 预测误差。
# =====================================================================
module DataPipeline

using LinearAlgebra, Random, Statistics, Dates
using DataFrames, CSV, XLSX, Arrow, Clustering, StableRNGs

using ..Cases

const PIPELINE_VERSION = 9

const FORMAL_TAIL_CRITERIA = (
    "absOmega",
    "positiveOmega",
    "negativeOmega",
    "forecastNetLoad",
    "availableWind",
    "forecastLineLoading",
    "reserveResponseStress",
)

export Panel, build_panel, build_delta!, time_split, compress_scenarios,
       apply_screen_and_split, run_pipeline

# ---------------------------------------------------------------------
struct Panel
    system::String
    timestamps::Vector{DateTime}
    load_buses::Vector{Int}
    wind_buses::Vector{Int}
    wind_cap::Vector{Float64}
    Lf::Matrix{Float64}   # T × n_load  预测负荷
    Lact::Matrix{Float64} # T × n_load  实际负荷
    Wf::Matrix{Float64}   # T × n_wind  预测可用风电
    Wact::Matrix{Float64} # T × n_wind  实际可用风电
    δ::Matrix{Float64}    # T × nbus    净注入误差(母线坐标)
    Ω::Vector{Float64}    # T
end

# ---------- 工具: 风场容量 / 读风电 / 读负荷 ----------
function wind_capacity_from_name(fname::AbstractString)
    m = match(r"capacity-?\s*(\d+(?:\.\d+)?)\s*MW", fname)
    m === nothing ? error("无法从文件名解析风电容量: $fname") : parse(Float64, m.captures[1])
end

"读 15min 风电 xlsx 的时间列(A)与 Power(MW)列(按表头定位, 列布局随文件而异), 重采样为逐时均值。"
function read_wind_hourly(path::AbstractString)
    xf = XLSX.readxlsx(path)
    sh = XLSX.sheetnames(xf)[1]
    ws = xf[sh]
    dim = XLSX.get_dimension(ws)
    nrow = dim.stop.row_number; ncol = dim.stop.column_number
    hdr = vec(XLSX.readdata(path, sh, XLSX.CellRange(1, 1, 1, ncol)))
    pidx = findfirst(h -> h !== missing && occursin("Power", string(h)), hdr)
    pidx === nothing && error("xlsx 找不到 Power 列: $path")
    tcol = vec(XLSX.readdata(path, sh, XLSX.CellRange(2, 1, nrow, 1)))
    pcol = vec(XLSX.readdata(path, sh, XLSX.CellRange(2, pidx, nrow, pidx)))
    ts = DateTime[]; pw = Float64[]
    for i in 1:length(tcol)
        t = tcol[i]; p = pcol[i]
        (t === missing || p === missing) && continue
        pv = p isa Number ? Float64(p) : tryparse(Float64, string(p))   # 跳过公式串等
        pv === nothing && continue
        dt = t isa DateTime ? t : DateTime(string(t), dateformat"yyyy-mm-dd HH:MM:SS")
        push!(ts, dt); push!(pw, pv)
    end
    # 按小时聚合(均值)
    bucket = Dict{DateTime,Vector{Float64}}()
    for i in 1:length(ts)
        h = floor(ts[i], Hour)
        push!(get!(bucket, h, Float64[]), pw[i])
    end
    hrs = sort(collect(keys(bucket)))
    vals = [mean(bucket[h]) for h in hrs]
    length(hrs) <= 1 && return hrs, vals

    full_hrs = DateTime[]
    full_vals = Float64[]
    filled = 0
    for i in 1:(length(hrs)-1)
        push!(full_hrs, hrs[i])
        push!(full_vals, vals[i])
        gap = Int(round(Dates.value(hrs[i+1] - hrs[i]) / 3_600_000))
        gap >= 1 || error("风电小时序列非递增: $path")
        for k in 1:(gap-1)
            α = k / gap
            push!(full_hrs, hrs[i] + Hour(k))
            push!(full_vals, (1 - α) * vals[i] + α * vals[i+1])
            filled += 1
        end
    end
    push!(full_hrs, last(hrs))
    push!(full_vals, last(vals))
    filled > 0 && @info "风电小时序列缺口已线性插补" file=basename(path) filled=filled
    return full_hrs, full_vals
end

"读 BUS_LOADS.csv: 首列时间, 其余列名=母线号。返回 (timestamps, busids, 矩阵 T×nbus_load)。"
function read_load_csv(path::AbstractString)
    df = CSV.read(path, DataFrame)
    tcol = names(df)[1]
    fmt = dateformat"yyyy-mm-dd H:MM"
    raw_ts = [DateTime(strip(string(x)), fmt) for x in df[!, tcol]]
    ts = [first(raw_ts) + Hour(i - 1) for i in 1:length(raw_ts)]
    if raw_ts != ts
        bad = count(i -> i == length(raw_ts) ? false : Dates.value(raw_ts[i+1] - raw_ts[i]) != 3_600_000,
                    1:length(raw_ts))
        @info "负荷时间标签不连续, 已按行号重建连续小时轴" file=basename(path) bad_gaps=bad start=first(raw_ts)
    end
    busids = parse.(Int, names(df)[2:end])
    M = Matrix{Float64}(coalesce.(df[!, 2:end], 0.0))
    return ts, busids, M
end

# ---------- AR(1) 合成预测误差 ----------
"返回与 base 同形状的平稳 AR(1) 误差矩阵, 稳态标准差 = sigma(标量或按列)。"
function synth_ar1_error(rng, base::Matrix{Float64}, sigma, phi::Float64)
    T, K = size(base)
    e = zeros(T, K)
    sig = sigma isa AbstractVector ? sigma : fill(float(sigma), K)
    c = sqrt(1 - phi^2)
    for k in 1:K
        e[1, k] = sig[k] * randn(rng)
        for t in 2:T
            e[t, k] = phi * e[t-1, k] + c * sig[k] * randn(rng)
        end
    end
    return e
end

# ---------- 构建面板 (RBTS / RTS79) ----------
function build_panel(system::String, cfg::Dict, case_dir::String)
    sysl = lowercase(system)
    if sysl in ("rbts", "rts79")
        return _build_panel_rbts_rts79(system, cfg, case_dir)
    elseif sysl in ("rts_gmlc", "gmlc")
        return _build_panel_gmlc(system, cfg, case_dir)
    else
        error("未知系统 $system")
    end
end

function _build_panel_rbts_rts79(system, cfg, case_dir)
    rng = StableRNG(cfg["run"]["seed"])
    d   = cfg["data"]; fc = d["forecast"]

    # --- 负荷 (实际): 读满全年, 由 run_pipeline 决定应力筛选/划分 ---
    load_csv = joinpath(case_dir, "BUS_LOADS.csv")
    lts, load_buses, Lall = read_load_csv(load_csv)
    T = size(Lall, 1)                       # 全年小时数
    Lact = Lall[1:T, :]

    # --- 风电 (实际, 逐时) ---
    wind_files = filter(f -> occursin("Wind farm site", f), readdir(case_dir))
    sort!(wind_files)                      # site1..site6 顺序
    wbus_map = cfg["network"]["wind_bus"][uppercase(system)]
    wind_buses = Int[]; wind_cap = Float64[]; Wcols = Vector{Vector{Float64}}()
    for f in wind_files
        sm = match(r"site\s*(\d+)", f)
        key = "site$(sm.captures[1])"
        haskey(wbus_map, key) || continue
        wts, vals = read_wind_hourly(joinpath(case_dir, f))
        @assert length(vals) >= T "风电样本不足: $f"
        if T > 1
            gaps = [Dates.value(wts[i+1] - wts[i]) for i in 1:(T-1)]
            all(gaps .== 3_600_000) || error("风电小时序列不连续: $f")
        end
        if first(wts) != first(lts)
            @info "风电与负荷原始年份/起点不同, 按年内顺序小时对齐" system=system file=f load_start=first(lts) wind_start=first(wts)
        end
        push!(wind_buses, wbus_map[key])
        push!(wind_cap, wind_capacity_from_name(f))
        push!(Wcols, vals[1:T])
    end
    Wact = reduce(hcat, Wcols)             # T × n_wind
    caprow = reshape(wind_cap, 1, :)
    n_under = count(Wact .< 0.0)
    n_over = count(Wact .> caprow)
    if n_under + n_over > 0
        @info "实际风电越界值已截断到 [0, capacity]" system=system below_zero=n_under above_capacity=n_over
    end
    Wact = clamp.(Wact, 0.0, caprow)

    # --- 合成 AR(1) 预测 ---
    if fc["method"] == "ar1"
        σL = vec(d["forecast"]["load_sigma_frac"] .* mean(Lact, dims=1))   # 每母线
        eL = synth_ar1_error(rng, Lact, σL, float(fc["load_phi"]))
        Lf = clamp.(Lact .- eL, 0.0, Inf)
        σW = fc["wind_sigma_frac"] .* wind_cap
        eW = synth_ar1_error(rng, Wact, σW, float(fc["wind_phi"]))
        Wf = clamp.(Wact .- eW, 0.0, reshape(wind_cap, 1, :))
    else
        error("forecast.method=$(fc["method"]) 暂未实现")
    end

    ts = lts[1:T]
    nbus_load = length(load_buses)
    nbus_wind = length(wind_buses)
    return Panel(system, ts, load_buses, wind_buses, wind_cap,
                 Lf, Lact, Wf, Wact,
                 zeros(T, 0), zeros(T))    # δ,Ω 由 build_delta! 填
end

# ---------- RTS_GMLC 适配器 ----------
# 读 GMLC 时序 CSV (Year,Month,Day,Period, value...), 按 group 行聚合为逐时。
function _read_gmlc_seq(path::AbstractString, group::Int)
    df = CSV.read(path, DataFrame)
    vcols = names(df)[5:end]
    M = Matrix{Float64}(coalesce.(df[!, 5:end], 0.0))
    group == 1 && return M, vcols
    nh = size(M, 1) ÷ group
    H = zeros(nh, size(M, 2))
    for h in 1:nh
        H[h, :] = vec(mean(M[((h-1)*group+1):(h*group), :], dims=1))
    end
    return H, vcols
end

# 常规可调度机组掩码 (与 .m gen 行对齐): 排除 风/光/储/调相机。
function _gmlc_conv_mask(case_dir::AbstractString)
    gp = joinpath(case_dir, "RTS_Data", "SourceData", "gen.csv")
    df = CSV.read(gp, DataFrame)
    fuel = string.(df[!, "Fuel"])
    excl = Set(["Wind", "Solar", "Storage", "Sync_Cond"])
    return [!(fuel[i] in excl) for i in 1:length(fuel)]
end

function _build_panel_gmlc(system, cfg, case_dir)
    d = cfg["data"]
    T = (d["split_train_days"] + d["split_val_days"] + d["split_test_days"]) * 24
    ts_dir = joinpath(case_dir, "RTS_Data", "timeseries_data_files")
    DAl, acols = _read_gmlc_seq(joinpath(ts_dir, "Load", "DAY_AHEAD_regional_Load.csv"), 1)
    RTl, _     = _read_gmlc_seq(joinpath(ts_dir, "Load", "REAL_TIME_regional_Load.csv"), 12)
    DAw, wcols = _read_gmlc_seq(joinpath(ts_dir, "WIND", "DAY_AHEAD_wind.csv"), 1)
    RTw, _     = _read_gmlc_seq(joinpath(ts_dir, "WIND", "REAL_TIME_wind.csv"), 12)
    @assert min(size(DAl,1), size(RTl,1), size(DAw,1), size(RTw,1)) >= T "GMLC 时序不足 $T 小时"

    bdf = CSV.read(joinpath(case_dir, "RTS_Data", "SourceData", "bus.csv"), DataFrame)
    busid = Int.(bdf[!, "Bus ID"]); area = Int.(bdf[!, "Area"]); Pd = Float64.(bdf[!, "MW Load"])
    load_buses = busid[Pd .> 0]
    area_total = Dict{Int,Float64}()
    for i in 1:length(busid); area_total[area[i]] = get(area_total, area[i], 0.0) + Pd[i]; end

    nL = length(load_buses); Lf = zeros(T, nL); Lact = zeros(T, nL)
    for (j, b) in enumerate(load_buses)
        bi = findfirst(==(b), busid); a = area[bi]
        ac = findfirst(==(string(a)), acols)
        share = Pd[bi] / area_total[a]
        Lf[:, j]   = DAl[1:T, ac] .* share
        Lact[:, j] = RTl[1:T, ac] .* share
    end

    wind_buses = [parse(Int, split(c, "_")[1]) for c in wcols]
    Wf = DAw[1:T, :]; Wact = RTw[1:T, :]
    wcap = vec(maximum(Wact, dims = 1))
    ts = [DateTime(2020, 1, 1, 0) + Hour(h - 1) for h in 1:T]
    return Panel(system, ts, load_buses, wind_buses, wcap, Lf, Lact, Wf, Wact,
                 zeros(T, 0), zeros(T))
end

# ---------- 内部净注入误差 δ (母线坐标) ----------
# 论文采用净负荷误差 δ_paper=(l_act-l_f)-(w_act-w_f)。代码内部存储
# δ_code=(w_act-w_f)-(l_act-l_f)=-δ_paper, 并在模型响应中同步使用相反号。
"δ_b = (w_act-w_f)_b - (l_act-l_f)_b ; Ω=1ᵀδ. 返回新 Panel。"
function build_delta!(panel::Panel, rc::RawCase)
    PTDF, busids, busidx, _ = build_ptdf(rc)
    nbus = length(busids); T = length(panel.timestamps)
    δ = zeros(T, nbus)
    eW = panel.Wact .- panel.Wf
    eL = panel.Lact .- panel.Lf
    for (j, b) in enumerate(panel.wind_buses)
        δ[:, busidx[b]] .+= eW[:, j]
    end
    for (j, b) in enumerate(panel.load_buses)
        δ[:, busidx[b]] .-= eL[:, j]
    end
    Ω = vec(sum(δ, dims=2))
    return Panel(panel.system, panel.timestamps, panel.load_buses, panel.wind_buses,
                 panel.wind_cap, panel.Lf, panel.Lact, panel.Wf, panel.Wact, δ, Ω)
end

# ---------- 应力筛选: 低风高负荷应力域(外生、无泄露) ----------
# 默认用预测净负荷 1ᵀl^f - 1ᵀw^f 作为低风高负荷的单一应力分数。
# 分数只用预测值, 决策时已知; 保留最高的前 frac 比例小时。
function screen_indices(panel::Panel, frac::Float64)
    netload = vec(sum(panel.Lf, dims=2)) .- vec(sum(panel.Wf, dims=2))
    thr = quantile(netload, 1 - frac)
    return sort(findall(netload .>= thr))    # 时间顺序
end

"把 Panel 沿时间维限制到 idx(母线/容量不变)。"
function restrict_panel(p::Panel, idx::Vector{Int})
    return Panel(p.system, p.timestamps[idx], p.load_buses, p.wind_buses, p.wind_cap,
                 p.Lf[idx, :], p.Lact[idx, :], p.Wf[idx, :], p.Wact[idx, :],
                 isempty(p.δ) ? p.δ : p.δ[idx, :], isempty(p.Ω) ? p.Ω : p.Ω[idx])
end

# ---------- 时间顺序划分 (完整时间轴上的固定天数窗口) ----------
function time_split(panel::Panel, cfg::Dict)
    d = cfg["data"]
    td, vd, ed = d["split_train_days"], d["split_val_days"], d["split_test_days"]
    resolution = Float64(get(d, "resolution_hours", 1.0))
    resolution > 0 || error("data.resolution_hours must be positive")
    steps_per_day_float = 24.0 / resolution
    steps_per_day = round(Int, steps_per_day_float)
    isapprox(steps_per_day_float, steps_per_day; atol = 1e-12, rtol = 0.0) ||
        error("data.resolution_hours must divide 24 exactly")
    ntr, nva, nte = Int(td) * steps_per_day, Int(vd) * steps_per_day,
                     Int(ed) * steps_per_day
    T = length(panel.timestamps)
    required = ntr + nva + nte
    T >= required ||
        error("full panel has $T samples but the fixed chronological protocol requires $required")
    tr = 1:ntr
    va = (ntr+1):(ntr+nva)
    te = (ntr+nva+1):required
    return (train=tr, val=va, test=te)
end

"Apply the configured chronological split and low-wind/high-load screen."
function apply_screen_and_split(panel::Panel, cfg::Dict)
    sccfg = get(cfg["data"], "scenario", Dict("screen" => false))
    if !get(sccfg, "screen", false)
        d = cfg["data"]
        nd = (d["split_train_days"] + d["split_val_days"] +
              d["split_test_days"]) * 24
        selected = collect(1:min(nd, length(panel.timestamps)))
        restricted = restrict_panel(panel, selected)
        split = time_split(restricted, cfg)
        meta = (
            screen = false,
            order = "none",
            threshold_scope = "none",
            threshold = NaN,
            selected_indices = selected,
            selected_train = selected[split.train],
            selected_val = selected[split.val],
            selected_test = selected[split.test],
            source_split = split,
        )
        return (panel = restricted, split = split, meta = meta)
    end

    frac = Float64(sccfg["netload_top_frac"])
    order = String(get(sccfg, "screen_order", "split_then_screen"))
    netload = vec(sum(panel.Lf, dims = 2)) .- vec(sum(panel.Wf, dims = 2))

    if order == "split_then_screen"
        source_split = time_split(panel, cfg)
        threshold_scope = String(get(sccfg, "threshold_scope", "train"))
        threshold_scope == "train" ||
            error("split_then_screen requires threshold_scope=\"train\"")
        threshold = quantile(netload[source_split.train], 1 - frac)
        selected_train = [i for i in source_split.train if netload[i] >= threshold]
        selected_val = [i for i in source_split.val if netload[i] >= threshold]
        selected_test = [i for i in source_split.test if netload[i] >= threshold]
        all(x -> !isempty(x), (selected_train, selected_val, selected_test)) ||
            error("Stress screen produced an empty chronological split")

        selected = vcat(selected_train, selected_val, selected_test)
        restricted = restrict_panel(panel, selected)
        ntr, nva, nte = length(selected_train), length(selected_val), length(selected_test)
        split = (
            train = 1:ntr,
            val = (ntr + 1):(ntr + nva),
            test = (ntr + nva + 1):(ntr + nva + nte),
        )
        meta = (
            screen = true,
            order = order,
            threshold_scope = threshold_scope,
            threshold = threshold,
            selected_indices = selected,
            selected_train = selected_train,
            selected_val = selected_val,
            selected_test = selected_test,
            source_split = source_split,
        )
        return (panel = restricted, split = split, meta = meta)
    end
    error("Formal protocol requires data.scenario.screen_order=\"split_then_screen\"; got $order")
end

# ---------- 尾部样本选择 ----------
"Lexicographic merit dispatch used only for the model-independent forecast-point audit."
function _forecast_merit_dispatch(rc::RawCase, netload::Float64)
    gd = Cases.gendata(rc)
    online = findall(gd.status)
    isempty(online) && error("forecast-point dispatch has no online generator")
    g = zeros(length(gd.pmax))
    g[online] .= gd.pmin[online]
    minimum_generation = sum(g)
    maximum_generation = sum(gd.pmax[online])
    tolerance = 1e-8 * max(1.0, maximum_generation)
    netload >= minimum_generation - tolerance ||
        error("forecast net load $netload is below aggregate online Pmin $minimum_generation")
    netload <= maximum_generation + tolerance ||
        error("forecast net load $netload exceeds aggregate online Pmax $maximum_generation")
    remaining = clamp(netload - minimum_generation, 0.0,
                      maximum_generation - minimum_generation)
    merit = sort(online; by = j -> (gd.cost[j], j), alg = Base.Sort.MergeSort)
    for j in merit
        add = min(remaining, gd.pmax[j] - gd.pmin[j])
        g[j] += add
        remaining -= add
        remaining <= tolerance && break
    end
    abs(sum(g) - netload) <= tolerance ||
        error("forecast-point merit dispatch did not balance the forecast net load")
    return g, gd
end

function _forecast_tail_metrics(panel::Panel, rc::RawCase, idx)
    PTDF, busids, busidx, Fmax, _, _, online_lines = build_ptdf(rc)
    N = length(idx)
    forecast_netload = vec(sum(panel.Lf[idx, :], dims = 2)) .-
                       vec(sum(panel.Wf[idx, :], dims = 2))
    available_wind = vec(sum(panel.Wact[idx, :], dims = 2))
    # DataPipeline stores injection error (wind error minus load error); the
    # paper's net-load error has the opposite sign.
    omega_paper = .-panel.Ω[idx]
    line_loading = zeros(N)
    reserve_pressure = zeros(N)
    finite_lines = findall(online_lines .& isfinite.(Fmax))
    for tloc in 1:N
        g, gd = _forecast_merit_dispatch(rc, forecast_netload[tloc])
        injection = zeros(length(busids))
        for j in eachindex(g)
            injection[busidx[gd.bus[j]]] += g[j]
        end
        for (j, bus) in enumerate(panel.wind_buses)
            injection[busidx[bus]] += panel.Wf[idx[tloc], j]
        end
        for (j, bus) in enumerate(panel.load_buses)
            injection[busidx[bus]] -= panel.Lf[idx[tloc], j]
        end
        abs(sum(injection)) <= 1e-7 * max(1.0, abs(forecast_netload[tloc])) ||
            error("forecast-point injection is not balanced")
        line_loading[tloc] = isempty(finite_lines) ? 0.0 :
            maximum(abs((PTDF * injection)[line]) / Fmax[line] for line in finite_lines)
        online_gens = findall(gd.status)
        up_headroom = sum(gd.pmax[j] - g[j] for j in online_gens)
        down_headroom = sum(g[j] - gd.pmin[j] for j in online_gens)
        up_ratio = omega_paper[tloc] > 0 ?
            omega_paper[tloc] / max(up_headroom, eps(Float64)) : 0.0
        down_ratio = omega_paper[tloc] < 0 ?
            -omega_paper[tloc] / max(down_headroom, eps(Float64)) : 0.0
        reserve_pressure[tloc] = max(up_ratio, down_ratio)
    end
    return (omega_paper = omega_paper, forecast_netload = forecast_netload,
            available_wind = available_wind, line_loading = line_loading,
            reserve_pressure = reserve_pressure)
end

_descending_ranking(values) = sortperm(eachindex(values);
    by = i -> (-Float64(values[i]), Int(i)), alg = Base.Sort.MergeSort)
_ascending_ranking(values) = sortperm(eachindex(values);
    by = i -> (Float64(values[i]), Int(i)), alg = Base.Sort.MergeSort)

function _tail_rankings(panel::Panel, rc::RawCase, idx)
    metrics = _forecast_tail_metrics(panel, rc, idx)
    return Dict(
        "absOmega" => _descending_ranking(abs.(metrics.omega_paper)),
        "positiveOmega" => _descending_ranking(metrics.omega_paper),
        "negativeOmega" => _ascending_ranking(metrics.omega_paper),
        "forecastNetLoad" => _descending_ranking(metrics.forecast_netload),
        "availableWind" => _ascending_ranking(metrics.available_wind),
        "forecastLineLoading" => _descending_ranking(metrics.line_loading),
        "reserveResponseStress" => _descending_ranking(metrics.reserve_pressure),
    )
end

function tail_indices(panel::Panel, rc::RawCase, idx, criteria, n_tail)
    requested = Tuple(String.(criteria))
    requested == FORMAL_TAIL_CRITERIA ||
        error("tail criteria must equal the frozen formal order $(collect(FORMAL_TAIL_CRITERIA)); got $(collect(requested))")
    length(idx) >= n_tail || error("training sample has fewer rows than n_tail=$n_tail")
    rankings = _tail_rankings(panel, rc, idx)
    picks = Int[]
    # First pass: the semantic head of every stream, with stable uniqueness.
    for criterion in FORMAL_TAIL_CRITERIA
        candidate = first(rankings[criterion])
        candidate in picks || push!(picks, candidate)
    end
    # Cross-metric overlap can leave fewer than six unique heads. Continue the
    # same frozen ranking streams in round-robin order; never switch to a
    # performance-driven or single-metric fill rule.
    rank = 2
    while length(picks) < n_tail
        added = false
        for criterion in FORMAL_TAIL_CRITERIA
            ranking = rankings[criterion]
            rank > length(ranking) && continue
            candidate = ranking[rank]
            if !(candidate in picks)
                push!(picks, candidate)
                added = true
                length(picks) == n_tail && break
            end
        end
        rank += 1
        added || rank <= length(idx) + 1 || error("could not fill the frozen tail set")
    end
    return picks[1:n_tail]   # local positions within idx
end

# ---------- 尾部保留加权 k-means 压缩 ----------
# 关键口径(论文§净注入误差, 第165行): δ 无法唯一恢复实际负荷, 必须单独保留负荷样本。
# 因此代表样本携带 *负荷误差分量* eL 与 *风电误差分量* eW(均为相对预测的偏差),
# 模型在每个快照 t 用 实际量 = 预测量(t) + 误差分量 重建。δ̄ 由分量映射导出。
function compress_scenarios(panel::Panel, rc::RawCase, idx, cfg::Dict)
    rng = StableRNG(cfg["run"]["seed"] + 7)
    cc  = cfg["data"]["compress"]
    K   = cc["K"]; n_tail = cc["n_tail"]
    N   = length(idx)
    _, busids, busidx, _ = build_ptdf(rc)
    nbus = length(busids)
    nL = length(panel.load_buses); nW = length(panel.wind_buses)
    p_hat = cc["weighted"] ? fill(1/N, N) : fill(1/N, N)   # 默认等权; 可扩展

    # 误差分量(偏差): eL = l_act - l_f, eW = w_act - w_f
    eL = panel.Lact[idx, :] .- panel.Lf[idx, :]    # N × nL
    eW = panel.Wact[idx, :] .- panel.Wf[idx, :]    # N × nW

    tail_loc = tail_indices(panel, rc, idx, cc["tail_criteria"], n_tail)
    body_loc = setdiff(1:N, tail_loc)
    kbody = K - length(tail_loc)
    @assert kbody >= 1 "K 太小, 容不下尾部样本"

    # 在 *联合分量空间* [eL eW] 标准化后聚类(保留负荷/风电分解, 服务 q^{S+} 与弃风审计)
    F = hcat(eL, eW)[body_loc, :]
    μ = vec(mean(F, dims=1)); s = vec(std(F, dims=1)); s[s .== 0] .= 1
    Z = ((F .- μ') ./ s')'
    R = kmeans(Z, kbody; rng=rng)
    asg = R.assignments

    eLbar = zeros(K, nL); eWbar = zeros(K, nW)
    δbar  = zeros(K, nbus); π = zeros(K); is_tail = falses(K)
    assigned_center = zeros(Int, N)

    function setδ!(r, eLr, eWr)
        for (j, b) in enumerate(panel.wind_buses); δbar[r, busidx[b]] += eWr[j]; end
        for (j, b) in enumerate(panel.load_buses); δbar[r, busidx[b]] -= eLr[j]; end
    end

    for k in 1:kbody
        members = body_loc[findall(==(k), asg)]
        w = p_hat[members]; wsum = sum(w)
        eLbar[k, :] = vec(sum(eL[members, :] .* w, dims=1) ./ max(wsum, eps()))
        eWbar[k, :] = vec(sum(eW[members, :] .* w, dims=1) ./ max(wsum, eps()))
        setδ!(k, eLbar[k, :], eWbar[k, :]); π[k] = wsum
        assigned_center[members] .= k
    end
    for (j, tl) in enumerate(tail_loc)
        r = kbody + j
        eLbar[r, :] = eL[tl, :]; eWbar[r, :] = eW[tl, :]
        setδ!(r, eLbar[r, :], eWbar[r, :]); π[r] = p_hat[tl]; is_tail[r] = true
        assigned_center[tl] = r
    end
    π ./= sum(π)
    # Preserve raw-training distribution summaries for literature baselines.
    # Representative centers alone omit within-cluster covariance and do not
    # define a statistically defensible support box.
    delta_raw = panel.δ[idx, :]
    raw_mean = vec(sum(delta_raw .* reshape(p_hat, :, 1), dims = 1))
    centered = delta_raw .- raw_mean'
    raw_cov = centered' * (centered .* reshape(p_hat, :, 1))
    raw_lower = vec(minimum(delta_raw, dims = 1))
    raw_upper = vec(maximum(delta_raw, dims = 1))
    all(>(0), assigned_center) || error("scenario compression left an unassigned raw sample")
    # Certified transport radius: send each body point to its actual k-means
    # cluster center and each retained tail point to its singleton center.  The
    # induced column masses are exactly π, hence this is a feasible coupling
    # from the raw empirical distribution to the compressed distribution and
    # its L1 cost is a valid upper bound on W1.  The historical nearest-L1
    # expression did not preserve π and could be strictly below W1.
    compression_l1 = sum(p_hat[i] * sum(
        abs(delta_raw[i, b] - δbar[assigned_center[i], b]) for b in 1:nbus)
        for i in 1:N)
    return (δbar=δbar, eLbar=eLbar, eWbar=eWbar, π=π, is_tail=is_tail,
            busids=busids, load_buses=panel.load_buses, wind_buses=panel.wind_buses,
            raw_n=N, raw_mean=raw_mean, raw_cov=raw_cov,
            raw_lower=raw_lower, raw_upper=raw_upper,
            compression_l1=compression_l1, tail_local_indices=tail_loc,
            tail_source_indices=Int.(idx[tail_loc]))
end

# ---------- 顶层: 跑完整数据管线并落盘 ----------
function run_pipeline(system::String, cfg::Dict; case_dir::String, out_dir::String)
    mkpath(out_dir)
    rc = parse_matpower(_case_file(system, case_dir); name=system)
    panel = build_panel(system, cfg, case_dir)          # 全年面板
    panel = build_delta!(panel, rc)
    # --- 先按完整时间轴划分，再用训练期预测净负荷阈值筛选 ---
    T_full = length(panel.timestamps)
    full_timestamps = copy(panel.timestamps)
    prepared = apply_screen_and_split(panel, cfg)
    panel = prepared.panel
    sp = prepared.split
    screen_meta = prepared.meta
    if screen_meta.screen
        @info "应力筛选 $system: 全年 $T_full h → 低风高负荷 $(length(panel.timestamps)) h" order=screen_meta.order threshold_scope=screen_meta.threshold_scope threshold=screen_meta.threshold
    end
    @info "划分 $system: train $(length(sp.train)) / val $(length(sp.val)) / test $(length(sp.test)) h"
    sc = compress_scenarios(panel, rc, sp.train, cfg)

    # 落盘: 代表样本文件 {δ̄, eL̄, eW̄, π, is_tail}
    # (δ 仅供检查; 模型用 eL/eW 分量 + 各快照预测点重建实际量)
    bus = sc.busids
    dfc = DataFrame(sc.δbar, ["delta_bus$(b)" for b in bus])
    for (j, b) in enumerate(sc.load_buses); dfc[!, "loaderr_bus$(b)"] = sc.eLbar[:, j]; end
    for (j, b) in enumerate(sc.wind_buses); dfc[!, "winderr_bus$(b)"] = sc.eWbar[:, j]; end
    dfc.pi = sc.π; dfc.is_tail = sc.is_tail
    dfc.tail_source_index = Vector{Union{Missing,Int}}(missing, nrow(dfc))
    dfc.tail_timestamp = Vector{Union{Missing,String}}(missing, nrow(dfc))
    tail_rows = findall(sc.is_tail)
    for (row, source_index) in zip(tail_rows, sc.tail_source_indices)
        dfc.tail_source_index[row] = source_index
        dfc.tail_timestamp[row] = string(panel.timestamps[source_index])
    end
    CSV.write(joinpath(out_dir, "scenarios_K$(cfg["data"]["compress"]["K"]).csv"), dfc)

    # 落盘: 全小时面板 (审计/逐快照求解用): 预测/实际 负荷与风电 + δ
    pdf = DataFrame(timestamp = panel.timestamps)
    for (j, b) in enumerate(panel.load_buses)
        pdf[!, "lf_bus$(b)"]   = panel.Lf[:, j]
        pdf[!, "lact_bus$(b)"] = panel.Lact[:, j]
    end
    for (j, b) in enumerate(panel.wind_buses)
        pdf[!, "wf_bus$(b)"]   = panel.Wf[:, j]
        pdf[!, "wact_bus$(b)"] = panel.Wact[:, j]
    end
    for (j, b) in enumerate(bus); pdf[!, "delta_bus$(b)"] = panel.δ[:, j]; end
    Arrow.write(joinpath(out_dir, "panel.arrow"), pdf)

    open(joinpath(out_dir, "split.toml"), "w") do io
        println(io, "train = [$(first(sp.train)), $(last(sp.train))]")
        println(io, "val   = [$(first(sp.val)), $(last(sp.val))]")
        println(io, "test  = [$(first(sp.test)), $(last(sp.test))]")
        println(io, "load_buses = [", join(panel.load_buses, ", "), "]")
        println(io, "wind_buses = [", join(panel.wind_buses, ", "), "]")
        println(io, "screen_order = \"$(screen_meta.order)\"")
        println(io, "threshold_scope = \"$(screen_meta.threshold_scope)\"")
        println(io, "netload_threshold = $(screen_meta.threshold)")
        println(io, "source_train = [$(first(screen_meta.source_split.train)), $(last(screen_meta.source_split.train))]")
        println(io, "source_val = [$(first(screen_meta.source_split.val)), $(last(screen_meta.source_split.val))]")
        println(io, "source_test = [$(first(screen_meta.source_split.test)), $(last(screen_meta.source_split.test))]")
        println(io, "source_train_hours = $(length(screen_meta.source_split.train))")
        println(io, "source_val_hours = $(length(screen_meta.source_split.val))")
        println(io, "source_test_hours = $(length(screen_meta.source_split.test))")
        println(io, "selected_train_hours = $(length(screen_meta.selected_train))")
        println(io, "selected_val_hours = $(length(screen_meta.selected_val))")
        println(io, "selected_test_hours = $(length(screen_meta.selected_test))")
        println(io, "source_train_timestamp = [\"$(full_timestamps[first(screen_meta.source_split.train)])\", \"$(full_timestamps[last(screen_meta.source_split.train)])\"]")
        println(io, "source_val_timestamp = [\"$(full_timestamps[first(screen_meta.source_split.val)])\", \"$(full_timestamps[last(screen_meta.source_split.val)])\"]")
        println(io, "source_test_timestamp = [\"$(full_timestamps[first(screen_meta.source_split.test)])\", \"$(full_timestamps[last(screen_meta.source_split.test)])\"]")
        println(io, "tail_rule = \"seven semantic rankings, stable unique, fixed round-robin fill\"")
        println(io, "tail_criteria = [\"", join(FORMAL_TAIL_CRITERIA, "\", \""), "\"]")
    end
    gen_keep = lowercase(system) in ("rts_gmlc", "gmlc") ?
               _gmlc_conv_mask(case_dir) : nothing
    return (pipeline_version=PIPELINE_VERSION,
            panel=panel, split=sp, screen_meta=screen_meta,
            scenarios=sc, case=rc, gen_keep=gen_keep)
end

function _case_file(system, case_dir)
    sysl = lowercase(system)
    sysl == "rbts"  && return joinpath(case_dir, "RBTS.txt")
    sysl == "rts79" && return joinpath(case_dir, "RTS79_case.txt")
    (sysl == "rts_gmlc" || sysl == "gmlc") &&
        return joinpath(case_dir, "RTS_Data", "FormattedData", "MATPOWER", "RTS_GMLC.m")
    error("未配置算例文件: $system")
end

end # module
