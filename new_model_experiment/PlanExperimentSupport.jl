"""
PlanExperimentSupport

Auditing, provenance, and statistical helpers for the comprehensive TPS
experiment protocol. This module adds no optimization variable, constraint,
loss, target, or objective term. It only evaluates a fixed decision with the
same physical equations used by `DRCCExp.Audit` and records reproducible
evidence.
"""
module PlanExperimentSupport

using Dates
using Random
using SHA
using Statistics

using Main.DRCCExp
using Main.DRCCExp: Audit

export directional_audit, final_directional_audit, reference_quality_accepted,
       sha256_file, provenance_record, wilson_interval, edns_precision,
       paired_bootstrap_mean_difference, conditional_mean,
       execution_source_provenance, assert_execution_source_provenance

"SHA-256 of a concrete input artifact, suitable for a run manifest."
sha256_file(path::AbstractString) = bytes2hex(sha256(read(path)))

"Best-effort source revision without making a repository a hidden prerequisite."
function _git_revision(root::AbstractString)
    try
        return readchomp(`git -C $root rev-parse HEAD`)
    catch
        return "unavailable_not_git_repository"
    end
end

"Deterministic hash receipt for every file that can affect a formal execution."
function execution_source_provenance(root::AbstractString)
    root_abs = normpath(abspath(root))
    paths = String[]
    for fixed in (joinpath(root_abs, "Project.toml"),
                  joinpath(root_abs, "Manifest.toml"),
                  joinpath(root_abs, "config", "experiment.toml"))
        isfile(fixed) || error("formal execution source is missing: $fixed")
        push!(paths, fixed)
    end
    for source_dir in (joinpath(root_abs, "src"),
                       joinpath(root_abs, "new_model_experiment"))
        isdir(source_dir) || error("formal execution source directory is missing: $source_dir")
        for (dir, subdirs, files) in walkdir(source_dir)
            filter!(name -> name != "results", subdirs)
            for name in files
                extension = lowercase(splitext(name)[2])
                extension in (".jl", ".ps1") || continue
                push!(paths, normpath(joinpath(dir, name)))
            end
        end
    end
    sort!(unique!(paths))
    hashes = Dict{String,String}()
    for path in paths
        relative = replace(relpath(path, root_abs), '\\' => '/')
        hashes[relative] = sha256_file(path)
    end
    receipt = join(("$path=$(hashes[path])" for path in sort(collect(keys(hashes)))), "\n")
    return (tree_sha256 = bytes2hex(sha256(receipt)), files_sha256 = hashes)
end

"Fail closed when the execution sources no longer match a frozen lock."
function assert_execution_source_provenance(record::AbstractDict;
                                            root::AbstractString)
    haskey(record, "source_tree_sha256") ||
        error("formal lock lacks source_tree_sha256; regenerate it after freezing the corrected source tree")
    current = execution_source_provenance(root)
    String(record["source_tree_sha256"]) == current.tree_sha256 ||
        error("formal execution source tree differs from the frozen lock")
    haskey(record, "source_files_sha256") || error("formal lock lacks source_files_sha256")
    locked = Dict(String(k) => String(v) for (k, v) in record["source_files_sha256"])
    locked == current.files_sha256 || error("formal execution source-file receipt differs from the frozen lock")
    return current
end

"Common provenance fields that must accompany every calibration and NSMC file."
function provenance_record(; root::AbstractString, protocol_path::AbstractString,
                           config_path::AbstractString, data_paths::AbstractVector{<:AbstractString},
                           scenario_id::AbstractString, seed::Integer)
    source = execution_source_provenance(root)
    return Dict{String,Any}(
        "scenario_id" => scenario_id,
        "seed" => Int(seed),
        "created_utc" => string(Dates.now(Dates.UTC)),
        "git_commit" => _git_revision(root),
        "protocol_path" => abspath(protocol_path),
        "protocol_sha256" => sha256_file(protocol_path),
        "config_path" => abspath(config_path),
        "config_sha256" => sha256_file(config_path),
        "data_paths" => abspath.(data_paths),
        "data_sha256" => Dict(abspath(path) => sha256_file(path) for path in data_paths),
        "source_tree_sha256" => source.tree_sha256,
        "source_files_sha256" => source.files_sha256,
    )
end

"""Return six directional physical-event flags and raw-unit amplitudes.

The formulas intentionally mirror `Audit.audit_snapshot`: the realized state
uses actual `delta` and actual load/wind through `snap`; only the decision and
its declared affine recourse are supplied. Aggregate flags are cross-checked
against the shared audit routine so a reporting-only change cannot silently
alter a reliability event definition.
"""
function directional_audit(cd, x, snap;
                           beta::AbstractVector = cd.βbar,
                           rhoG::Real = 1.0,
                           alphaS::Union{Nothing,AbstractVector} = nothing,
                           tol_mw::Float64 = Audit.DEFAULT_AUDIT_TOL_MW)
    nG, nD, nE = cd.nG, cd.nD, cd.nE
    length(beta) == nG || error("directional audit beta length mismatch")
    α = isnothing(alphaS) ? Float64[] : Float64.(alphaS)
    !isempty(α) && length(α) != nD && error("directional audit alphaS length mismatch")
    gen_rec = isfinite(rhoG)
    shed_rec = !isempty(α)

    base_x = (g = x.g, rU = x.rU, rD = x.rD, snom = x.snom, cW = x.cW,
              beta = beta, ρG = Float64(rhoG), αS = α)
    aggregate = Audit.audit_snapshot(cd, base_x, snap; tol_mw = tol_mw, βbar = beta)

    Ω = sum(snap.δ)
    f0 = [sum(cd.MG[ell, j] * x.g[j] for j in 1:nG) +
          sum(cd.MW[ell, w] * (snap.wf[w] - x.cW[w]) for w in 1:cd.nW) -
          sum(cd.MD[ell, d] * (snap.lf[d] - x.snom[d]) for d in 1:nD)
          for ell in 1:nE]
    ptdf_delta = cd.PTDF * snap.δ
    mg_beta = [sum(cd.MG[ell, j] * beta[j] for j in 1:nG) for ell in 1:nE]
    md_alpha = shed_rec ?
        [sum(cd.MD[ell, d] * α[d] for d in 1:nD) for ell in 1:nE] : zeros(nE)
    realized_flow = [f0[ell] + ptdf_delta[ell] -
                     (gen_rec ? Ω * Float64(rhoG) * mg_beta[ell] : 0.0) -
                     (shed_rec ? Ω * md_alpha[ell] : 0.0) for ell in 1:nE]
    realized_shed = [x.snom[d] - (shed_rec ? Ω * α[d] : 0.0) for d in 1:nD]

    ru_excess = gen_rec ? maximum([-Ω * Float64(rhoG) * beta[j] - x.rU[j]
                                    for j in 1:nG]; init = -Inf) : -Inf
    rd_excess = gen_rec ? maximum([Ω * Float64(rhoG) * beta[j] - x.rD[j]
                                    for j in 1:nG]; init = -Inf) : -Inf
    finite_lines = [ell for ell in 1:nE if isfinite(cd.F_max[ell])]
    fp_excess = isempty(finite_lines) ? -Inf :
        maximum(realized_flow[ell] - cd.F_max[ell] for ell in finite_lines)
    fm_excess = isempty(finite_lines) ? -Inf :
        maximum(-realized_flow[ell] - cd.F_max[ell] for ell in finite_lines)
    sm_excess = maximum(-realized_shed[d] for d in 1:nD)
    sp_excess = maximum(realized_shed[d] - snap.lact[d] for d in 1:nD)
    reserve_up = ru_excess > tol_mw
    reserve_down = rd_excess > tol_mw
    line_pos = any(realized_flow[ell] - cd.F_max[ell] > 1e-6 * cd.F_max[ell]
                   for ell in finite_lines)
    line_neg = any(-realized_flow[ell] - cd.F_max[ell] > 1e-6 * cd.F_max[ell]
                   for ell in finite_lines)
    shed_low = sm_excess > tol_mw
    shed_high = sp_excess > tol_mw
    aggregate.reserve_viol == (reserve_up || reserve_down) ||
        error("directional reserve audit disagrees with shared aggregate audit")
    aggregate.line_viol == (line_pos || line_neg) ||
        error("directional line audit disagrees with shared aggregate audit")
    aggregate.shedbound_viol == (shed_low || shed_high) ||
        error("directional shed-bound audit disagrees with shared aggregate audit")

    return merge(aggregate, (
        RU_event = reserve_up, RD_event = reserve_down,
        Fp_event = line_pos, Fm_event = line_neg,
        Sm_event = shed_low, Sp_event = shed_high,
        RU_excess_mw = max(ru_excess, 0.0),
        RD_excess_mw = max(rd_excess, 0.0),
        Fp_excess_mw = max(fp_excess, 0.0),
        Fm_excess_mw = max(fm_excess, 0.0),
        Sm_excess_mw = max(sm_excess, 0.0),
        Sp_excess_mw = max(sp_excess, 0.0),
        realized_shed_mw = aggregate.shed_mw,
        realized_max_flow_ratio = aggregate.max_ratio,
    ))
end

"Directional audit adapter for the exact final seven-target result."
function final_directional_audit(cd, result, snap; tol_mw::Float64 = Audit.DEFAULT_AUDIT_TOL_MW)
    return directional_audit(cd, result.x, snap; beta = result.x.beta,
                             rhoG = 1.0, alphaS = result.x.alphaS, tol_mw = tol_mw)
end

"Uniform status/residual/gap acceptance for the seven reference targets."
function reference_quality_accepted(cal, acceptance::AbstractDict)
    residual = Float64(get(acceptance, "residual", 1e-5))
    gap_abs = Float64(get(acceptance, "gap_abs", 5e-5))
    gap_rel = Float64(get(acceptance, "gap_rel", 1e-5))
    n = 7
    hasproperty(cal, :Z0c) && Float64(cal.Z0c) > 0.0 || return false
    sign_tolerance = Float64(get(acceptance, "reference_nonpositive_tolerance", 0.0))
    sign_tolerance >= 0.0 || return false
    hasproperty(cal, :Z0m) || return false
    length(cal.Z0m) == 6 && all(z -> isfinite(z) && z <= sign_tolerance, cal.Z0m) ||
        return false
    length(cal.reference_status) == n || return false
    for field in (:reference_primal_residual, :reference_dual_residual,
                  :reference_gap_abs, :reference_gap_rel)
        length(getproperty(cal, field)) == n || return false
    end
    for i in 1:n
        string(cal.reference_status[i]) in ("OPTIMAL", "ALMOST_OPTIMAL") || return false
        primal, dual = cal.reference_primal_residual[i], cal.reference_dual_residual[i]
        ga, gr = cal.reference_gap_abs[i], cal.reference_gap_rel[i]
        all(isfinite, (primal, dual, ga, gr)) || return false
        primal <= residual && dual <= residual || return false
        ga <= gap_abs || gr <= gap_rel || return false
    end
    return true
end

"Wilson score interval for a binomial event probability."
function wilson_interval(k::Integer, n::Integer; z::Float64 = 1.959963984540054)
    0 <= k <= n || error("Wilson interval requires 0 <= k <= n")
    n > 0 || return (lower = NaN, upper = NaN)
    p = k / n
    denom = 1 + z^2 / n
    center = (p + z^2 / (2n)) / denom
    radius = z * sqrt(p * (1 - p) / n + z^2 / (4n^2)) / denom
    return (lower = max(0.0, center - radius), upper = min(1.0, center + radius))
end

"Mean, standard error, and COV of EDNS contributions on a fixed denominator."
function edns_precision(values::AbstractVector{<:Real})
    n = length(values)
    n == 0 && return (mean = NaN, sd = NaN, se = NaN, cov = NaN)
    mu = mean(values)
    n == 1 && return (mean = mu, sd = NaN, se = NaN, cov = NaN)
    sd = std(values; corrected = true)
    se = sd / sqrt(n)
    return (mean = mu, sd = sd, se = se, cov = abs(mu) > 0 ? se / abs(mu) : Inf)
end

conditional_mean(values::AbstractVector{<:Real}; positive_tol::Float64 = Audit.DEFAULT_AUDIT_TOL_MW) =
    isempty(filter(>(positive_tol), values)) ? 0.0 : mean(filter(>(positive_tol), values))

"Paired nonparametric bootstrap for the mean difference `left - right`."
function paired_bootstrap_mean_difference(left::AbstractVector{<:Real}, right::AbstractVector{<:Real};
                                          B::Int = 2000, seed::Int = 20260728)
    length(left) == length(right) || error("paired bootstrap vectors must have equal length")
    n = length(left)
    n > 1 || return (estimate = n == 0 ? NaN : mean(left .- right), lower = NaN, upper = NaN,
                     B = B, n = n)
    B >= 100 || error("paired bootstrap B must be at least 100")
    diff = Float64.(left) .- Float64.(right)
    rng = MersenneTwister(seed)
    samples = Vector{Float64}(undef, B)
    for b in 1:B
        samples[b] = mean(diff[rand(rng, 1:n, n)])
    end
    sort!(samples)
    lo = samples[clamp(ceil(Int, 0.025 * B), 1, B)]
    hi = samples[clamp(floor(Int, 0.975 * B), 1, B)]
    return (estimate = mean(diff), lower = lo, upper = hi, B = B, n = n)
end

end # module
