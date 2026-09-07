"""Runtime compatibility patches for Clarabel.jl on the RTS-79 final model.

Clarabel 0.11.1 implements `_scale_values_KKT!` by advanced-indexed broadcast.
Julia 1.12 therefore creates a temporary copy of every SOC-map slice before
writing it back.  On the large RTS-79 final model, the SYSTEM job failed in
that allocation with `ReadOnlyMemoryError()` before its first calibration row.

The same Julia/Clarabel combination also materializes temporary vectors in the
second-order-cone (SOC) scaling update: the calculation of `K.λ[2:end]` uses
two broadcast products, an array addition, and a sliced multiplication.  The
long RTS79 calibration reached that method and failed before producing any
calibration output.  The SOC replacement below preserves the upstream scalar
operation order for every entry while writing directly to the preallocated
cone work vectors.

The method below performs the same entrywise multiplication directly on the
KKT nonzero vector.  It changes no decision variable, constraint, loss,
target, candidate, state, solver tolerance, or selection rule.  Its equality
to the unpatched implementation was verified on frozen RTS-79 validation
snapshot 1890; objective and all reported residual/gap values were identical.
"""
module ClarabelKKTMemoryPatch

using Clarabel
import MathOptInterface as MOI
using SparseArrays

export PATCH_ID, installed, soc_scaling_installed, triplet_preallocation_installed

const PATCH_ID = "clarabel-kkt-soc-triplet-preallocation-v3"

"""In-place equivalent of `@. KKT.nzval[index] *= scale` for Clarabel's
structurally unique SOC expansion-map indices."""
function Clarabel._scale_values_KKT!(KKT::SparseArrays.SparseMatrixCSC{T,Ti},
                                     index::AbstractVector{Ti}, scale::T) where {T,Ti}
    @inbounds for nzindex in index
        KKT.nzval[nzindex] *= scale
    end
    return nothing
end

installed() = hasmethod(Clarabel._scale_values_KKT!,
                        Tuple{SparseArrays.SparseMatrixCSC{Float64,Int}, Vector{Int}, Float64})

"""Allocation-free, entrywise equivalent of Clarabel 0.11.1's SOC scaling
update.  The original method is in `coneops_socone.jl`.  This implementation
does not change the cone map, scaling strategy, tolerances, or termination
logic; it only replaces temporary array expressions with writes to `K.w` and
`K.λ`, which are already the upstream solver workspaces."""
function Clarabel.update_scaling!(K::Clarabel.SecondOrderCone{T},
                                  s::AbstractVector{T},
                                  z::AbstractVector{T},
                                  μ::T,
                                  scaling_strategy::Clarabel.ScalingStrategy) where {T}
    zscale = Clarabel._sqrt_soc_residual(z)
    sscale = Clarabel._sqrt_soc_residual(s)
    if iszero(zscale) || iszero(sscale)
        return false
    end

    # Clarabel names the scalar leading scale `η` and the vector scaling point
    # `λ`; keep this internal convention exactly.
    K.η = sqrt(sscale / zscale)

    # Upstream: `w .= s ./ sscale`; then its first coordinate receives the
    # opposite-sign dual contribution.  These assignments preserve the same
    # elementwise arithmetic without allocating an intermediate broadcast.
    w = K.w
    @inbounds for i in eachindex(w)
        w[i] = s[i] / sscale
    end
    w[1] += z[1] / zscale
    @inbounds for i in 2:length(w)
        w[i] -= z[i] / zscale
    end

    wscale = Clarabel._sqrt_soc_residual(w)
    iszero(wscale) && return false
    @inbounds for i in eachindex(w)
        w[i] /= wscale
    end

    w1sq = Clarabel.sumsq(@view w[2:end])
    w[1] = sqrt(1 + w1sq)

    # Upstream first forms `a .* s1 + b .* z1`, scales the resulting slice,
    # and finally scales all η entries.  Retaining that scalar order avoids a
    # mathematical or floating-point-reassociation change.
    λhalf = 0.5 * wscale
    λ = K.λ
    λ[1] = λhalf
    a = (λhalf + z[1] / zscale) / sscale
    b = (λhalf + s[1] / sscale) / zscale
    η_scale = inv(s[1] / sscale + z[1] / zscale + 2 * λhalf)
    global_scale = sqrt(sscale * zscale)
    @inbounds for i in 2:length(λ)
        λ[i] = a * s[i] + b * z[i]
        λ[i] *= η_scale
        λ[i] *= global_scale
    end
    λ[1] *= global_scale

    if Clarabel.is_sparse_expandable(K)
        sparse_data = K.sparse_data
        two_w1 = 2 * w[1]
        wsq = w[1] * w[1] + w1sq
        wsqinv = 1 / wsq
        sparse_data.d = wsqinv / 2

        u0 = sqrt(wsq - sparse_data.d)
        u1 = two_w1 / u0
        v0 = zero(T)
        v1 = sqrt(2 * (2 + wsqinv) / (2 * wsq - wsqinv))
        sparse_data.u[1] = u0
        sparse_data.v[1] = v0
        @inbounds for i in 2:length(w)
            sparse_data.u[i] = u1 * w[i]
            sparse_data.v[i] = v1 * w[i]
        end
    end

    return true
end

soc_scaling_installed() = hasmethod(
    Clarabel.update_scaling!,
    Tuple{Clarabel.SecondOrderCone{Float64}, AbstractVector{Float64},
          AbstractVector{Float64}, Float64, Clarabel.ScalingStrategy},
)

"""Count, without altering, the exact entries that Clarabel's MOI wrapper
will append to its constraint triplet.  The supported final-model constraints
reach the wrapper as vector-affine functions or vectors of variables.  This
first pass merely reads those functions so that the upstream triplet vectors
can be allocated once at their final length rather than repeatedly grown.
"""
function _constraint_triplet_length(src::MOI.ModelLike, constraint_types)
    nterms = 0
    for (F, S) in constraint_types
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
            f = MOI.get(src, MOI.ConstraintFunction(), ci)
            if f isa MOI.VectorAffineFunction
                nterms += length(f.terms)
            elseif f isa MOI.VectorOfVariables
                nterms += length(f.variables)
            else
                error("Clarabel triplet preallocation does not recognize constraint function $(typeof(f))")
            end
        end
    end
    return nterms
end

"""Allocation-equivalent replacement for Clarabel 0.11.1's
`MOIwrapper.process_constraints`.

The upstream method starts `I`, `J`, and `V` as empty vectors and grows them
one coefficient at a time.  On the RTS-79 final model, the last growth needs
both the old and new multi-million-entry arrays concurrently and can fail
before the same sparse matrix is assembled.  This method performs one
read-only count pass, reserves that exact capacity, then invokes the same
upstream constraint push routines in the original order.  It does not change
the constraint matrix, right-hand side, cone sequence, variables, objective,
or solver settings.
"""
function Clarabel.MOIwrapper.process_constraints(
    dest::Clarabel.MOIwrapper.Optimizer{T},
    src::MOI.ModelLike,
    idxmap,
) where {T}
    rowranges = dest.rowranges
    m = mapreduce(length, +, values(rowranges), init = 0)
    b = zeros(T, m)
    constraint_types = MOI.get(src, MOI.ListOfConstraintTypesPresent())
    nterms = _constraint_triplet_length(src, constraint_types)
    I = sizehint!(Clarabel.DefaultInt[], nterms)
    J = sizehint!(Clarabel.DefaultInt[], nterms)
    V = sizehint!(T[], nterms)
    cone_spec = Clarabel.SupportedCone[]

    for (F, S) in constraint_types
        Clarabel.MOIwrapper.push_constraint!(
            (I, J, V), b, cone_spec, src, idxmap, rowranges, F, S,
        )
    end

    V .= -V
    n = MOI.get(src, MOI.NumberOfVariables())
    A = SparseArrays.sparse(I, J, V, m, n)
    return (A, b, cone_spec)
end

triplet_preallocation_installed() = hasmethod(
    Clarabel.MOIwrapper.process_constraints,
    Tuple{Clarabel.MOIwrapper.Optimizer{Float64}, MOI.ModelLike, Any},
)

end # module
