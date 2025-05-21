
function split_merge_slow(o1, o2; normalize_spectrum=true, directional=false, new_dim=nothing, value=0)
    comm = commonind(o1, o2)
    local link_new
    if new_dim === nothing || new_dim == dim(comm)
        link_new = comm
    else
        link_new = Index(new_dim, tags=tags(comm))
    end
    l1 = uniqueinds(o1, comm)
    l2 = uniqueinds(o2, comm)
    
    A = svd(o1 * o2, l1; maxdim=dim(link_new))
    U, S, V = A
    if value != 0
        S.tensor.storage[dim(comm)+1:end] .= value
    end
    norm_ = 1
    if normalize_spectrum
        norm_ = norm(S)
        S ./= norm_
    end
    if directional
        S2 = itensor(S.tensor, (link_new, A.v))
        o1 = U * δ(A.u, link_new)
        o2 = V * S2
    else
        comm = commonind(o1, o2)
        S_sqrt = sqrt.(S)
        S1 = itensor(S_sqrt.tensor, (A.u, link_new))
        S2 = itensor(S_sqrt.tensor, (link_new, A.v))

        o1 = U * S1
        o2 = V * S2
    end
    return o1, o2, S.tensor.storage, norm_
end

inv_cutoff_func(x; cutoff=1e-6) = x < cutoff ? 0 : 1/x
isnan2(x) = sum(isnan.(x)) > 0
function split_merge(o1, o2; cutoff=1e-6, directional=false, normalize_spectrum=true, new_dim=nothing)
    # https://arxiv.org/pdf/1808.00680
    comm = commonind(o1, o2)
    M1 = o1 * conj(prime(o1, comm))
    M2 = o2 * conj(prime(o2, comm))
    
    M1_ =  reshape(M1.tensor.storage, dim(comm),dim(comm))
    D1, u1 = eigen(M1_)
    M2_ = reshape(M2.tensor.storage, dim(comm),dim(comm))
    D2, u2 = eigen(M2_)
    
    #@show D1
    #@show D2
    D1_sqrt, D2_sqrt = sqrt.(abs.(D1)), sqrt.(abs.(D2))
    # Stable inversion by only inverting the non-zero eigenvalues
    
    lambda_1 = reshape(D1_sqrt, :, 1) .* u1'
    lambda_2 = u2 .* reshape(D2_sqrt, 1, :)

    lambda_ = lambda_1 * lambda_2
    w1, S, w2 = svd(lambda_)

    norm_ = 1
    if normalize_spectrum
        norm_ = norm(S)
        S ./= norm_
    end
    D1_inv = inv_cutoff_func.(D1_sqrt; cutoff)
    x_inv = u1 .* reshape(D1_inv, 1, :)
    x_inv = x_inv * w1

    D2_inv = inv_cutoff_func.(D2_sqrt; cutoff)
    y_inv = w2' .* reshape(D2_inv, 1, :)
    y_inv = y_inv * u2'
    
    if directional
        x_ = x_inv .* reshape(abs.(S), 1, :)
        y_ = y_inv
    else
        x_ = x_inv .* reshape(sqrt.(abs.(S)), 1, :)
        y_ = y_inv .* reshape(sqrt.(abs.(S)), :, 1)
    end
    
    #@show diag(y_ * x_)
    
    x_ = itensor(x_, comm', comm)
    y_ = itensor(y_, comm, comm')
    o1n = apply(o1, x_)
    o2n = apply(o2, y_)

    #@show norm(o1n * o2n - o1 * o2)
    return o1n, o2n, S, norm_
end



function Smatrix(S, l)
    inv_S = QuantumNaturalfPEPS.inv_cutoff_func.(S; cutoff=1e-6)
    Sinv = itensor(diagm(inv_S), l, l')
    S = itensor(diagm(S), l, l')
    return S, Sinv
end

function get_peps_sqrtS(peps, Sx, Sy, i, j; exception=(-1,-1))
    Ss = []
    S_invs = []
    if 1 < i && !(exception == (i-1, j))
        l = commonind(peps[i-1, j], peps[i, j])
        S = Sx[i-1, j, :]
        S, Sinv = Smatrix(sqrt.(S), l)
        
        push!(Ss, S)
        push!(S_invs, Sinv)
    end
    
    if i < size(peps, 1) && !(exception == (i+1, j))
        l = commonind(peps[i+1, j], peps[i, j])
        S = Sx[i, j, :]
        S, Sinv = Smatrix(sqrt.(S), l)
        
        push!(Ss, S)
        push!(S_invs, Sinv)
    end
    
    if 1 < j && !(exception == (i, j-1))
        l = commonind(peps[i, j-1], peps[i, j])
        S = Sy[i, j-1, :]
        S, Sinv = Smatrix(sqrt.(S), l)
        
        push!(Ss, S)
        push!(S_invs, Sinv)
    end
    
    if j < size(peps, 2) && !(exception == (i, j+1))
        l = commonind(peps[i, j+1], peps[i, j])
        S = Sy[i, j, :]
        S, Sinv = Smatrix(sqrt.(S), l)
        
        push!(Ss, S)
        push!(S_invs, Sinv)
    end
    
    return Ss, S_invs
end

function ITensors.apply(A::ITensor, B::Vector)
    for i in 1:length(B)
        A = apply(A, B[i])
    end
    return A
end

function split_merge!(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; split_merge_=split_merge, new_dim=maxbonddim(peps),
                      top_bottom_direction=false, left_right_direction=false, vidal=false, kwargs...)
    Sx = ones(size(peps, 1)-1, size(peps, 2), new_dim)
    Sy = ones(size(peps, 1), size(peps, 2)-1, new_dim)
    lognorm = 0
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            if vidal
                Ssqrts_1, S_invs_1 = get_peps_sqrtS(peps, Sx, Sy, i, j; exception=(i+1, j))
                Ssqrts_2, S_invs_2 = get_peps_sqrtS(peps, Sx, Sy, i+1, j; exception=(i, j))
                peps[i, j] = apply(peps[i, j], Ssqrts_1)
                peps[i+1, j] = apply(peps[i+1, j], Ssqrts_2)
            end
            peps[i, j], peps[i+1, j], S, norm_ = split_merge_(peps[i, j], peps[i+1, j]; new_dim, directional=top_bottom_direction, kwargs...)
            lognorm += log(norm_)
            Sx[i, j, :] .= S
            if vidal
                peps[i, j] = apply(peps[i, j], S_invs_1)
                peps[i+1, j] = apply(peps[i+1, j], S_invs_2)
            end
        end
        if j < size(peps, 2)
            if vidal
                Ssqrts_1, S_invs_1 = get_peps_sqrtS(peps, Sx, Sy, i, j; exception=(i, j+1))
                Ssqrts_2, S_invs_2 = get_peps_sqrtS(peps, Sx, Sy, i, j+1; exception=(i, j))
                peps[i, j] = apply(peps[i, j], Ssqrts_1)
                peps[i, j+1] = apply(peps[i, j+1], Ssqrts_2)
            end
            peps[i, j], peps[i, j+1], S, norm_ = split_merge_(peps[i, j], peps[i, j+1]; new_dim, directional=left_right_direction, kwargs...)
            lognorm += log(norm_)
            Sy[i, j, :] .= S
            if vidal
                peps[i, j] = apply(peps[i, j], S_invs_1)
                peps[i, j+1] = apply(peps[i, j+1], S_invs_2)
            end
        end
    end
    if isa(peps, QuantumNaturalfPEPS.PEPS)
        peps.bond_dim = new_dim
    end
    return Sx, Sy, peps, lognorm
end

function super_orthonormalization!(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; k=1000, error=1e-8, verbose=false, kwargs...)
    # TODO: Use belief propagation to speed up the process https://arxiv.org/pdf/2306.17837v4#page=14.14
    Sx, Sy, peps, lognorm = split_merge!(peps; kwargs...)
    res = -1.
    for i in 2:k
        Sx2, Sy2, peps, lognorm_ = split_merge!(peps; kwargs...)
        lognorm += lognorm_
        res = mean(abs2, Sx2 .- Sx)
        res += mean(abs2, Sy2 .- Sy)
        Sx, Sy = Sx2, Sy2
        if verbose
            @info "iter $i: $res"
            flush(stdout)
            flush(stderr)
        end
        if res < error
            break
        end
    end

    return Sx, Sy, peps, res, lognorm
end

super_orthonormalization(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; kwargs...) = super_orthonormalization!(deepcopy(peps); kwargs...)


# Random gague transformation
function random_gauge_transform(o1, o2)
    comm = commonind(o1, o2)
    R = randomITensor(comm, comm')
    Rinv = itensor(inv(R.tensor), reverse(R.tensor.inds))
    #m = o1 * o2
    o1 = apply(o1, R)
    o2 = apply(o2, Rinv)
    #@show norm(o1*o2 - m)
    return o1, o2
end
random_gauge_transform(peps) = random_gauge_transform!(deepcopy(peps))
function random_gauge_transform!(peps)
     
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            peps[i,j], peps[i+1,j] = random_gauge_transform(peps[i,j], peps[i+1,j])
        end
        if j < size(peps, 2)
            peps[i,j], peps[i,j+1] = random_gauge_transform(peps[i,j], peps[i,j+1])
        end
    end
    return peps
end

function divide_on_split(o1, o2, S; directional=false)
    l = commonind(o1, o2)
    
    if directional
        inv_S = QuantumNaturalfPEPS.inv_cutoff_func.(S; cutoff=1e-6)
        Sinv = itensor(diagm(inv_S), l, l')
        o2 = apply(o2, Sinv)
        return o1, o2
    else
        inv_S = QuantumNaturalfPEPS.inv_cutoff_func.(sqrt.(S); cutoff=1e-6)
        
        Sinv = itensor(diagm(inv_S), l, l')
        o1 = apply(o1, Sinv)
        o2 = apply(o2, Sinv)
        return o1, o2
    end
end

function divide_by_spectrum!(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; Sx=nothing, Sy=nothing, directional=false, horizontal=true, vertical=true, kwargs...)
    if Sx === nothing
        Sx, Sy, peps = QuantumNaturalfPEPS.super_orthonormalization!(peps; kwargs...)
    end

    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1) && horizontal
            S = Sx[i, j, :]
            peps[i, j], peps[i+1, j] = divide_on_split(peps[i, j], peps[i+1,j], S; directional)
        end
        if j < size(peps, 2) && vertical
            S = Sy[i, j, :]
            peps[i, j], peps[i, j+1] = divide_on_split(peps[i, j], peps[i,j+1], S; directional)
        end
    end
    return Sx, Sy, peps
end
divide_by_spectrum(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; kwargs...) = divide_by_spectrum!(deepcopy(peps); kwargs...)