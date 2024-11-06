
function reduce_dim(l1, l2)
    m = zeros(dim(l1), dim(l2))
    l = min(dim(l1), dim(l2))
    m[1:l, 1:l] .= diagm(ones(l))
    return ITensor(m, l1, l2)
end

function split_merge_slow(o1, o2)
    comm = commonind(o1, o2)
    l1 = uniqueinds(o1, comm)
    l2 = uniqueinds(o2, comm)
    
    U, S, V = svd(o1 * o2, l1)
    o1 = U * (sqrt.(S) * reduce_dim(S.tensor.inds[2], comm))
    o2 = V * (sqrt.(S) * reduce_dim(S.tensor.inds[1], comm))
    return o1, o2, S.tensor.storage[1:dim(comm)]
end

inv_cutoff_func(x; cutoff=1e-6) = x < cutoff ? 0 : 1/x
isnan2(x) = sum(isnan.(x)) > 0
function split_merge(o1, o2; cutoff=1e-6, directional=false, normalize_spectrum=true)
    # https://arxiv.org/pdf/1808.00680
    comm = commonind(o1, o2)
    
    M1 = o1 * prime(o1, comm)
    M2 = o2 * prime(o2, comm)
    
    M1_, M2_ =  reshape(M1.tensor.storage, dim(comm),dim(comm)), reshape(M2.tensor.storage, dim(comm),dim(comm))
    D1, u1 = eigen(M1_)

    
    D2, u2 = eigen(M2_)
    #@show D1
    #@show D2
    D1_sqrt, D2_sqrt = sqrt.(abs.(D1)), sqrt.(abs.(D2))
    # Stable inversion by only inverting the non-zero eigenvalues
    D1_inv = inv_cutoff_func.(D1_sqrt; cutoff)
    D2_inv = inv_cutoff_func.(D2_sqrt; cutoff)
    lambda_1 = reshape(D1_sqrt, :, 1) .* u1'
    lambda_2 = u2 .* reshape(D2_sqrt, 1, :)

    lambda_ = lambda_1 * lambda_2
    w1, S, w2 = svd(lambda_)
    if norm(S) < 1e-3
        @show norm(o1) ,norm(o2)
    end
    if normalize_spectrum
        S ./= norm(S)
    end
    x_inv = u1 .* reshape(D1_inv, 1, :)
    x_inv = x_inv * w1

    y_inv = w2' .* reshape(D2_inv, 1, :)
    y_inv = y_inv * u2'
    
    if directional
        x_ = x_inv
        y_ = y_inv .* reshape(abs.(S), :, 1)
    else
        x_ = x_inv .* reshape(sqrt.(abs.(S)), 1, :)
        y_ = y_inv .* reshape(sqrt.(abs.(S)), :, 1)
    end
    
    #@show diag(y_ * x_)
    
    x_ = ITensor(x_, comm', comm)
    y_ = ITensor(y_, comm, comm')
    o1n = apply(o1, x_)
    o2n = apply(o2, y_)
    return o1n, o2n, S
end

function split_merge!(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; split_merge_=split_merge, kwargs...)
    Sx = Array{Float64}(undef, size(peps, 1)-1, size(peps, 2), peps.bond_dim)
    Sy = Array{Float64}(undef, size(peps, 1), size(peps, 2)-1, peps.bond_dim)
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            peps[i, j], peps[i+1, j], S = split_merge_(peps[i, j], peps[i+1, j]; kwargs...)
            Sx[i, j, :] .= S
        end
        if j < size(peps, 2)
            peps[i, j], peps[i, j+1], S = split_merge_(peps[i, j], peps[i, j+1]; kwargs...)
            Sy[i, j, :] .= S
        end
    end
    return Sx, Sy
end

function super_orthonormalization!(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; k=1000, error=1e-8, verbose=false, kwargs...)   
    Sx, Sy = split_merge!(peps; kwargs...)
    local res
    for i in 2:k
        Sx2, Sy2 = split_merge!(peps; kwargs...)
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

    return Sx, Sy, peps, res
end

super_orthonormalization(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; kwargs...) = super_orthonormalization!(deepcopy(peps); kwargs...)


# Random gague transformation
function random_gauge_transform(o1, o2)
    comm = commonind(o1, o2)
    R = randomITensor(comm, comm')
    Rinv = ITensor(inv(R.tensor), reverse(R.tensor.inds))
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

function divide_by_split(o1, o2, S)
    #@show inds(o1) inds(o2)
    l = commonind(o1, o2)
    
    inv_S = QuantumNaturalfPEPS.inv_cutoff_func.(sqrt.(S); cutoff=1e-6)
    
    Sinv = ITensor(diagm(inv_S), l, l')
    o1 = apply(o1, Sinv)
    o2 = apply(o2, Sinv)
    return o1, o2
end

function divide_by_spectrum(peps::Union{QuantumNaturalfPEPS.PEPS, Matrix{ITensor}}; Sx=nothing, Sy=nothing, α=1)
    peps = deepcopy(peps)
    
    if Sx === nothing
        Sx, Sy, peps = QuantumNaturalfPEPS.super_orthonormalization!(peps)
    end

    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if i < size(peps, 1)
            S = Sx[i, j, :]
            S = α .* S .+ (1-α) * S[1] 
            peps[i, j], peps[i+1, j] = divide_by_split(peps[i, j], peps[i+1,j], S)
        end
        if j < size(peps, 2)
            S = Sy[i, j, :]
            S = α .* S .+ (1-α) * S[1] 
            peps[i, j], peps[i, j+1] = divide_by_split(peps[i, j], peps[i,j+1], S)
        end
    end
    return Sx, Sy, peps
end

function multiply_spectrum!(peps, spectrum)
    spectrum = sqrt.(spectrum)
    spectrum = diagm(spectrum)
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        for li in linkinds(peps, i, j)
            t = ITensor(spectrum , li, li')
            peps[i, j] = apply(peps[i, j], t)
        end
    end
    return peps
end