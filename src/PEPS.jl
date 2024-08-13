mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs
    bond_dim::Integer
    
    sample_dim::Integer
    sample_cutoff::Real
    contract_dim::Integer
    contract_cutoff::Real
    double_contract_dim::Integer
    double_contract_cutoff::Real

    function PEPS(tensors::Matrix{ITensor}, bond_dim::Integer; sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim,
                                                               sample_cutoff=1e-13, contract_cutoff=1e-13, double_contract_cutoff=1e-13,
                                                               shift=false)
        peps = new(tensors, nothing, bond_dim, sample_dim, sample_cutoff, contract_dim, contract_cutoff, double_contract_dim, double_contract_cutoff)
        return shift!(peps, shift)
    end
end

Base.size(peps::PEPS, args...) = size(peps.tensors, args...)
Base.getindex(peps::PEPS, args...) = getindex(peps.tensors, args...)
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)
Base.show(io::IO, peps::PEPS) = print(io, "PEPS(L=$(size(peps)), bond_dim=$(peps.bond_dim), sample_dim=$(peps.sample_dim), contract_dim=$(peps.contract_dim), double_contract_dim=$(peps.double_contract_dim))")
Base.eltype(peps::PEPS) = eltype(peps.tensors[1, 1])

function Base.getproperty(x::PEPS, y::Symbol)
    if y === :double_layer_envs
        double_layer_envs = getfield(x, :double_layer_envs)
        if double_layer_envs === nothing
            double_layer_envs = generate_double_layer_envs(x)
            setfield!(x, :double_layer_envs, double_layer_envs)
            @warn "PEPS: Double layer environments generated automatically"
        end
        return double_layer_envs
        
    else
        return getfield(x, y)
    end
end

Base.convert(::Type{Vector}, peps::PEPS) = flatten(peps)
function flatten(peps::PEPS) # Flattens the tensors into a vector
    type = eltype(peps)
    θ = Vector{type}(undef, length(peps))
    pos = 1
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            shift = prod(dim.(inds(peps[i,j])))
            x = @view θ[pos:pos+shift-1]
            permute_reshape_and_copy!(x, peps[i, j], (siteind(peps, i, j), linkinds(peps, i, j)...))
            #permute_reshape_and_copy!(x, peps[i, j], (linkinds(peps,i,j)..., siteind(peps,i,j)))
            pos = pos+shift
        end
    end
    return θ
end

function Base.length(peps::PEPS)
    x = 0
    for ten in peps.tensors
        x += length(ITensors.tensor(ten))
    end
    return x
end

function tensor_std(peps::PEPS)
    mean_, mean_2 = 0, 0
    l = 0
    for ten in peps.tensors
        mean_ += sum(ten.tensor.storage)
        mean_2 += sum(x->x^2, ten.tensor.storage)
        l += length(ITensors.tensor(ten))
    end
    mean_ /= l
    mean_2 /= l
    return sqrt(mean_2 - mean_^2)
end

function shift!(peps::PEPS, shift::Bool) 
    if shift
        return shift!(peps, 2 * tensor_std(peps) / peps.bond_dim)
    else
        return peps
    end
end

function shift!(peps::PEPS, shift::Number)
    for ten in peps.tensors
        ten .+= shift 
    end
    return peps
end

function write!(peps::PEPS, θ::Vector{T}; reset_double_layer=true) where T# Writes the vector θ into the tensors.
    @assert eltype(peps) == T "The type of the PEPS and the vector θ must be the same type $T != $(eltype(peps))"
    pos = 1
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            shift = prod(dim.(inds(peps[i,j])))
            peps[i, j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
            pos += shift
        end
    end
    if reset_double_layer
        peps.double_layer_envs = nothing
    end
end

ITensors.siteinds(type, Lx, Ly) = [siteind(type; addtags="nx=$i,ny=$j") for i in 1:Lx, j in 1:Ly]
siteinds_compat(phys_dim, Lx, Ly) = [Index(phys_dim, "phys_$(j)_$(i)") for i in 1:Lx, j in 1:Ly] # TO be removed

# initializes a PEPS
PEPS(type, Lx::Int64, Ly::Int64; kwargs...) = PEPS(type, siteinds(type, Lx, Ly); kwargs...)
function PEPS(::Type{S}, type, Lx::Int64, Ly::Int64; kwargs...) where {S<:Number}
    hilbert = siteinds(type, Lx, Ly)
    return PEPS(S, hilbert; kwargs...)
end

PEPS(hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, kwargs...) = PEPS(Float64, hilbert, bond_dim; kwargs...)

function PEPS(::Type{S}, hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, tensor_init=random_unitary, shift=true, kwargs...) where {S<:Number}
    
    tensors = isoPEPS_tensor_init(S, hilbert, bond_dim; tensor_init)
    return PEPS(tensors, bond_dim; shift, kwargs...)
end

# alternative initializer that initializes the peps as an isopeps
#   |
#   v /
# ->[]->
#   |
#   v
function isoPEPS_tensor_init(::Type{S}, hilbert, bond_dim; tensor_init=random_unitary) where {S<:Number}
    Lx, Ly = size(hilbert)

    h_links, v_links = init_Links(hilbert; bond_dim)

    # filling the matrix of tensors with random unitary ITensors wich share the same indices with their neighbours
    tensors = Array{ITensor}(undef, Lx, Ly)
    for i in 1:Lx
        for j in 1:Ly
            outgoing_inds = Vector{Index{Int64}}()
            ingoing_inds = Vector{Index{Int64}}()
            push!(ingoing_inds, hilbert[i, j])


            if j != Ly
                push!(outgoing_inds, h_links[i,j])
            end
            if i != Lx
                push!(outgoing_inds, v_links[i,j])
            end
            if j != 1
                push!(ingoing_inds, h_links[i,j-1])
            end
            if i != 1
                push!(ingoing_inds, v_links[i-1,j])
            end

            tensors[i,j] = tensor_init(S, ingoing_inds, outgoing_inds)
        end
    end
    return tensors
end

function PEPS_tensor_init(::Type{S}, hilbert, bond_dim; tensor_init=randomITensor) where {S<:Number}
    Lx, Ly = size(hilbert)

    h_links, v_links = init_Links(hilbert; bond_dim)

    # filling the matrix of tensors with random unitary ITensors wich share the same indices with their neighbours
    tensors = Array{ITensor}(undef, Lx, Ly)
    inds = Vector{Index{Int64}}()
    for i in 1:Lx
        for j in 1:Ly
            push!(inds, hilbert[i, j])

            if j != Ly
                push!(oinds, h_links[i,j])
            end
            if i != Lx
                push!(oinds, v_links[i,j])
            end
            if j != 1
                push!(inds, h_links[i,j-1])
            end
            if i != 1
                push!(inds, v_links[i-1,j])
            end

            tensors[i,j] = tensor_init(S, inds)
        end
    end
    return tensors
end

function init_Links(hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1)
    Lx, Ly = size(hilbert)

    # initializing bond indices
    h_links = Array{Index{Int64}}(undef, Lx, Ly-1)
    v_links = Array{Index{Int64}}(undef, Lx-1, Ly)
    for i in 1:Lx
        for j in 1:Ly-1
            h_links[i,j] = Index(bond_dim, "h_link, $(i)$(j) -> $(i)$(j+1)")
        end
    end
    for i in 1:Lx-1
        for j in 1:Ly
            v_links[i,j] = Index(bond_dim, "v_link, $(i)$(j) -> $(i+1)$(j)")
        end
    end
    return h_links, v_links
end

function ITensors.siteind(peps::PEPS, i, j)
    Lx, Ly = size(peps)
    if Lx == 1 && Ly == 1
        return firstind(peps[1, 1])
    elseif Lx == 1
        return uniqueind(peps[i,j], peps[i,j%Ly + 1], peps[i, (j-2+Lx)%Ly+1])
    elseif Ly == 1
        return uniqueind(peps[i,j], peps[i%Lx+1, j], peps[(i-2+Lx)%Lx+1, j])
    end
    si = uniqueind(peps[i,j], peps[i%Lx+1, j], peps[(i-2+Lx)%Lx+1, j], peps[i,j%Ly + 1], peps[i, (j-2+Ly)%Ly+1])
    @assert si !== nothing
    return si
end

ITensors.linkinds(peps::PEPS, i, j) = filter!(!=(siteind(peps, i, j)), collect(inds(peps[i ,j])))

ITensors.siteinds(peps::PEPS) = [siteind(peps, i, j) for i in 1:size(peps, 1), j in 1:size(peps, 2)]

# returns the exact inner product of 2 peps (only used for testing purposes)
function inner_peps(psi::PEPS, psi2::PEPS)
    x = 1
    for i in 1:size(psi)[1]
        for j in 1:size(psi)[2]
            x *= psi[i,j]*psi2[i,j]*delta(siteind(psi,i,j), siteind(psi2,i,j))
        end
    end
    return x[1]
end

function get_projector(i, index; shift=1)
    @assert index.space >= i+shift "The index space is too small for the given shift"
    return onehot(index=>i+shift)
end

function get_projected(peps::PEPS, S, i, j)
    if i === Colon() && j === Colon()
        return [get_projected(peps, S, i, j) for i in 1:size(peps, 1), j in 1:size(peps, 2)]
    elseif i === Colon()
        return [get_projected(peps, S, i, j) for i in 1:size(peps, 1)]
    elseif j === Colon()
        return [get_projected(peps, S, i, j) for j in 1:size(peps, 2)]
    end

    index = siteind(peps, i, j)
    return peps[i,j] * get_projector(S[i, j], index)
end

get_projected(peps::PEPS, S::Matrix{Int64}) = [get_projected(peps, S, i, j)  for i in 1:size(S, 1), j in 1:size(S, 2)]

function contract_peps_exact(peps)
    x = 1
    for i in 1:size(peps,1)
        for j in 1:size(peps,2)
            x *= peps[i,j]
        end
    end

    return x[1]
end