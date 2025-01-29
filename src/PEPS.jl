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
    show_warning::Bool

    mask::Matrix

    function PEPS(tensors::Matrix{ITensor}, bond_dim::Integer; sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim,
                                                               sample_cutoff=1e-13, contract_cutoff=1e-13, double_contract_cutoff=1e-13,
                                                               shift=false, show_warning=false, mask=ones(size(tensors)))
        peps = new(tensors, nothing, bond_dim, sample_dim, sample_cutoff, contract_dim, contract_cutoff, double_contract_dim, double_contract_cutoff, show_warning, mask)
        return shift!(peps, shift)
    end
end

maxbonddim(peps::PEPS) = peps.bond_dim

Base.size(peps::PEPS, args...) = size(peps.tensors, args...)
Base.getindex(peps::PEPS, args...) = getindex(peps.tensors, args...)
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)
Base.show(io::IO, peps::PEPS) = print(io, "PEPS(L=$(size(peps)), bond_dim=$(peps.bond_dim), sample_dim=$(peps.sample_dim), contract_dim=$(peps.contract_dim), double_contract_dim=$(peps.double_contract_dim))")
Base.eltype(peps::PEPS) = eltype(peps.tensors[1, 1])

function Base.getproperty(x::PEPS, y::Symbol)
    if y === :double_layer_envs
        double_layer_envs = getfield(x, :double_layer_envs)
        if double_layer_envs === nothing
            @warn "PEPS: Double layer environments generated automatically"
            double_layer_envs = generate_double_layer_envs(x)
            setfield!(x, :double_layer_envs, double_layer_envs)
        end
        return double_layer_envs
        
    else
        return getfield(x, y)
    end
end

Base.convert(::Type{Vector}, peps::PEPS; mask=peps.mask) = flatten(peps; mask)
function flatten(peps::PEPS; mask=peps.mask) # Flattens the tensors into a vector
    type = eltype(peps)
    θ = Vector{type}(undef, length(peps; mask))
    pos = 1
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            if mask[i,j]!=0
                shift = prod(dim.(inds(peps[i,j])))
                x = @view θ[pos:pos+shift-1]
                permute_reshape_and_copy!(x, peps[i, j], (siteind(peps, i, j), linkinds(peps, i, j)...))
                pos = pos+shift
            end
        end
    end
    return θ
end

#function Base.length(peps::PEPS)
#    x = 0
#    for ten in peps.tensors
#        x += length(ITensors.tensor(ten))
#    end
#    return x
#end

function Base.length(peps::PEPS; mask=peps.mask)
    x = 0
    for i in 1:size(peps,1)
        for j in 1:size(peps,2)
            if mask[i,j]!=0
                x += length(ITensors.tensor(peps.tensors[i,j]))
            end
        end
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
        return shift!(peps, 2 * tensor_std(peps) / maxbonddim(peps))
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

function write!(peps::PEPS, θ::Vector{T}; reset_double_layer=true, mask=peps.mask) where T# Writes the vector θ into the tensors.
    @assert eltype(peps) == T "The type of the PEPS and the vector θ must be the same type $T != $(eltype(peps))"
    pos = 1
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            if mask[i,j]!=0
                shift = prod(dim.(inds(peps[i,j])))
                peps[i, j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
                pos += shift
            end
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

PEPS(hilbert::Matrix{Index{Int64}}; kwargs...) = PEPS(Float64, hilbert; kwargs...)


"""
    PEPS(S, hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, tensor_init=random_unitary, shift=true, kwargs...) where {S<:Number}
    tensor_init: function that initializes the tensors, default=random_unitary other options: randomITensor
"""
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
            h_links[i,j] = Index(bond_dim, "h_link, $(i);$(j) -> $(i);$(j+1)")
        end
    end
    for i in 1:Lx-1
        for j in 1:Ly
            v_links[i,j] = Index(bond_dim, "v_link, $(i);$(j) -> $(i+1);$(j)")
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

function get_projector(i::Int, index; shift=1)
    @assert index.space >= i+shift "The dimension is $(index.space) but the requested index is $(i+shift)"
    return onehot(index=>i+shift)
end

function get_projected(peps::PEPS, S::Matrix{Int64}, i, j)
    if i === Colon() && j === Colon()
        return [get_projected(peps, S[i, j], i, j) for i in 1:size(peps, 1), j in 1:size(peps, 2)]
    elseif i === Colon()
        return [get_projected(peps, S[i, j], i, j) for i in 1:size(peps, 1)]
    elseif j === Colon()
        return [get_projected(peps, S[i, j], i, j) for j in 1:size(peps, 2)]
    end
    return get_projected(peps, S[i, j], i, j)
end

get_projected(peps::PEPS, Sij::Int64, i::Int64, j::Int64) = peps[i,j] * get_projector(Sij, siteind(peps, i, j))
get_projected(peps::PEPS, S::Matrix{Int64}) = get_projected(peps, S, :, :)

function contract_peps_exact(peps)
    x = 1
    for i in 1:size(peps,1)
        for j in 1:size(peps,2)
            x *= peps[i,j]
        end
    end

    return x[1]
end

function write_Tensor!(peps, tensor, i, j)
    @assert eltype(tensor) == eltype(peps) "The type of the PEPS and the tensor must be the same type $(eltype(tensor)) != $(eltype(peps))"
    indices = Vector{Index}()    
    if j != 1
        push!(indices, commoninds(peps[i,j], peps[i,j-1])...)
    end
    if i != size(peps,1)
        push!(indices, commoninds(peps[i,j], peps[i+1,j])...)
    end
    if j != size(peps,2)
        push!(indices, commoninds(peps[i,j], peps[i,j+1])...)
    end
    if i != 1
        push!(indices, commoninds(peps[i,j], peps[i-1,j])...)
    end
    push!(indices, siteind(peps, i, j))
    
    if typeof(tensor)==ITensor
        replaceind!.(Ref(tensor), inds(tensor), indices)
    else
        tensor = ITensor(tensor, indices)
    end
    perm = NDTensors.getperm(inds(peps[i,j]), indices)
    # TODO: Why do you permute the tensors here?
    peps[i,j] = ITensor(permutedims(tensor.tensor, perm))
end

# Writes Array of Tensors into fPEPS with a pattern e.g.
# pattern = [1 2 3
#            3 4 5]
# fPEPS has the following tensors in its bulk: 1 2 3 1 2 3 ...
#                                              3 4 5 3 4 5 ...
#                                              1 2 3 1 2 3 ... 
function iPEPS_to_fPEPS(iPEPS, Lx, Ly, pattern; vectors=:random)
    T = eltype(iPEPS[1])
    samplecut = marginalcut = bdim = size(iPEPS[1], 1)
    contract_dim = 3*bdim
    hilbert = siteinds("S=1/2", Lx, Ly)

    peps = PEPS(T, hilbert; bond_dim=bdim, sample_dim=samplecut, double_contract_dim=marginalcut, contract_dim, shift=false, show_warning=true)
    
    return iPEPS_to_fPEPS!(peps::PEPS, iPEPS, Lx, Ly, pattern; vectors)
end

function iPEPS_to_fPEPS!(peps::PEPS, iPEPS, Lx, Ly, pattern; vectors=:random)
    peps.double_layer_envs = nothing
    iPEPS_to_fPEPS_bulk!(peps, iPEPS, pattern)
    if vectors != :none
        iPEPS_to_fPEPS_boundary!(peps, iPEPS, pattern; vectors)
    end
    return peps
end

function iPEPS_to_fPEPS_bulk!(peps, iPEPS, pattern)
    for i in 2:size(peps,1)-1
        for j in 2:size(peps,2)-1
            x = pattern[(i-2)%(size(pattern,1))+1, (j-2)%(size(pattern,2))+1]
            write_Tensor!(peps, iPEPS[x], i, j)
        end
    end
end

# vectors can be either: Array{Vector, 2*size(peps,1) + 2*size(peps,2) + 4} -> will be used subsequently to contract boundary Tensors
#                        Array{Vector, 4} -> will be used subsequently to contract boundary Tensors. A different Vector will be used for 
#                                            different contract-direction. 1: left, 2: down, 3: right, 4: up
#                        :random          -> generates random Vectors for contraction
#                        :four            -> generates 4 random Vectors. A different Vector will be used for different contract-directions.
#                        :ones            -> generates 4 one-valued vectors
#                        :none            -> border peps stay random
function iPEPS_to_fPEPS_boundary!(peps, iPEPS, pattern; vectors=:random)
    if !isa(vectors, AbstractArray)
        vectors = generate_vectors(peps, iPEPS, vectors)
    end

    isFour = length(vectors)==4

    indices = Index.(size(iPEPS[1]))
    for j in [1,size(peps,2)]
        for i in 1:size(peps,1)
            x = pattern[(i-2+size(pattern,1))%size(pattern,1) + 1, (j-2+size(pattern,2))%size(pattern,2) + 1]
            j == 1 ? y = 1 : y = 3
            if isFour
                ipeps_ten = ITensor(iPEPS[x], indices)*ITensor(vectors[y], indices[y])
            else
                ipeps_ten = ITensor(iPEPS[x], indices)*ITensor(popfirst!(vectors), indices[y])
            end
            if i == 1 || i == size(peps,1)
                i == 1 ? y = 4 : y = 2
                if isFour
                    ipeps_ten *= ITensor(vectors[y], indices[y])
                else
                    ipeps_ten *= ITensor(popfirst!(vectors), indices[y])
                end
            end
            peps[i,j] = write_Tensor!(peps, ipeps_ten, i, j)
        end
    end

    for j in 2:size(peps,2)-1
        for i in [1, size(peps,1)]
            x = pattern[(i-2+size(pattern,1))%size(pattern,1) + 1, (j-2+size(pattern,2))%size(pattern,2) + 1]
            i == 1 ? y = 4 : y = 2
            if isFour
                ipeps_ten = ITensor(iPEPS[x], indices)*ITensor(vectors[y], indices[y])
            else
                ipeps_ten = ITensor(iPEPS[x], indices)*ITensor(popfirst!(vectors), indices[y])
            end
            peps[i,j] = write_Tensor!(peps, ipeps_ten, i, j)
        end
    end
end

function generate_vectors(peps, iPEPS, vector_type)
    if vector_type == :random
        return [rand(maxbonddim(peps)) for i in 1:2*size(peps,1) + 2*size(peps,2) + 4]
    elseif vector_type == :four
        return [rand(maxbonddim(peps)) for i in 1:4]
    elseif vector_type == :ones
        return [ones(maxbonddim(peps)) for i in 1:4]
    else
        throw(ArgumentError("Unrecognized vector_type: $vector_type"))
    end
end