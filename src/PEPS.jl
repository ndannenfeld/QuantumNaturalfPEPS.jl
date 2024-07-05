mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs
    bond_dim::Integer
    sample_dim::Integer
    contract_dim::Integer
    double_contract_dim::Integer
    PEPS(tensors::Matrix{ITensor}, bond_dim::Integer; sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim) = new(tensors, nothing, bond_dim, sample_dim, contract_dim, double_contract_dim)
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
            @info "Double layer environments generated automatically"
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
            θ[pos:pos+shift-1] = reshape(Array(peps[i,j], inds(peps[i,j])), :)
            pos = pos+shift
        end
    end
    return θ
end

function Base.length(peps::PEPS)
    x = 0
    for ten in peps.tensors
        x += prod(size(ten))
    end
    return x
end

function write!(peps::PEPS, θ::Vector{T}) where T# Writes the vector θ into the tensors.
    @assert eltype(peps) == T "The type of the PEPS and the vector θ must be the same type $T != $(eltype(peps))"
    pos = 1
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            shift = prod(dim.(inds(peps[i,j])))
            peps[i, j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
            pos += shift
        end
    end
    peps.double_layer_envs = nothing
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



function PEPS_tensor_init(::Type{S}, hilbert, links) where {S<:Number}
    Lx, Ly = size(hilbert)

    # filling the matrix of tensors with random ITensors wich share the same indices with their neighbours
    tensors = Array{ITensor}(undef, Lx, Ly)
    for i in 1:Ly
        for j in 1:Lx
            phys_ind = hilbert[j, i]
            if i == 1
                if j == 1
                    tensors[j, i] = randomITensor(S, links[1], links[Ly*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    tensors[j, i] = randomITensor(S, links[Lx-1], links[Ly*(Lx-1)+Lx], phys_ind)
                else
                    tensors[j, i] = randomITensor(S, links[j-1],links[j],links[Ly*(Lx-1)+j], phys_ind)
                end
            elseif i == Ly
                if j == 1
                    tensors[j, i] = randomITensor(S, links[Ly*(Lx-1)+(Ly-2)*Lx+1], links[(Ly-1)*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    tensors[j, i] = randomITensor(S, links[Ly*(Lx-1)+(Ly-1)*Lx], links[Ly*(Lx-1)], phys_ind)
                else
                    tensors[j, i] = randomITensor(S, links[Ly*(Lx-1)+(Ly-2)*Lx+j], links[(Ly-1)*(Lx-1)+j-1], links[(Ly-1)*(Lx-1)+j], phys_ind)
                end
            elseif j == 1
                tensors[j, i] = randomITensor(S, links[Ly*(Lx-1)+(i-2)*Lx+1], links[Ly*(Lx-1)+(i-1)*Lx+1], links[(i-1)*(Lx-1)+1], phys_ind)
            elseif j == Lx
                tensors[j, i] = randomITensor(S, links[Ly*(Lx-1)+(i-1)*Lx], links[Ly*(Lx-1)+(i)*Lx], links[i*(Lx-1)], phys_ind)
            else
                tensors[j, i] = randomITensor(S, links[(i-1)*(Lx-1)+j-1], links[(i-1)*(Lx-1)+j], links[Ly*(Lx-1)+(i-2)*Lx+j], links[Ly*(Lx-1)+(i-1)*Lx+j], phys_ind)
            end
        end
    end
    return tensors
end
# TODO: Write an alternative initializer that initializes the peps as an isopeps
#   |
#   v /
# ->[]->
#   |
#   v

function PEPS(::Type{S}, hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, tensor_init=PEPS_tensor_init, kwargs...) where {S<:Number}
    Lx, Ly = size(hilbert)

    # initializing bond indices
    links = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        # TODO: Improve the naming of the links to make them more readable
        links[i] = Index(bond_dim, "Link,l=$(i)")
    end
    tensors = tensor_init(S, hilbert, links)
    
    return PEPS(tensors, bond_dim; kwargs...)
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
    si = uniqueind(peps[i,j], peps[i%Lx+1, j], peps[(i-2+Lx)%Lx+1, j], peps[i,j%Ly + 1], peps[i, (j-2+Lx)%Ly+1])
    return si
end

function linkinds(peps::PEPS, i,j)
    return filter!(!=(siteind(peps,i,j)), collect(inds(peps[i,j])))
end

ITensors.siteinds(peps::PEPS) = [siteind(peps, i, j) for i in 1:size(peps, 1), j in 1:size(peps, 2)]

# returns the exact inner product of 2 peps (only used for testing purposes)
function inner_peps(psi::PEPS, psi2::PEPS)
    x = 1
    for i in 1:size(psi)[1]
        for j in 1:size(psi)[2]
            x *= psi[i,j]*psi2[i,j]*delta(inds(psi[i,j], "phys_$(j)_$(i)"), inds(psi2[i,j], "phys_$(j)_$(i)"))
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

get_projected(peps::PEPS, S) = [get_projected(peps, S, i, j)  for i in 1:size(peps, 1), j in 1:size(peps, 2)]

function contract_peps_exact(peps)
    x = 1
    for i in 1:size(peps,1)
        for j in 1:size(peps,2)
            x *= peps[i,j]
        end
    end

    return x[1]
end