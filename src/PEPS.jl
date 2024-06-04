# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::ComplexF64 # TODO: Why is this complex?
    Environment(env, f) = new(env, f)
end

mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs::Union{Vector{Environment}, Nothing}
    norm::Float64
    bond_dim::Integer
    sample_dim::Integer
    contract_dim::Integer
    double_contract_dim::Integer
    PEPS(tensors::Matrix{ITensor}, bond_dim::Integer; norm=0, sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim) = new(tensors, nothing, norm, bond_dim, sample_dim, contract_dim, double_contract_dim)
end

Base.size(peps::PEPS, args...) = size(peps.tensors, args...)
Base.getindex(peps::PEPS, i::Int) = peps.tensors[i, :]
Base.getindex(peps::PEPS, i::Int, j::Int) = peps.tensors[i, j] # You can use peps[i, j]
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)
Base.show(io::IO, peps::PEPS) = print(io, "PEPS(L=$(size(peps)), bond_dim=$(peps.bond_dim), sample_dim=$(peps.sample_dim), contract_dim=$(peps.contract_dim), double_contract_dim=$(peps.double_contract_dim))")
Base.eltype(peps::PEPS) = eltype(peps.tensors[1,1]) # TODO: Make sure that the code works for both complex and real numbers

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

function flatten(peps::PEPS) # Flattens the tensors into a vector
    type = eltype(peps)
    θ = type[]
    # TODO: Slow, first calculate the length and then fill the vector
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            append!(θ, reshape(Array(peps[i,j], inds(peps[i,j])), :))
        end
    end
    return θ
end

Base.length(peps::PEPS) = length(flatten(peps)) # length(θ) # TODO: Write a more efficient version

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
PEPS(type, Lx::Int64, Ly::Int64; kwargs...) = PEPS(type, siteinds(type, Lx, Ly), kwargs...)
function PEPS(::Type{S}, type, Lx::Int64, Ly::Int64; kwargs...) where {S<:Number}
    hilbert = siteinds(type, Lx, Ly)
    return PEPS(S, hilbert, kwargs...)
end

PEPS(hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, kwargs...) = PEPS(Float64, hilbert, bond_dim, kwargs...)

function PEPS(::Type{S}, hilbert::Matrix{Index{Int64}}; bond_dim::Int64=1, kwargs...) where {S<:Number}
    Lx, Ly = size(hilbert)
    tensors = Array{ITensor}(undef, Lx, Ly)

    # initializing bond indices
    indices = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        indices[i] = Index(bond_dim, "Link,l=$(i)")
    end
    
    # filling the matrix of tensors with random ITensors wich share the same indices with their neighbours
    for i in 1:Ly
        for j in 1:Lx
            phys_ind = hilbert[j, i]
            if i == 1
                if j == 1
                    peps_tensor = randomITensor(S, indices[1], indices[Ly*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(S, indices[Lx-1], indices[Ly*(Lx-1)+Lx], phys_ind)
                else
                    peps_tensor = randomITensor(S, indices[j-1],indices[j],indices[Ly*(Lx-1)+j], phys_ind)
                end
            elseif i == Ly
                if j == 1
                    peps_tensor = randomITensor(S, indices[Ly*(Lx-1)+(Ly-2)*Lx+1], indices[(Ly-1)*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(S, indices[Ly*(Lx-1)+(Ly-1)*Lx], indices[Ly*(Lx-1)], phys_ind)
                else
                    peps_tensor = randomITensor(S, indices[Ly*(Lx-1)+(Ly-2)*Lx+j], indices[(Ly-1)*(Lx-1)+j-1], indices[(Ly-1)*(Lx-1)+j], phys_ind)
                end
            elseif j == 1
                peps_tensor = randomITensor(S, indices[Ly*(Lx-1)+(i-2)*Lx+1], indices[Ly*(Lx-1)+(i-1)*Lx+1], indices[(i-1)*(Lx-1)+1], phys_ind)
            elseif j == Lx
                peps_tensor = randomITensor(S, indices[Ly*(Lx-1)+(i-1)*Lx], indices[Ly*(Lx-1)+(i)*Lx], indices[i*(Lx-1)], phys_ind)
            else
                peps_tensor = randomITensor(S, indices[(i-1)*(Lx-1)+j-1], indices[(i-1)*(Lx-1)+j], indices[Ly*(Lx-1)+(i-2)*Lx+j], indices[Ly*(Lx-1)+(i-1)*Lx+j], phys_ind)
            end
            
            tensors[j, i] = peps_tensor
        end
    end
    
    return PEPS(tensors, bond_dim; kwargs...)
end

function ITensors.siteind(peps::PEPS, i, j)
    # TODO: Get the indices from the PEPS instead of generating new ones (see ITensors/src/ITensorMPS/abstractmps.jl:620
    return 
end
function ITensors.siteinds(peps::PEPS)
    # TODO: Replace with the following line
    #hilbert = [siteind(peps, i, j) for i in 1:size(peps, 1), j in 1:size(peps, 2)]
    hilbert = siteinds("S=1/2", size(peps, 1), size(peps, 2))
    return hilbert
end

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


