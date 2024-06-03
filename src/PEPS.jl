# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::ComplexF64
    Environment(env, f) = new(env, f)
end

mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs::Vector{Environment}
    norm::Float64
    bond_dim::Integer
    sample_dim::Integer
    contract_dim::Integer
    double_contract_dim::Integer
    PEPS(tensors, bond_dim; norm=0, sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim) = new(tensors, [Environment(MPS(), 1) for i in 1:size(tensors)[1]-1], norm, bond_dim, sample_dim, contract_dim, double_contract_dim)
end

Base.size(peps::PEPS) = size(peps.tensors)
Base.getindex(peps::PEPS, i::Int) = peps.tensors[i, :]
Base.getindex(peps::PEPS, i::Int, j::Int) = peps.tensors[i, j] # You can use peps[i, j]
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)
Base.show(io::IO, peps::PEPS) = print(io, "PEPS(L=$(size(peps)), bond_dim=$(peps.bond_dim), sample_dim=$(peps.sample_dim), contract_dim=$(peps.contract_dim), double_contract_dim=$(peps.double_contract_dim))")

function flatten(peps::PEPS) # Flattens the tensors into a vector
    θ = ComplexF64[]
    # TODO: Slow, first calculate the length and then fill the vector
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            append!(θ, reshape(Array(peps[i,j], inds(peps[i,j])), :))
        end
    end
    return θ
end

Base.length(peps::PEPS) = length(flatten(peps)) # length(θ)

function write!(peps::PEPS, θ::Vector{ComplexF64}) # Writes the vector θ into the tensors.
    pos = 1
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            shift = prod(dim.(inds(peps[i,j])))
            peps[i, j] *= im # Why do we need this?
            peps[i, j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
            pos += shift
        end
    end
end

# initializes a PEPS
function PEPS(Lx::Int64, Ly::Int64, phys_dim::Int64, bond_dim::Int64)
    tensors = Array{ITensor}(undef, Lx,Ly)

    # initializing bond indices
    indices = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        indices[i] = Index(bond_dim, "ind_$(i)")
    end
    
    # filling the matrix of tensors with random ITensors wich share the same indices with their neighbours
    for i in 1:Ly
        for j in 1:Lx
            phys_ind = Index(phys_dim, "phys_$(i)_$(j)")
            if i == 1
                if j == 1
                    peps_tensor = randomITensor(indices[1], indices[Ly*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(indices[Lx-1], indices[Ly*(Lx-1)+Lx], phys_ind)
                else
                    peps_tensor = randomITensor(indices[j-1],indices[j],indices[Ly*(Lx-1)+j], phys_ind)
                end
            elseif i == Ly
                if j == 1
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-2)*Lx+1], indices[(Ly-1)*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-1)*Lx], indices[Ly*(Lx-1)], phys_ind)
                else
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-2)*Lx+j], indices[(Ly-1)*(Lx-1)+j-1], indices[(Ly-1)*(Lx-1)+j], phys_ind)
                end
            elseif j == 1
                peps_tensor = randomITensor(indices[Ly*(Lx-1)+(i-2)*Lx+1], indices[Ly*(Lx-1)+(i-1)*Lx+1], indices[(i-1)*(Lx-1)+1], phys_ind)
            elseif j == Lx
                peps_tensor = randomITensor(indices[Ly*(Lx-1)+(i-1)*Lx], indices[Ly*(Lx-1)+(i)*Lx], indices[i*(Lx-1)], phys_ind)
            else
                peps_tensor = randomITensor(indices[(i-1)*(Lx-1)+j-1], indices[(i-1)*(Lx-1)+j], indices[Ly*(Lx-1)+(i-2)*Lx+j], indices[Ly*(Lx-1)+(i-1)*Lx+j], phys_ind)
            end
            
            tensors[j, i] = peps_tensor
            
        end
    end
    
    return PEPS(tensors, bond_dim)
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


