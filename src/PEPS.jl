# true Environment is env.*exp(f)
mutable struct Environment
    env::MPS
    f::Float64
    Environment(env, f) = new(env, f)
end

mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs::Vector{Environment}
    bond_dim::Integer
    sample_dim::Integer
    contract_dim::Integer
    double_contract_dim::Integer
    PEPS(tensors, bond_dim; sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=bond_dim) = new(tensors, [Environment(MPS(), 1) for i in 1:size(tensors)[1]-1], bond_dim, sample_dim, contract_dim, double_contract_dim)
end

Base.size(peps::PEPS) = size(peps.tensors)
Base.getindex(peps::PEPS, i::Int) = peps.tensors[i, :]
Base.getindex(peps::PEPS, i::Int, j::Int) = peps.tensors[i, j] # You can use peps[i, j]
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)

function flatten(peps::PEPS) # Flattens the tensors into a vector
    θ = Float64[]
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            append!(θ, reshape(Array(peps[i,j], inds(peps[i,j])), :))
        end
    end
    return θ
end

Base.length(peps::PEPS) = length(flatten(peps)) # length(θ)

function write!(peps::PEPS, θ::Vector{Float64}) # Writes the vector θ into the tensors.
    pos = 1
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            shift = prod(dim.(inds(peps[i,j])))
            peps[i,j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
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
            
            tensors[j,i] = peps_tensor
            
        end
    end
    
    return PEPS(tensors, bond_dim)
end


# Computes the environments and <ψ|S>
function get_logψ_and_envs(peps::PEPS, S::Array{Int64,2}, Env_top = nothing)
    
    overwrite = true    # if Env_top is given and the bond dimension is sufficient, we do not need to calculate it again
    if Env_top == nothing
        Env_top = Array{Environment}(undef, size(S,1)-1)
    else
        if maxlinkdim(Env_top[i].env) >= peps.contract_dim
            overwrite = false
        end
    end
    Env_down = Array{Environment}(undef, size(S,1)-1)
    out = Array{Float64}(undef, 2)

    # for every row we calculate the environments once from the top down and once from the bottom up
    for i in 1:size(S,1)
        i_prime = size(S,1)+1-i     # i_prime is used for env_down to count backwards form end to 2
        if i == 1
            if overwrite
                # to calculate the environments, we parse the S row to ITensors and contract each peps.tensor in that row along the physical Index
                Env = MPS([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)])
                normE = norm(Env)
                Env_top[i] = Environment(Env/normE, log(normE))
            end
            
            Env = MPS([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)])
            normE = norm(Env)
            Env_down[i] = Environment(Env/normE, log(normE))
        elseif i == size(S,1)

            # once we calculated all environments we calculate <ψ|S> using the environments
            out[1] = inner(Env_top[i-1].env,MPS([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)])) * exp(Env_top[i-1].f)
            out[2] = inner(Env_down[i-1].env,MPS([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)])) * exp(Env_down[i-1].f)
        else
            if overwrite
                Env = apply(MPO([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)]), Env_top[i-1].env, maxdim=peps.contract_dim)
                normE = norm(Env)
                Env_top[i] = Environment(Env/normE, log(normE)+Env_top[i-1].f)
            end
            
            Env = apply(MPO([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)]), Env_down[i-1].env, maxdim=peps.contract_dim)
            normE = norm(Env)
            Env_down[i] = Environment(Env/normE, log(normE)+Env_down[i-1].f)
        end
    end
    return mean(out), Env_top, Env_down
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

# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ>S>
function get_Ok(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number; Ok::Vector=zeros(length(peps)))
    pos = 1
    shift = 0
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            # peps_S gives an array of ITensors wich are peps[i,y]*S[i,y] for all tensors in the row except for i,j
            peps_S = [(j_p != j ? peps[i,j_p]*ITensor([(S[i,j_p]+1)%2, S[i,j_p]], inds(peps[i,j_p], "phys_$(j_p)_$(i)")) : 1) for j_p in 1:size(S,2)]
            if i == 1
                # we get the differential tensor if we contract the environments with the peps_S
                Ok_Tensor = exp(env_down[end].f)*contract(env_down[end].env .* peps_S)
            elseif i == size(peps)[1]
                Ok_Tensor = exp(env_top[end].f)*contract(env_top[end].env .* peps_S)
            else
                Ok_Tensor = exp(env_top[i-1].f + env_down[end-i+1].f)*contract(env_top[i-1].env .* peps_S .* env_down[end-i+1].env)
            end

            # lastly we reshape the tensor to a vector to obtain the gradient
            shift = prod(dim.(inds(Ok_Tensor)))
            Ok[pos:pos+shift-1] = reshape(Array(Ok_Tensor./logψ, inds(Ok_Tensor)), :)
            pos = pos+shift
        end
    end
    return Ok
end

# calculates the field double_layer_envs of peps
function update_double_layer_envs!(peps::PEPS)

    # first of all we have to rename some indices because the bra and the ket would otherwise share all indices which would lead to troubles in the contraction
    indices_outer = Array{Index}(undef, size(peps)[2])      # corresponds to indices connecting different rows
    for i in 1:length(indices_outer)
        indices_outer[i] = Index(peps.bond_dim, "Ket_$(i)")
    end
    
    bra = MPO(peps[size(peps)[1]])
    # throughout the delta function is used to rename indices
    ket = MPO(peps[size(peps)[1]].*delta.(commoninds.(peps[size(peps)[1]], peps[size(peps)[1]-1]), indices_outer))
    
    indices_inner = Array{Index}(undef, size(peps)[2]-1)       # corresponds to indices connecting tensors in the same row
    for i in 1:length(indices_inner)
        indices_inner[i] = Index(peps.bond_dim, "Ket_inner_$(i)")
        ket[i] = ket[i]*delta(indices_inner[i], commoninds(peps[size(peps)[1],i], peps[size(peps)[1],i+1]))
        ket[i+1] = ket[i+1]*delta(indices_inner[i], commoninds(peps[size(peps)[1],i], peps[size(peps)[1],i+1]))
    end
    
    # the contraction of the bra and ket yields the first double_layer_env if we combine the resulting outer indices
    E_mpo = contract(bra,ket,maxdim=peps.double_contract_dim)
    E_mps = MPS((E_mpo.*combiner.(indices_outer, commoninds.(peps[size(peps)[1]], peps[size(peps)[1]-1]), tags="-1")).data)
    
    normE = norm(E_mps)
    peps.double_layer_envs[end] = Environment(E_mps/normE, log(normE))
    
    for i in size(peps)[1]-1:-1:2
        C = Array{Array{ITensor}}(undef, 2)
        bra = MPO(peps[i])
        ket = MPO(peps[i])

        # for the other rows we need 2 outer indices (one pointing up and one down which will be contracted with the double_layer_env of the previous iteration)
        for j in [-1,1]
            for k in 1:length(indices_outer)
                indices_outer[k] = Index(peps.bond_dim, "Ket_$(j)_$(k)")
            end
            ket = ket.*delta.(indices_outer, commoninds.(peps[i], peps[i+j]))
            C[Int((j+1)/2 +1)] = combiner.(indices_outer, commoninds.(peps[i], peps[i+j]), tags="$(j)")
        end

        for j in 1:length(indices_inner)
            indices_inner[j] = Index(peps.bond_dim, "Ket_inner_$(j)")
            ket[j] = ket[j]*delta(indices_inner[j], commoninds(peps[i,j], peps[i,j+1]))
            ket[j+1] = ket[j+1]*delta(indices_inner[j], commoninds(peps[i,j], peps[i,j+1]))
        end

        E_mpo = contract(bra,ket,maxdim=peps.double_contract_dim)
        E_mpo = E_mpo.*C[1]
        E_mpo = E_mpo.*C[2]
         
        # contracts the new row with the previous double_layer_env to obtain the next double_layer_env
        E_mps = apply(E_mpo.*delta.(reduce(vcat, collect.(inds.(E_mpo, "1"))), reduce(vcat, collect.(inds.(peps.double_layer_envs[i].env, "-1")))), peps.double_layer_envs[i].env, maxdim=peps.double_contract_dim)
    
        normE = norm(E_mps)
        peps.double_layer_envs[i-1] = Environment(E_mps/normE, log(normE) + peps.double_layer_envs[i].f)
    end

end

# generates a sample of a given peps along with pc and the top environments
function get_sample(peps::PEPS)
    S = Array{Int64}(undef, size(peps))
    
    indices_inner = Array{Index}(undef, size(peps)[2]-1)
    indices_outer = Array{Index}(undef, size(peps)[2])
    
    E = Array{ITensor}(undef, size(peps)[2]-1)          # contraction of unsampled tensors in a row from right to left
    sigma = Array{ITensor}(undef, size(peps)[2])        # contraction of already sampled tensors in a row from left to right
    
    env_top = Array{Environment}(undef, size(peps)[1]-1)
    
    P_S = ITensor()
    
    pc = 1
    for row in 1:size(peps)[1]
        bra = MPO(peps[row])
        ket = MPO(peps[row])
        
        if row != 1         # in every normal step we just contract the row with the env_top of the previous iteration
            bra = apply(bra, env_top[row-1].env, maxdim=peps.sample_dim)
            ket = apply(ket, env_top[row-1].env, maxdim=peps.sample_dim)
        end
        if row == 1         # in the first iteration we have to rename the inner indices
            for k in 1:length(indices_inner)
                indices_inner[k] = Index(peps.bond_dim, "Ket_$(k)")
                ket[k] = ket[k]*delta(indices_inner[k], commoninds(peps[row,k], peps[row,k+1]))
                ket[k+1] = ket[k+1]*delta(indices_inner[k], commoninds(peps[row,k], peps[row,k+1]))
            end
        end
        if row != size(peps)[1]     
            for k in 1:length(indices_outer)    # the outer indices of the ket are renamed in every step except for the last one
                indices_outer[k] = Index(peps.bond_dim, "Ket_out_$(k)")
                ket[k] = ket[k]*delta(indices_outer[k], commoninds(peps[row,k], peps[row+1,k]))
            end
        
            for i in size(peps)[2]:-1:2     # calculate E for the current row
                C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
                if i == size(peps)[2]
                    E[i-1] = contract(bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
                else
                    E[i-1] = contract(E[i]*bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
                end
            end
        else
            for i in size(peps)[2]:-1:2     # calculate E for the last row (no double_layer_env)
                if i == size(peps)[2]
                    E[i-1] = contract(bra[i]*ket[i])
                else
                    E[i-1] = contract(E[i]*bra[i]*ket[i])
                end
            end
        end
         
        # this loop goes now through every tensor in the fixed row
        for i in 1:size(peps)[2]
            ket[i] = delta(inds(ket[i], "phys_$(i)_$(row)"), Index(2, "ket_phys"))*ket[i]
            
            # sigma[i] is first used as the unsampled contraction of bra & ket with the double_layer_envs up to position i. Later it will be overwritten with the sampled contraction
            if row != size(peps)[1]
                C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
                sigma[i] = bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i]
            else
                sigma[i] = bra[i]*ket[i]
            end
              
            # P_S is the 2x2 matrix used for the sampling of S[row,i]
            if i == 1
                P_S = contract(E[i]*sigma[i])
            elseif i == size(peps)[2]
                P_S = contract(sigma[i-1]*sigma[i])
            else
                P_S = contract(sigma[i-1]*E[i]*sigma[i])
            end
                           
            p0 = P_S[1,1]
            p1 = P_S[2,2]
            # The actual sampling
            if rand() < p0/(p0+p1)
                S[row,i] = 0 
                pc *= p0
                p0 = p0/(p0+p1)
            else
                S[row,i] = 1
                pc *= p1
                p0 = p1/(p0+p1)
            end

            # now we have a sample for S[row,i] and can overwrite sigma with the sampled contraction
            if i == 1
                sigma[i] = sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)"))
                sigma[i] = sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(ket[i], "ket_phys")) / p0
            else
                sigma[i] = (sigma[i-1]*sigma[i])* ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)"))
                sigma[i] = (sigma[i-1]*sigma[i])* ITensor([(S[row,i]+1)%2, S[row,i]], inds(ket[i], "ket_phys")) / p0
            end
        end
        
        # These last steps are used to calculate env_top
        bra = bra.*[ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)")) for i in 1:size(peps)[2]]
            
        if row != size(peps)[1]
            norm_bra = norm(bra)
            if row == 1
                env_top[row] = Environment(MPS(bra.data)/norm_bra, log(norm_bra)) 
            else
                env_top[row] = Environment(bra/norm_bra, log(norm_bra)+env_top[row-1].f)
            end
        end 
        
    end
    
    return S, pc, env_top
end