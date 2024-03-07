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

function init_PEPS(Lx::Int64, Ly::Int64, phys_dim::Int64, bond_dim::Int64)
    tensors = Array{ITensor}(undef, Lx,Ly)

    # initializing bonds
    indices = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        indices[i] = Index(bond_dim, "ind_$(i)")
    end
    
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



function get_logψ_and_envs(peps::PEPS, S::Array{Int64,2}, Env_top = nothing)
    if Env_top == nothing
        Env_top = Array{Environment}(undef, size(S,1)-1)
    end
    Env_down = Array{Environment}(undef, size(S,1)-1)
    out = Array{Float64}(undef, 2)
    for i in 1:size(S,1)
        i_prime = size(S,1)+1-i 
        if i == 1
            Env = MPS([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)])
            normE = norm(Env)
            Env_top[i] = Environment(Env/normE, log(normE))
            
            Env = MPS([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)])
            normE = norm(Env)
            Env_down[i] = Environment(Env/normE, log(normE))
        elseif i == size(S,1)
            out[1] = inner(Env_top[i-1].env,MPS([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)])) * exp(Env_top[i-1].f)
            out[2] = inner(Env_down[i-1].env,MPS([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)])) * exp(Env_down[i-1].f)
        else
            Env = apply(MPO([peps[i,j]*ITensor([(S[i,j]+1)%2, S[i,j]], inds(peps[i,j],"phys_$(j)_$(i)")) for j in 1:size(S,2)]), Env_top[i-1].env, maxdim=peps.contract_dim)
            normE = norm(Env)
            Env_top[i] = Environment(Env/normE, log(normE)+Env_top[i-1].f)
            
            Env = apply(MPO([peps[i_prime,j]*ITensor([(S[i_prime,j]+1)%2, S[i_prime,j]], inds(peps[i_prime,j],"phys_$(j)_$(i_prime)")) for j in 1:size(S,2)]), Env_down[i-1].env, maxdim=peps.contract_dim)
            normE = norm(Env)
            Env_down[i] = Environment(Env/normE, log(normE)+Env_down[i-1].f)
        end
    end
    return mean(out), Env_top, Env_down
end

function inner_peps(psi::PEPS, psi2::PEPS)
    x = 1
    for i in 1:size(psi)[1]
        for j in 1:size(psi)[2]
            x *= psi[i,j]*psi2[i,j]*delta(inds(psi[i,j], "phys_$(j)_$(i)"), inds(psi2[i,j], "phys_$(j)_$(i)"))
        end
    end
    return x[1]
end

function differentiate(E::Array{MPS,2}, f::Array{Float64,2}, peps::PEPS, S::Array{Array{Float64,1},2}, i::Int64, j::Int64)
    if i == 1
        return exp(f[end,2])*contract(E[end,2].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)])
    elseif i == size(S,2)
        return exp(f[end,1])*contract(E[end,1].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)])
    end   
    return exp(f[i-1,1])*exp(f[end-i+1,2])*contract(E[i-1,1].*(E[end-i+1,2].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)]))
end

function update_double_layer_envs!(peps::PEPS)
    indices_outer = Array{Index}(undef, size(peps)[2])
    for i in 1:length(indices_outer)
        indices_outer[i] = Index(peps.bond_dim, "Ket_$(i)")
    end
    
    bra = MPO(peps[size(peps)[1]])
    ket = MPO(peps[size(peps)[1]].*delta.(commoninds.(peps[size(peps)[1]], peps[size(peps)[1]-1]), indices_outer))
    
    indices_inner = Array{Index}(undef, size(peps)[2]-1)
    for i in 1:length(indices_inner)
        indices_inner[i] = Index(peps.bond_dim, "Ket_inner_$(i)")
        ket[i] = ket[i]*delta(indices_inner[i], commoninds(peps[size(peps)[1],i], peps[size(peps)[1],i+1]))
        ket[i+1] = ket[i+1]*delta(indices_inner[i], commoninds(peps[size(peps)[1],i], peps[size(peps)[1],i+1]))
    end
    
    E_mpo = contract(bra,ket,maxdim=peps.double_contract_dim)
    E_mps = MPS((E_mpo.*combiner.(indices_outer, commoninds.(peps[size(peps)[1]], peps[size(peps)[1]-1]), tags="-1")).data)
    
    normE = norm(E_mps)
    peps.double_layer_envs[end] = Environment(E_mps/normE, log(normE))
        
    for i in size(peps)[1]-1:-1:2
        C = Array{Array{ITensor}}(undef, 2)
        bra = MPO(peps[i])
        ket = MPO(peps[i])
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
         
        E_mps = apply(E_mpo.*delta.(reduce(vcat, collect.(inds.(E_mpo, "1"))), reduce(vcat, collect.(inds.(peps.double_layer_envs[i].env, "-1")))), peps.double_layer_envs[i].env, maxdim=peps.double_contract_dim)
    
        normE = norm(E_mps)
        peps.double_layer_envs[i-1] = Environment(E_mps/normE, log(normE) + peps.double_layer_envs[i].f)
    end

end


function get_sample(peps::PEPS)
    S = Array{Int64}(undef, size(peps))
    
    indices_inner = Array{Index}(undef, size(peps)[2]-1)
    indices_outer = Array{Index}(undef, size(peps)[2])
    
    E = Array{ITensor}(undef, size(peps)[2]-1)
    sigma = Array{ITensor}(undef, size(peps)[2])
    
    env_top = Array{Environment}(undef, size(peps)[1]-1)
    
    P_S = ITensor()
    
    pc = 1
    for row in 1:size(peps)[1]
        bra = MPO(peps[row])
        ket = MPO(peps[row])
        
        if row != 1
            bra = apply(bra, env_top[row-1].env, maxdim=peps.sample_dim)
            ket = apply(ket, env_top[row-1].env, maxdim=peps.sample_dim)
        end
        if row == 1
            for k in 1:length(indices_inner)
                indices_inner[k] = Index(peps.bond_dim, "Ket_$(k)")
                ket[k] = ket[k]*delta(indices_inner[k], commoninds(peps[row,k], peps[row,k+1]))
                ket[k+1] = ket[k+1]*delta(indices_inner[k], commoninds(peps[row,k], peps[row,k+1]))
            end
        end
        if row != size(peps)[1]
            for k in 1:length(indices_outer)
                indices_outer[k] = Index(peps.bond_dim, "Ket_out_$(k)")
                ket[k] = ket[k]*delta(indices_outer[k], commoninds(peps[row,k], peps[row+1,k]))
            end
        end
        if row != size(peps)[1]
            for i in size(peps)[2]:-1:2
                C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
                if i == size(peps)[2]
                    E[i-1] = contract(bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
                else
                    E[i-1] = contract(E[i]*bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
                end
            end
        else
            for i in size(peps)[2]:-1:2
                if i == size(peps)[2]
                    E[i-1] = contract(bra[i]*ket[i])
                else
                    E[i-1] = contract(E[i]*bra[i]*ket[i])
                end
            end
        end
         
        for i in 1:size(peps)[2]
            ket[i] = delta(inds(ket[i], "phys_$(i)_$(row)"), Index(2, "ket_phys"))*ket[i]
            
            if row != size(peps)[1]
                C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
                sigma[i] = bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i]
            else
                sigma[i] = bra[i]*ket[i]
            end
                
            if i == 1
                P_S = contract(E[i]*sigma[i])
            elseif i == size(peps)[2]
                P_S = contract(sigma[i-1]*sigma[i])
            else
                P_S = contract(sigma[i-1]*E[i]*sigma[i])
            end
                           
            p0 = P_S[1,1]
            p1 = P_S[2,2]
            println("0: $(p0), 1: $(p1)")
            if rand() < p0/(p0+p1)
                S[row,i] = 0 
                pc *= p0
            else
                S[row,i] = 1
                pc *= p1
                p0 = p1
            end
            if i == 1
                sigma[i] = sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)"))
                sigma[i] = sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(ket[i], "ket_phys")) / p0
            else
                sigma[i] = sigma[i-1]*sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)"))
                sigma[i] = sigma[i-1]*sigma[i]* ITensor([(S[row,i]+1)%2, S[row,i]], inds(ket[i], "ket_phys")) / p0
            end
        end
        
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