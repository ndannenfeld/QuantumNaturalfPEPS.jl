# renames the indices changing_inds to new ones and stores them in indices_outer
function rename_indices!(ket, indices_outer, changing_inds)
    for i in 1:length(indices_outer)
        indices_outer[i] = Index(dim(changing_inds[i]), "Ket_$(i)")
        ket[i] = ket[i]*delta(changing_inds[i], indices_outer[i])
    end  
end

# renames all inner indices in a MPO/MPS
function rename_indices!(ket)
    indices_inner = Array{Index}(undef, length(ket)-1)
    for i in 1:length(indices_inner)
        indices_inner[i] = Index(maxlinkdim(ket), "Ket_inner_$(i)")
        old_ind = commoninds(ket[i], ket[i+1])

        ket[i] = ket[i]*delta(indices_inner[i], old_ind)
        ket[i+1] = ket[i+1]*delta(indices_inner[i], old_ind)
    end
end

# calculates a double layer environment row by contracting a peps_row with itself along the physical indices
# also combines the outgoing indices of the double layer
function generate_double_layer_env_row(peps_row, peps_row_above, contract_dim)
    indices_outer = Array{Index}(undef, length(peps_row))
    
    bra = conj(MPO(peps_row))
    ket = copy(MPO(peps_row))
     
    rename_indices!(ket)
    com_inds = commoninds.(peps_row, peps_row_above)
    rename_indices!(ket, indices_outer, com_inds)

    E_mpo = contract(bra,ket,maxdim=contract_dim)
    
    E_mps = MPS((E_mpo.*combiner.(indices_outer, com_inds, tags="-1")).data)
    # TODO: Fix this to use the new norm baked into Environment
    #normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    #return Environment(E_mps./normE, length(bra)*log(normE))
    return Environment(E_mps; normalize=true)
end

function generate_double_layer_env_row(peps_row, peps_row_above, peps_row_below, peps_double_env, contract_dim)
    C = Array{Array{ITensor}}(undef, 2)
    indices_outer = Array{Index}(undef, length(peps_row))

    bra = conj(MPO(peps_row))
    ket = copy(MPO(peps_row))
    
    rename_indices!(ket, indices_outer, commoninds.(peps_row, peps_row_above))
    C[1] = combiner.(indices_outer, commoninds.(peps_row, peps_row_above), tags="$(-1)")
    rename_indices!(ket, indices_outer, commoninds.(peps_row, peps_row_below))
    C[2] = combiner.(indices_outer, commoninds.(peps_row, peps_row_below), tags="$(1)")

    rename_indices!(ket)

    E_mpo = contract(bra,ket,maxdim=contract_dim)
    E_mpo = E_mpo.*C[1]
    E_mpo = E_mpo.*C[2]

    E_mps = apply(E_mpo.*delta.(reduce(vcat, collect.(inds.(E_mpo, "1"))), reduce(vcat, collect.(inds.(peps_double_env.env, "-1")))), peps_double_env.env, maxdim=contract_dim)

    # TODO: Fix this to use the new norm baked into Environment
    #normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    #return Environment(E_mps./normE, length(bra)*log(normE)+peps_double_env.f)
    return Environment(E_mps, peps_double_env.f; normalize=true)
end

# calculates the field double_layer_envs and the norm of peps
function update_double_layer_envs!(peps::PEPS)
    peps.double_layer_envs = generate_double_layer_envs(peps) 
end

function generate_double_layer_envs(peps::PEPS)
    double_layer_envs = Vector{Environment}(undef, size(peps, 1) - 1)
    # for every row we calculate the double layer environment
    double_layer_envs[end] = generate_double_layer_env_row(peps[size(peps, 1), :], peps[size(peps, 1)-1, :], peps.double_contract_dim)
    for i in size(peps, 1)-1:-1:2
        double_layer_envs[i-1] = generate_double_layer_env_row(peps[i, :], peps[i-1, :], peps[i+1, :], double_layer_envs[i], peps.double_contract_dim)
    end
    return double_layer_envs
end

# calculates the bra and the ket layer and applies (if available) already sampled rows (from above)
function get_bra_ket!(peps, row, indices_outer, env_top=nothing)
    bra = MPO(peps[row, :])
    ket = copy(bra)

    if row != 1      
        bra = apply(bra, env_top[row-1].env, maxdim=peps.sample_dim)
        ket = apply(ket, env_top[row-1].env, maxdim=peps.sample_dim)
    end
    rename_indices!(ket)
    
    if row != size(peps, 1)
        rename_indices!(ket, indices_outer, commoninds.(peps[row, :], peps[row+1, :]))
    end
    return conj(bra), ket
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_unsampled_Env_row!(bra, ket, peps, row, E, indices_outer)
    if row != size(peps, 1)
        C = combiner(indices_outer[end], commoninds(peps[row,size(peps, 2)], peps[row+1,size(peps, 2)]), tags = "1")

        E[end] = delta(inds(peps.double_layer_envs[row].env[end], "-1")[1], inds(C)[1])*C*bra[end]*ket[end]
        E[end] = E[end]*peps.double_layer_envs[row].env[end]
        for i in size(peps, 2)-1:-1:2
            C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
            E[i-1] = E[i]*peps.double_layer_envs[row].env[i]*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*C*ket[i]*bra[i]
        end
    else
        E[end] = bra[end]*ket[end]
        for i in size(peps, 2)-1:-1:2
            E[i-1] = E[i]*bra[i]*ket[i]
        end
    end
end

# returns the 2x2 matrix ρ_r which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)
function get_reduced_ρ(bra, ket, peps, row, i, E, indices_outer, sigma)
    ket[i] = delta(siteind(peps,row,i), Index(2, "ket_phys"))*ket[i]
   
    if row != size(peps, 1)
        C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
        sigma_1 = bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i]
    else
        sigma_1 = bra[i]*ket[i]
    end

    
    if i == 1
        ρ_r = E[i]*sigma_1
    elseif i == size(peps, 2)
        ρ_r = sigma*sigma_1
    else
        ρ_r = sigma*E[i]
        ρ_r = ρ_r*sigma_1
    end 
    
    return ρ_r, sigma_1
end

# samples from ρ_r and updates pc
function sample_ρr(ρ_r)
    # TODO: Generalize this to more than 2 states
    p0 = abs(ρ_r[1,1])
    p1 = abs(ρ_r[2,2])
   
    @assert imag(ρ_r[1,1]) < 1e-6
    if rand() < p0/(p0+p1) 
        return 0, p0/(p0+p1)
    else
        return 1, p1/(p0+p1)
    end
end

# after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
function update_sigma(peps, sigma, sigma_1, S, i, row, norm_factor)
    s = (sigma_1*sigma)* ITensor([(S+1)%2, S], siteind(peps, row, i))
    return (s)* ITensor([(S+1)%2, S], inds(sigma_1, "ket_phys")) / norm_factor
end

# generates a sample of a given peps along with pc and the top environments
function get_sample(peps::PEPS)
    S = Array{Int64}(undef, size(peps))
    
    indices_inner = Array{Index}(undef, size(peps, 2)-1)
    indices_outer = Array{Index}(undef, size(peps, 2))
    
    E = Array{ITensor}(undef, size(peps, 2)-1)
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    
    ρ_r = ITensor()
    
    logpc = 0
    # we loop through every row
    for row in 1:size(peps, 1)
        sigma = 1
        bra, ket = get_bra_ket!(peps, row, indices_outer, env_top)
        
        # we then calculate the unsampled environment (in one row)
        calculate_unsampled_Env_row!(bra, ket, peps, row, E, indices_outer)

        # then we loop through the different sites in one row
        for i in 1:size(peps, 2)
            
            # calculate the 2x2 matrix from which we sample
            ρ_r, sigma_1 = get_reduced_ρ(bra, ket, peps, row, i, E, indices_outer, sigma)
            
            # sample from ρ_r
            S[row, i], pc = sample_ρr(ρ_r)
            logpc += log(pc)
            
            # store the contraction of sampled sites in sigma
            sigma = update_sigma(peps, sigma, sigma_1, S[row, i], i, row, pc)                        
        end
        
        # the sampled bra is used to generate the top environments
        bra = bra.*[ITensor([(S[row,i]+1)%2, S[row,i]], siteind(peps, row, i)) for i in 1:size(peps, 2)]
        # TODO: Fix this to use the new norm baked into Environment
        if row != size(peps, 1)
            #norm_bra = maximum(abs.(reshape(Array(bra[1], inds(bra[1])), :))) 
            if row == 1
                #env_top[row] = Environment(MPS(bra.data)./norm_bra, length(bra)*log(norm_bra)) 
                env_top[row] = Environment(MPS(bra.data); normalize=true)
            else
                #env_top[row] = Environment(bra./norm_bra, length(bra)*log(norm_bra)+env_top[row-1].f)
                env_top[row] = Environment(MPS(bra.data), env_top[row-1].f; normalize=true)
            end
        end
    end
    
    return S, logpc, env_top
end