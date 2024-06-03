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
    
    normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    return Environment(E_mps./normE, length(bra)*log(normE))
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

    normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    return Environment(E_mps./normE, length(bra)*log(normE)+peps_double_env.f)
end

# calculates the field double_layer_envs and the norm of peps
function update_double_layer_envs!(peps::PEPS)
    peps.double_layer_envs = generate_double_layer_envs(peps)

    # TODO do we need this?
    # We also calculate the (log-)norm of the peps as it is used in get_sample to calculate p_c
    E_mpo = generate_double_layer_env_row(peps[1], peps[2], peps.double_contract_dim)
    E_mpo.env = E_mpo.env .*delta.(reduce(vcat, collect.(inds.(E_mpo.env, "-1"))), reduce(vcat, collect.(inds.(peps.double_layer_envs[1].env, "-1"))))
    
    for i in 1:length(E_mpo.env)
        E_mpo.env[i] = (E_mpo.env[i]*peps.double_layer_envs[1].env[i])
    end

    peps.norm = real(log(Complex(contract(E_mpo.env)[1])))+(peps.double_layer_envs[1].f + E_mpo.f)    
end

function generate_double_layer_envs(peps::PEPS)
    double_layer_envs = Vector{Environment}(undef, size(peps, 1) - 1)
    # for every row we calculate the double layer environment
    double_layer_envs[end] = generate_double_layer_env_row(peps[size(peps, 1)], peps[size(peps, 1)-1], peps.double_contract_dim)
    for i in size(peps, 1)-1:-1:2
        double_layer_envs[i-1] = generate_double_layer_env_row(peps[i], peps[i-1], peps[i+1], double_layer_envs[i], peps.double_contract_dim)
    end
    return double_layer_envs
end

# calculates the bra and the ket layer and applies (if available) already sampled rows (from above)
function get_bra_ket!(peps, row, indices_outer, env_top=nothing)
    bra = MPO(peps[row])
    ket = copy(bra)

    if row != 1      
        bra = apply(bra, env_top[row-1].env, maxdim=peps.sample_dim)
        ket = apply(ket, env_top[row-1].env, maxdim=peps.sample_dim)
    end
    rename_indices!(ket)
    
    if row != size(peps, 1)
        rename_indices!(ket, indices_outer, commoninds.(peps[row], peps[row+1]))
    end
    return conj(bra), ket
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_E!(bra, ket, peps, row, E, indices_outer)
    if row != size(peps, 1)
        C = combiner(indices_outer[end], commoninds(peps[row,size(peps, 2)], peps[row+1,size(peps, 2)]), tags = "1")
        E[end] = contract(bra[end]*ket[end]*C*delta(inds(peps.double_layer_envs[row].env[end], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[end])

        for i in size(peps, 2)-1:-1:2
            C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
            E[i-1] = contract(E[i]*bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
        end
    else
        E[end] = contract(bra[end]*ket[end])
        for i in size(peps, 2)-1:-1:2
            E[i-1] = contract(E[i]*bra[i]*ket[i])
        end
    end
end

# returns the 2x2 matrix P_S which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)
function get_PS(bra, ket, peps, row, i, E, indices_outer, sigma)
    ket[i] = delta(inds(ket[i], "phys_$(i)_$(row)"), Index(2, "ket_phys"))*ket[i]
   
    if row != size(peps, 1)
        C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
        sigma_1 = bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i]
    else
        sigma_1 = bra[i]*ket[i]
    end

    
    if i == 1
        P_S = contract(E[i]*sigma_1)
    elseif i == size(peps, 2)
        P_S = contract(sigma*sigma_1)
    else
        P_S = contract(sigma*E[i])
        P_S = contract(P_S*sigma_1)
    end 
    
    return P_S, sigma_1
end

# samples from P_S and updates pc
function sample_PS!(P_S, pc)
    p0 = abs(P_S[1,1])
    p1 = abs(P_S[2,2])
   
    @assert imag(P_S[1,1]) < 10e-6
    if rand() < p0/(p0+p1)
        pc += log(p0/(p0+p1))
        return p0/(p0+p1), 0, pc
    else
        pc += log(p1/(p0+p1))
        return p1/(p0+p1), 1, pc
    end
end

# after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
function update_sigma(sigma, sigma_1, S, i, row, norm_factor)
    s = (sigma_1*sigma)* ITensor([(S+1)%2, S], inds(sigma_1, "phys_$(i)_$(row)"))
    return (s)* ITensor([(S+1)%2, S], inds(sigma_1, "ket_phys")) / norm_factor
end

# generates a sample of a given peps along with pc and the top environments
function get_sample(peps::PEPS)
    S = Array{Int64}(undef, size(peps))
    
    indices_inner = Array{Index}(undef, size(peps, 2)-1)
    indices_outer = Array{Index}(undef, size(peps, 2))
    
    E = Array{ITensor}(undef, size(peps, 2)-1)
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    
    P_S = ITensor()
    
    psi_S = 0
    pc = 0
    # we loop through every row
    for row in 1:size(peps, 1)
        sigma = 1
        bra, ket = get_bra_ket!(peps, row, indices_outer, env_top)
        
        # we then calculate the unsampled environment (in one row)
        calculate_E!(bra, ket, peps, row, E, indices_outer)

        # then we loop through the different sites in one row
        for i in 1:size(peps, 2)
            
            # calculate the 2x2 matrix from which we sample
            P_S, sigma_1 = get_PS(bra, ket, peps, row, i, E, indices_outer, sigma)
            
            # sample from P_S
            norm_factor, S[row,i], pc = sample_PS!(P_S, pc)
            
            # store the contraction of sampled sites in sigma
            sigma = update_sigma(sigma, sigma_1, S[row,i], i, row, norm_factor)                        
        end
        
        # the sampled bra is used to generate the top environments
        bra = bra.*[ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)")) for i in 1:size(peps, 2)]
            
        if row != size(peps, 1)
            norm_bra = maximum(abs.(reshape(Array(bra[1], inds(bra[1])), :)))
            if row == 1
                env_top[row] = Environment(MPS(bra.data)./norm_bra, length(bra)*log(norm_bra)) 
            else
                env_top[row] = Environment(bra./norm_bra, length(bra)*log(norm_bra)+env_top[row-1].f)
            end
        else
            psi_S = 2*real(log(Complex(contract(bra)[1])) + env_top[row-1].f)
        end
    end
    
    return S, pc,psi_S ,env_top
end