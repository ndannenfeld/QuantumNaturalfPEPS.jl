# renames the indices changing_inds to new ones and stores them in indices_outer
function rename_indices!(ket, indices_outer, changing_inds)
    for j in 1:length(indices_outer)
        indices_outer[j] = Index(dim(changing_inds[j]), "Ket_$(j)")
        
        ket[j] = subsinds(ket[j], changing_inds[j], indices_outer[j])
    end  
end

# renames all inner indices in a MPO/MPS
function rename_indices!(ket)
    for j in 1:length(ket)-1
        old_ind = commoninds(ket[j], ket[j+1])[1]
        new_ind = Index(dim(old_ind), "Ket_inner_$(j)")

        ket[j] = subsinds(ket[j], old_ind, new_ind)
        ket[j+1] = subsinds(ket[j+1], old_ind, new_ind)
    end
end
allindices(ket) = unique(Iterators.flatten(inds.(ket)))
otherinds(ket, siteinds_) = setdiff(allindices(ket), siteinds_)

# calculates a double layer environment row by contracting a peps_row with itself along the physical indices
# also combines the outgoing indices of the double layer
function generate_double_layer_env_row(peps_row, peps_row_above, maxdim; cutoff=1e-13)
    indices_outer = Array{Index}(undef, length(peps_row))
    
    bra = conj(MPO(peps_row))
    ket = copy(MPO(peps_row))
    
    #links = otherinds(ket, siteinds(peps, i, :))
    # prime!(links, ket) # TODO do something like this
    rename_indices!(ket)
    com_inds = commoninds.(peps_row, peps_row_above)
    rename_indices!(ket, indices_outer, vcat(com_inds...))

    E_mpo = contract(bra, ket; maxdim, cutoff)
    
    E_mps = MPS((E_mpo.*combiner.(indices_outer, com_inds, tags="up")).data)
    return Environment(E_mps; normalize=true)
end

function generate_double_layer_env_row(peps_row, peps_row_above, peps_row_below, peps_double_env, maxdim; cutoff=1e-13)
    C = Array{ITensor}(undef, 2, length(peps_row))
    indices_outer = Array{Index}(undef, length(peps_row))

    bra = conj(MPO(peps_row))
    ket = copy(MPO(peps_row)) # TODO: Why copy here?
    
    com_inds = commoninds.(peps_row, peps_row_above)
    rename_indices!(ket, indices_outer, vcat(com_inds...))
    for j in 1:length(peps_row)
        C[1,j] = combiner(indices_outer[j], com_inds[j], tags="up")
    end

    com_inds = commoninds.(peps_row, peps_row_below)
    rename_indices!(ket, indices_outer, vcat(com_inds...))
    for j in 1:length(peps_row)
        C[2,j] = combiner_tar(indices_outer[j], com_inds[j]; target_ind=reduce(vcat, inds(peps_double_env.env[j], "up")), tags="down")
    end

    rename_indices!(ket)

    E_mpo = bra.*ket
    
    E_mpo = E_mpo.*C[1, :]
    E_mpo = E_mpo.*C[2, :]

    E_mps = contract(E_mpo, peps_double_env.env; maxdim, cutoff)

    return Environment(E_mps, peps_double_env.f; normalize=true)
end

# calculates the field double_layer_envs and the norm of peps
function update_double_layer_envs!(peps::PEPS)
    peps.double_layer_envs = generate_double_layer_envs(peps) 
end

function generate_double_layer_envs(peps::PEPS)
    Lx = size(peps, 1)
    double_layer_envs = Vector{Environment}(undef, Lx - 1)
    # for every row we calculate the double layer environment
    double_layer_envs[end] = generate_double_layer_env_row(peps[Lx, :], peps[Lx-1, :], peps.double_contract_dim; cutoff=peps.double_contract_cutoff)
    for i in Lx-1:-1:2
        double_layer_envs[i-1] = generate_double_layer_env_row(peps[i, :], peps[i-1, :], peps[i+1, :], double_layer_envs[i], peps.double_contract_dim; cutoff=peps.double_contract_cutoff)
    end
    return double_layer_envs
end

# calculates the bra and the ket layer and applies (if available) already sampled rows (from above)
function get_bra_ket(peps, i, indices_outer, env_top=nothing)
    #ket = copy(MPO(peps[i, :])) # TODO: Why copy here?
    #bra = conj(MPO(peps[i, :]))

    ket = MPO(peps[i, :])
    if i != 1   
        ket = contract(ket, env_top[i-1].env; maxdim=peps.sample_dim, cutoff=peps.sample_cutoff)
        #bra = contract(bra, conj(env_top[i-1].env); maxdim=peps.sample_dim, cutoff=peps.sample_cutoff) # TODO: Why are you doing the same operation twice?
    end
    bra = conj(ket)

    rename_indices!(ket) # TODO: Avoid indices renaming by using primelevels prime!(ket), like this is very confusing
    
    if i != size(peps, 1)
        rename_indices!(ket, indices_outer, vcat(commoninds.(peps[i, :], peps[i+1, :])...))
    end
    return bra, ket
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_unsampled_Env_row!(bra, ket, peps, i, E, indices_outer)
    if i != size(peps, 1)
        com_inds = commoninds(peps[i,size(peps, 2)], peps[i+1,size(peps, 2)])
        combined_indx = inds(peps.double_layer_envs[i].env[end], "up")[1]
        C = combiner_tar(indices_outer[end], com_inds; target_ind=combined_indx)

        E[end] = peps.double_layer_envs[i].env[end]*C*bra[end]*ket[end]
        for j in size(peps, 2)-1:-1:2
            com_inds = commoninds(peps[i,j], peps[i+1,j])
            combined_indx = inds(peps.double_layer_envs[i].env[j], "up")[1]
            C = combiner_tar(indices_outer[j], com_inds; target_ind=combined_indx)

            uncombined_double_layer = peps.double_layer_envs[i].env[j] * C
            E[j-1] = E[j] * ket[j]
            E[j-1] *= uncombined_double_layer
            E[j-1] *= bra[j]
        end
    else
        E[end] = bra[end]*ket[end]
        for j in size(peps, 2)-1:-1:2
            E[j-1] = ket[j]*E[j]
            E[j-1] *= bra[j]
        end
    end
end

# returns the phys_dimxphys_sim matrix ρ_r which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)
function get_reduced_ρ(bra, ket, peps, i, j, E, indices_outer, sigma)
    ket[j] = prime(ket[j], siteind(peps, i, j))
    #ket[j] = delta(siteind(peps, i, j), Index(2, "ket_phys"))*ket[j] # TODO: Fix this for compatibility with phys_dim!=2, using siteinds(peps, j, j)' would be better
   
    if i != size(peps, 1)
        com_inds = commoninds(peps[i,j], peps[i+1,j])
        C = combiner_tar(indices_outer[j], com_inds; target_ind=inds(peps.double_layer_envs[i].env[j], "up")[1])
        
        uncombined_double_layer = peps.double_layer_envs[i].env[j]*C
        sigma *= ket[j]
        sigma *= uncombined_double_layer
        sigma *= bra[j]
    else
        sigma *= ket[j]
        sigma *= bra[j]
    end

    if j == 1
        ρ_r = E[j]*sigma
    elseif j == size(peps, 2)
        ρ_r = sigma
    else
        ρ_r = sigma*E[j]
    end 
    
    return ρ_r, sigma
end

# samples from ρ_r and updates pc
function sample_ρr(ρ_r)
    k = size(ρ_r, 1) 
    T = real(eltype(ρ_r))
    p = Vector{T}(undef, k)
    for i in 1:k
        p[i] = abs(ρ_r[i, i])
        @assert imag(ρ_r[i, i]) / (p[i] + 1e-10) < 1e-8 "ρ_r is not real $(ρ_r[i,i])"
    end
    i = sample_p(p, normalize=true)
    return i-1, p[i]
end

function sample_p(probs::Vector{T}; normalize=true) where T<:Real
    if normalize
        probs ./= sum(probs)
    end
    r = rand()
    psum = 0
    for (i, p) in enumerate(probs)
        psum += p
        if psum > r
            return i
        end
    end
    error("probs is not normalized sum(probs)=$(sum(probs))")
end

# generates a sample of a given peps along with pc and the top environments
function get_sample(peps::PEPS; mode::Symbol=:full, alg="densitymatrix")
    S = Array{Int64}(undef, size(peps))
    
    indices_outer = Array{Index}(undef, size(peps, 2))
    
    E = Array{ITensor}(undef, size(peps, 2)-1)
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    
    ρ_r = ITensor()
    
    logpc = 0
    # we loop through every row
    for i in 1:size(peps, 1)
        sigma = 1
        bra, ket = get_bra_ket(peps, i, indices_outer, env_top)
        
        # we then calculate the unsampled environment (in one row)
        calculate_unsampled_Env_row!(bra, ket, peps, i, E, indices_outer)

        # then we loop through the different sites in one row
        for j in 1:size(peps, 2)
            
            # calculate the phys_dimxphys_dim matrix from which we sample
            ρ_r, sigma = get_reduced_ρ(bra, ket, peps, i, j, E, indices_outer, sigma)
            
            # sample from ρ_r
            S[i, j], pc = sample_ρr(ρ_r)
            logpc += log(pc)
            
            # after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
            site = siteind(peps, i, j)
            sigma *= get_projector(S[i, j], site)
            sigma *= get_projector(S[i, j], site')
            sigma /= pc                  
        end
        
        if mode === :fast
            # the sampled bra is used to generate the top environments
            ket = conj(bra) .* [get_projector(S[i, j], siteind(peps, i, j)) for j in 1:size(peps, 2)]
            if i == 1
                env_top[i] = Environment(MPS(ket.data); normalize=true)
            elseif i != size(peps, 1) 
                env_top[i] = Environment(MPS(ket.data), env_top[i-1].f; normalize=true)
            end

        elseif mode === :full
             # TODO: Should we be recalculating the top environment here? Is it slower?
            if i == 1
                peps_projected_1 = get_projected(peps, S, 1, :)
                env_top[1] = generate_env_row(peps_projected_1, peps.contract_dim; alg, cutoff=peps.contract_cutoff)
            elseif i != size(peps, 1) 
                peps_projected_row = get_projected(peps, S, i, :)
                env_top[i] = generate_env_row(peps_projected_row, peps.contract_dim; env_row_above=env_top[i-1], alg, cutoff=peps.contract_cutoff)
            end  
        end
        
       
    end
    
    return S, logpc, env_top
end