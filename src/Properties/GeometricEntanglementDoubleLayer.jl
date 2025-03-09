function update_sigma2(peps, sigma, t::ITensor, i, row, norm_factor)
    in_prime = ITensors.inds(sigma, "ket_phys")
    tprime = t * delta(in_prime, siteind(peps, row, i))
    s = sigma* t * conj(tprime)
    return s / norm_factor
end
function get_largest_eigenvec(ρ_r::ITensor; norm=false)
    val, vec = eigen(ρ_r, inds(ρ_r)[1], inds(ρ_r)[2])
            
    ind_ = argmax(abs.(val.tensor.storage))
    Z = 1
    if norm
        Z = sum(ρ_r[i, i] for i in 1:dims(ρ_r)[1])
    end
    max_val = val.tensor.storage[ind_] / Z
    prod_tensor = vec * onehot(inds(val)[2] => ind_)
    return max_val, prod_tensor
end
function geometric_entanglement_doublelayer(peps::AbstractPEPS; S=zeros(Int, size(peps)...))
    indices_outer = Array{Index}(undef, size(peps, 2))
    E = Array{ITensor}(undef, size(peps, 2)-1)
    
    env_top = Array{Environment}(undef, size(peps, 1)-1)
    
    ρ_r = ITensor()
    Us = Matrix{ITensor}(undef, size(peps, 1), size(peps, 2))
    
    logpc = 0
    # we loop through every row
    global bra
    for row in 1:size(peps, 1)
        sigma = 1
        bra, ket = get_bra_ket!(peps, row, indices_outer, env_top)
        
        # we then calculate the unsampled environment (in one row)
        calculate_unsampled_Env_row!(bra, ket, peps, row, E, indices_outer)
        
        prod_tensors = ITensor[]
        # then we loop through the different sites in one row
        for i in 1:size(peps, 2)
            
            # calculate the 2x2 matrix from which we sample
            ρ_r, sigma = get_reduced_ρ(bra, ket, peps, row, i, E, indices_outer, sigma)
            
            #S[row, i], pc = sample_ρr(ρ_r)
            pc, prod_tensor = get_largest_eigenvec(ρ_r; norm=true)
            push!(prod_tensors, prod_tensor)
            
            U = get_rotator(prod_tensor, S[row, i]; factor=1)
            Us[row, i] = U
            logpc += log(pc)
            
            # store the contraction of sampled sites in sigma
            sigma = update_sigma2(peps, sigma, prod_tensor, i, row, pc)                     
        end
        
        # the sampled bra is used to generate the top environments
        bra = bra .* prod_tensors
        if row != size(peps, 1) 
            if row == 1
                env_top[row] = Environment(MPS(bra.data); normalize=true)
            else
                env_top[row] = Environment(MPS(bra.data), env_top[row-1].f; normalize=true)
            end
        end
    end
    logψ = log(contract(bra)[]) + env_top[end].f
    return logpc, logψ, Us
end