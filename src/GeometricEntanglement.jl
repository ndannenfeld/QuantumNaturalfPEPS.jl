function constant_update(t, p, factor=0.1)
    d1 = t .- p
    d = sign.(d1) .* min.(factor, abs.(d1))
    if norm(t .- d) < 1e-4
        return constant_update(t, p, 0.9*factor)
    end
    return t - d
end

function exp_update(t, p, factor=0.1)
    return factor .* p .+ sqrt(1-factor^2) .* t
end

function basis_change(peps, ops::Array{ITensor, 2})
    peps = deepcopy(peps)
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        peps.tensors[i, j] = noprime(peps[i, j] * ops[i, j])
    end
    return peps
end

function basis_change(peps, ops::Array{ITensor, 1}, i::Int)
    peps = deepcopy(peps)
    for j in 1:size(peps, 2)
        peps.tensors[i, j] = noprime(peps[i, j] * ops[j])
    end
    return peps
end

function basis_change(peps; gate="H")
    peps = deepcopy(peps)
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            U = op(gate, siteind(peps, i, j))
            peps.tensors[i, j] = noprime(peps[i, j] * U)
        end
    end
    return peps
end

function get_rotator(Ok_Tensor, S; update=exp_update, factor=0.05)
    inds_ = inds(Ok_Tensor)[1]

    t = Ok_Tensor.tensor.storage
    t = t ./ norm(t)
    v = randn(2)
    
    if S == 0
        p = [1, 0]
        t = update(p, t, factor)
        t ./= norm(t)
        v = v - t' * v * t
        v ./= norm(v) * sign(v[2])
        u = zeros(2, 2)
        u[:,1] = t
        u[:,2] = v
    else
        p = [0, 1]
        t = update(p, t, factor)
        t ./= norm(t)
        v = v - t' * v * t
        v ./= norm(v) * sign(v[2])
        u = zeros(2, 2)
        u[:,2] = t
        u[:,1] = v
    end
    return ITensor(u, inds_, inds_')
end
function get_rotator(i, j, peps, env_top, env_down, h_envs_r, h_envs_l)
    Ok_Tensor = peps[i, j]
    f = 0
    if j != size(peps, 2)
        Ok_Tensor *= h_envs_r[i,j]
    end
    if i != 1
        Ok_Tensor *= env_top[i-1].env[j]
        f += env_top[i-1].f
    end
    if i != size(peps, 1)
        Ok_Tensor *= env_down[end-i+1].env[j]
        f += env_down[end-i+1].f
    end
    if j != 1
        Ok_Tensor *= h_envs_l[i,j-1]
    end
    return Ok_Tensor
end

function rotate_to_product_state(peps, S=zeros(Int, size(peps)...); k=20, verbose=false, kwargs...)
    local c
    logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
    h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S)

    for i in 1:size(peps, 1)
        Ok_Tensors = [get_rotator(i, j, peps, env_top, env_down, h_envs_r, h_envs_l) for j in 1:size(peps, 2)]
        Us = [get_rotator(Ok_Tensors[i, j], S[i,j]; kwargs...) for j in 1:size(peps, 2)]
        
        peps = basis_change(peps, Us, i)
        c = maximum([minimum(abs.(Ok_Tensor)/ norm(Ok_Tensor)) for Ok_Tensor in Ok_Tensors])
        if verbose
            @info "$i: $(round(c, digits=5)), $(round(logψ, digits=5))"
        end
        if c < 1e-5
            return peps
        end
    end
    @info "Only converged to $c in $k steps"
    return peps
end

function get_logZ(peps; k=10)
    logpcs = []
    logψs = []
    for i in 1:k
        S, logpc = QuantumNaturalfPEPS.get_sample(peps);
        logψ,  = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S)
        push!(logψs, logψ)
        push!(logpcs, logpc)
    end
    log_ratios =  2 .* real.(logψs) .- logpcs
    logZ = QuantumNaturalfPEPS.logsumexp(log_ratios) - log(length(logpcs))
    #logZ_second = QuantumNaturalfPEPS.logsumexp(2. .* log_ratios) - log(length(logpcs))
            
    return logZ
end

function geometric_entanglement(peps; k_opt=300, k_incr=10, k_samples=10, factor=0.1, kwargs...)
    if peps.double_layer_envs === nothing
        update_double_layer_envs!(peps)
    end
    S, = get_sample(peps); # draw a sample
    peps_rot = peps
    peps_rot = rotate_to_product_state(peps_rot, S; k=30, factor, kwargs...)
    peps_rot = rotate_to_product_state(peps_rot, S; k=k_opt, factor=0.9, kwargs...)

    logψ,  = get_logψ_and_envs(peps_rot, S)
    logZ = get_logZ(peps_rot; k=k_samples)
    return exp(2 * real(logψ) - logZ)
end


### Sweeping

function rotate_to_product_state_sweep_down(peps, logψ, env_top, env_down, h_envs_r, h_envs_l; S=zeros(Int, size(peps)...), verbose=false, kwargs...)
    c = 0
    for i in 1:size(peps, 1)
        view_r = @view h_envs_r[i,:]
        view_l = @view h_envs_l[i, :]
        QuantumNaturalfPEPS.get_horizontal_envs!(peps, env_top, env_down, S, i, view_r, view_l)
        h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S)
        
        Ok_Tensors = [QuantumNaturalfPEPS.get_rotator(i, j, peps, env_top, env_down, h_envs_r, h_envs_l) for j in 1:size(peps, 2)]
        Us = [QuantumNaturalfPEPS.get_rotator(Ok_Tensors[j], S[i, j]; kwargs...) for j in 1:size(peps, 2)]
        
        peps = QuantumNaturalfPEPS.basis_change(peps, Us, i)
        ci = maximum([minimum(abs.(Ok_Tensor)/ norm(Ok_Tensor)) for Ok_Tensor in Ok_Tensors])
        c = max(c, ci)
        
        peps_projected = QuantumNaturalfPEPS.get_projected(peps, S)
        if i == 1
            env_top[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected[1,:], peps.contract_dim; cutoff=peps.contract_cutoff)
            #env_down[end] = QuantumNaturalfPEPS.generate_env_row(peps_projected[size(peps, 1), :], peps.contract_dim; cutoff=peps.contract_cutoff)
        elseif i < size(peps, 1)
            env_top[i] = QuantumNaturalfPEPS.generate_env_row(peps_projected[i,:], peps.contract_dim; env_row_above=env_top[i-1], cutoff=peps.contract_cutoff)
        else
            env_down[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected[size(peps, 1), :], peps.contract_dim; cutoff=peps.contract_cutoff)
        end
        
        logψ = QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=min(i, size(peps, 1)-1))
        #@show logψ
        #logψ2, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
        #@show logψ-QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=min(i, size(peps, 1)-1))
    end
    
    return peps, c, logψ
end

function rotate_to_product_state_sweep_up(peps, logψ, env_top, env_down, h_envs_r, h_envs_l; S=zeros(Int, size(peps)...), verbose=false, kwargs...)
    c = 0
    for i in size(peps, 1):-1:1
        view_r = @view h_envs_r[i,:]
        view_l = @view h_envs_l[i, :]
        QuantumNaturalfPEPS.get_horizontal_envs!(peps, env_top, env_down, S, i, view_r, view_l)
        h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S)
        
        Ok_Tensors = [QuantumNaturalfPEPS.get_rotator(i, j, peps, env_top, env_down, h_envs_r, h_envs_l) for j in 1:size(peps, 2)]
        Us = [QuantumNaturalfPEPS.get_rotator(Ok_Tensors[j], S[i, j]; kwargs...) for j in 1:size(peps, 2)]
        
        peps = QuantumNaturalfPEPS.basis_change(peps, Us, i)
        ci = maximum([minimum(abs.(Ok_Tensor)/ norm(Ok_Tensor)) for Ok_Tensor in Ok_Tensors])
        c = max(c, ci)
        
        peps_projected = QuantumNaturalfPEPS.get_projected(peps, S)
        if i == size(peps, 1)
            env_down[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected[size(peps, 1), :], peps.contract_dim; cutoff=peps.contract_cutoff)
        elseif 1 < i
            iprime = size(peps, 1) + 1 - i 
            env_down[iprime] = QuantumNaturalfPEPS.generate_env_row(peps_projected[i,:], peps.contract_dim; env_row_above=env_down[iprime-1], cutoff=peps.contract_cutoff)
        else
            env_top[1] = QuantumNaturalfPEPS.generate_env_row(peps_projected[1,:], peps.contract_dim; cutoff=peps.contract_cutoff)
        end
        
        logψ = QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=max(1, i-1))
        
        #_, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
        #@show logψ-QuantumNaturalfPEPS.get_logψ(env_top, env_down; pos=max(1, i-1))
    end
    
    return peps, c, logψ
end

function rotate_to_product_state_sweep(peps; S=zeros(Int, size(peps)...), k=10, verbose=false, factor=1, error=1e-5, kwargs...)
    logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
    h_envs_r, h_envs_l = QuantumNaturalfPEPS.get_all_horizontal_envs(peps, env_top, env_down, S)
    local c
    for i in 1:k
        peps, c, logψ = rotate_to_product_state_sweep_down(peps, logψ, env_top, env_down, h_envs_r, h_envs_l; S, verbose=true, factor, kwargs...)
        peps, c, logψ = rotate_to_product_state_sweep_up(peps, logψ, env_top, env_down, h_envs_r, h_envs_l; S, verbose=true, factor, kwargs...)
        if verbose
            digits = Int(log10(1/error)) + 1
            @info "iter $i: $(round(c, digits=5)), $(round(logψ, digits=digits))"
            flush(stdout)
        end
        if c < error
            return peps, c, logψ
        end
    end
    @info "Only converged to $c in $k steps"
    
    #logψ, env_top, env_down = QuantumNaturalfPEPS.get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
    #@show logψ
    return peps, c, logψ
end