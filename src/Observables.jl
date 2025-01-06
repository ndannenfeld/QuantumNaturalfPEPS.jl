function get_ExpectationValue(peps::PEPS, O; it=100, threaded=false)
    hilbert = siteinds(peps)
    if !(O isa Vector)
        O = [O]
    end
    O_op = Array{TensorOperatorSum}(undef, length(O))

    for i in 1:length(O)
        O_op[i] = TensorOperatorSum(O[i], hilbert)
    end
    if threaded
        return get_ExpectationValues_multithread(peps, O_op; it)
    else
        return get_ExpectationValues_singlethread(peps, O_op; it)
    end
end

function get_ExpectationValues_multithread(peps, O_op; it=100)
    nr_threads = Threads.nthreads()
    k = ceil(Int, it / nr_threads)
    
    Obs = complex.(zeros(k*nr_threads, length(O_op)))
    logψs = Array{Complex}(undef, k*nr_threads)
    logpcs = Array{Complex}(undef, k*nr_threads)
    
    Threads.@threads for i in 1:nr_threads
            slice = (1+(i-1)*k):(i*k)
            Obser = @view Obs[slice, :]
            logψ = @view logψs[slice]
            logpc = @view logpcs[slice]
            get_ExpectationValues!(peps, O_op, Obser, logψ, logpc; it=k)
    end

    return Obs, compute_importance_weights(logψs, logpcs)
end

function get_ExpectationValues_singlethread(peps, O_op; it=100)
    O_loc = Array{Complex}(undef, it, length(O_op))
    logψ = Array{Complex}(undef, it)
    logpc = Array{Complex}(undef, it)
    
    get_ExpectationValues!(peps, O_op, O_loc, logψ, logpc; it)
    return O_loc, compute_importance_weights(logψ, logpc)
end

function get_ExpectationValues!(peps, O_op, Observable, logψ, logpc; it=100)

    for i in 1:it
        S, logpc[i], env_top = get_sample(peps)

        logψ[i], env_top, env_down, max_bond = get_logψ_and_envs(peps, S, env_top) 
        h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S)
            
        logψ_flipped = Dict{Any, Number}()
        for j in 1:length(O_op)
            O_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(O_op[j], S; get_flip_sites=true)
            Observable[i,j] = get_Ek(peps, O_op[j], env_top, env_down, S, logψ[i], h_envs_r, h_envs_l; logψ_flipped, Ek_terms=O_terms)
        end
    end

    return Observable, logψ, logpc
end
