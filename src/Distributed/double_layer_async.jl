function dumb_hash(peps::PEPS)
    return mean(norm.(peps.tensors))
end

function generate_async_double_layer_envs(peps_::PEPS; verbose=false)
    peps = peps_
    slock = Threads.ReentrantLock()
    
    hash_peps = dumb_hash(peps)
    double_layer_envs = peps.double_layer_envs
    run_flag = Threads.Atomic{Bool}(true)

    function double_layer_update(peps_::PEPS)
        lock(slock)
        peps = peps_
        peps.double_layer_envs = copy(double_layer_envs)
        unlock(slock)
        return peps.double_layer_envs
    end

    function generate_double_layer_envs()
        Lx = size(peps, 1)
        sites = siteinds(peps)
        
        while run_flag[]  # loop until the stop flag is set
            maxdim = peps.double_contract_dim
            cutoff = peps.double_contract_cutoff
            curr_hash = dumb_hash(peps)
            if curr_hash != hash_peps
                if verbose
                    @info "Updating double layer environments"
                end
                hash_peps = curr_hash
                # for every row we calculate the double layer environment
                d = generate_double_layer_env_row(peps[Lx, :], sites[Lx, :], maxdim; cutoff)
                lock(slock)
                double_layer_envs[end] = d
                unlock(slock)

                for i in Lx-1:-1:2
                    d = generate_double_layer_env_row(peps[i, :], sites[i, :], double_layer_envs[i], maxdim; cutoff)
                    lock(slock)
                    double_layer_envs[i-1] = d
                    unlock(slock)
                end
            else
                sleep(0.1)
            end
        end
    end

    @assert Threads.nthreads() > 1 "This function is only useful with more than one thread"
    # Kick off the environment‐updating loop in a separate Task
    t = Threads.@spawn generate_double_layer_envs()
    return double_layer_update, () -> (run_flag[] = false)
end