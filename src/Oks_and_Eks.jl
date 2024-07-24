
function compute_importance_weights(logψs, logpcs)
    log_ratios =  2 .* real.(logψs) .- logpcs
    logZ = logsumexp(log_ratios) - log(length(logpcs))
    return exp.(log_ratios .- logZ)
end

function generate_Oks_and_Eks(peps::PEPS, ham::OpSum; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return generate_Oks_and_Eks(peps, ham_op; kwargs...)
end

function generate_Oks_and_Eks(peps::PEPS, ham_op::TensorOperatorSum; threaded=false, multiproc=false, kwargs...)
    if multiproc
        return generate_Oks_and_Eks_multiproc(peps, ham_op; threaded, kwargs...)
    elseif threaded
        return generate_Oks_and_Eks_threaded(peps, ham_op; kwargs...)
    else
        return generate_Oks_and_Eks_singlethread(peps, ham_op; kwargs...)
    end
end


###### Single threaded
# this function returns a Ok_and_Eks function wich can be used to optimise via QNG.evolve
function generate_Oks_and_Eks_singlethread(peps::PEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), kwargs...)
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        write!(peps, Θ)
        @timeit timer "double_layer_envs" update_double_layer_envs!(peps) # update the double layer environments once for the peps 
        return Oks_and_Eks_singlethread(peps, ham_op, sample_nr; timer=timer, kwargs...)
    end
    return Oks_and_Eks_
end

# The central function is Oks and Eks
function Oks_and_Eks_singlethread(peps::PEPS, ham_op::TensorOperatorSum, sample_nr::Integer; timer=TimerOutput(), kwargs...)
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    
    Ok = Matrix{eltype_}(undef, sample_nr, length(peps))
    E_loc = Vector{eltype_}(undef, sample_nr)
    logψ = Vector{Complex{eltype_real}}(undef, sample_nr)
    samples = Vector{Matrix{Int}}(undef, sample_nr)
    logpc = Vector{eltype_real}(undef, sample_nr)

    for i in 1:sample_nr
        Ok_view = @view Ok[i, :]
        _, E_loc[i], logψ[i], samples[i], logpc[i] = Ok_and_Ek(peps, ham_op; timer, Ok=Ok_view, kwargs...)
    end
    
    return Ok, E_loc, logψ, samples, compute_importance_weights(logψ, logpc)
    # returns Gradient, local Energy, log(<ψ|S>), samples S, p
end

###### Multiple threads
function generate_Oks_and_Eks_threaded(peps::PEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), kwargs...)
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; reset_double_layer=true, kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        write!(peps, Θ; reset_double_layer)
        if reset_double_layer
            @timeit timer "double_layer_envs" update_double_layer_envs!(peps) # update the double layer environments once for the peps
        end
        return Oks_and_Eks_threaded(peps, ham_op, sample_nr; timer, kwargs...)
    end
    return Oks_and_Eks_
end

function Oks_and_Eks_threaded(peps, ham_op, sample_nr; Oks=nothing, importance_weights=true,
                                               timer=TimerOutput(), nr_threads=Threads.nthreads(), kwargs...)
    
    nr_parameters = length(peps)
    k = ceil(Int, sample_nr / nr_threads)
    
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)

    if Oks === nothing
        Oks = Array{eltype_, 3}(undef, nr_parameters, k, nr_threads)
    end
    samples = Matrix{Any}(undef, k, nr_threads)
    Eks = Matrix{eltype_}(undef, k, nr_threads)
    logψs = Matrix{Complex{eltype_real}}(undef, k, nr_threads)
    logpcs = Matrix{eltype_real}(undef,  k, nr_threads)
    
    Threads.@threads for i in 1:nr_threads
        for j in 1:k
            Ok = @view Oks[:, j, i]
            _, Eks[j, i], logψs[j, i], samples[j, i], logpcs[j, i] = Ok_and_Ek(peps, ham_op; Ok, kwargs...)
        end

    end
    Eks = reshape(Eks, :)
    Oks = reshape(Oks, size(Oks, 1), :)
    logψs = reshape(logψs, :)
    samples = reshape(samples, :)
    logpcs = reshape(logpcs, :)
    
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end

    return Eks, Oks, logψs, samples, weights
end

#### Multiprocessing

function generate_Oks_and_Eks_multiproc(peps::PEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), threaded=true, kwargs...)
    function Oks_and_Eks_(Θ::Vector{T}, sample_nr::Integer; kwargs2...) where T
        if length(kwargs2) > 0
            kwargs = merge(kwargs, kwargs2)
        end
        write!(peps, Θ)
        @timeit timer "double_layer_envs" update_double_layer_envs!(peps) # update the double layer environments once for the peps
        return Oks_and_Eks_multiproc(peps, ham_op, sample_nr; kwargs...)
    end
    return Oks_and_Eks_
end

function Oks_and_Eks_multiproc(peps, ham_op, sample_nr; Oks=nothing, importance_weights=true, 
                               n_threads=Distributed.remotecall_fetch(()->Threads.nthreads(), workers()[1]))

    nr_procs = length(workers())
    k = ceil(Int, sample_nr / nr_procs)
    k_thread = ceil(Int, k / n_threads)
    k_eff = k_thread * n_threads
    sample_nr_eff = k_eff * nr_procs
    nr_parameters = length(peps)
    # TODO: Send ham_op only once through the network
    out = [Distributed.remotecall(() -> Oks_and_Eks_threaded(peps, ham_op, k; importance_weights=false), w) for w in workers()]
    
    eltype_ = eltype(peps)
    eltype_real = real(eltype_)
    
    samples = Vector{Any}(undef, sample_nr_eff)
    Eks = Vector{eltype_}(undef, sample_nr_eff)
    logψs = Vector{Complex{eltype_real}}(undef, sample_nr_eff)
    logpcs = Vector{eltype_real}(undef, sample_nr_eff)
    
    if Oks === nothing
        Oks = Matrix{eltype_}(undef, nr_parameters, sample_nr_eff)
    end

    Threads.@threads for (i, out_i) in collect(enumerate(out))
        
        i1 = k_eff * (i - 1) + 1
        i2 = k_eff * i
        
        Eks[i1:i2], Oks[:, i1:i2], logψs[i1:i2], samples[i1:i2], logpcs[i1:i2] = fetch(out_i)
    end
    
    if importance_weights
        weights = compute_importance_weights(logψs, logpcs)
    else
        weights = logpcs
    end
    
    return Eks, Oks, logψs, samples, weights
end