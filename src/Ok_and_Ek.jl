# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek(peps, ham_op; timer=TimerOutput(), kwargs...)
     
    S, pc, psi_S, et = @timeit timer "sampling" get_sample(peps) # draw a sample
    logψ, et, ed = @timeit timer "logψ" get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, et, ed, S, logψ) # compute the local energy
    grad = @timeit timer "log_gradients" get_Ok(peps, et, ed, S, logψ) # compute the gradient

    return grad, E_loc, logψ, S, pc, psi_S
end

function generate_Oks_and_Eks(peps::PEPS, ham::OpSum; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return generate_Oks_and_Eks(peps, ham_op; kwargs...)
end

# this function returns a Ok_and_Eks function wich can be used to optimise via QNG.evolve
function generate_Oks_and_Eks(peps::PEPS, ham_op::TensorOperatorSum; timer=TimerOutput(), kwargs...)

    # The central function is Oks and Eks
    function Oks_and_Eks(Θ::Vector, sample_nr::Integer)
        grad = Matrix{ComplexF64}(undef, sample_nr, length(Θ))
        E_loc = Vector{ComplexF64}(undef, sample_nr)
        logψ = Vector{ComplexF64}(undef, sample_nr)
        S = Vector{Matrix{Int}}(undef, sample_nr)
        pc = zeros(sample_nr)
        psi_S = zeros(sample_nr)
        
        write!(peps, Θ)
        @timeit timer "double_layer_envs" update_double_layer_envs!(peps) # update the double layer environments once for the peps 
        for i in 1:sample_nr
            grad[i,:], E_loc[i], logψ[i], S[i], pc[i], psi_S[i] = Ok_and_Ek(peps, ham_op; timer, kwargs...)
        end
        
        # TODO: what is psi_S?
        Z = 1/sample_nr * sum(exp.(psi_S - pc))
        p = (exp.(psi_S) ./Z) ./ exp.(pc) # determine the estimate for pc given the samples drawn
        
        return grad, E_loc, logψ, S, p
        # returns Gradient, local Energy, log(<ψ|S>), samples S, p
    end
    return Oks_and_Eks
end
