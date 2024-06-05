
# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek(peps, ham_op; timer=TimerOutput(), Ok=nothing, kwargs...)
     
    S, pc, psi_S, et = @timeit timer "sampling" get_sample(peps) # draw a sample
    logψ, et, ed = @timeit timer "logψ" get_logψ_and_envs(peps, S) # compute the environments of the peps according to that sample
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, et, ed, S, logψ) # compute the local energy
    grad = @timeit timer "log_gradients" get_Ok(peps, et, ed, S, logψ; Ok) # compute the gradient

    return grad, E_loc, logψ, S, pc, psi_S
end

