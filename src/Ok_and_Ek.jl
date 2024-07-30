
# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek(peps, ham_op; timer=TimerOutput(), Ok=nothing, kwargs...)
     
    S, logpc, env_top = @timeit timer "sampling" get_sample(peps) # draw a sample
    logψ, env_top, env_down = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top) # compute the environments of the peps according to that sample
    h_envs_r, h_envs_l = @timeit timer "horizontal_envs" get_all_horizontal_envs(peps, env_top, env_down, S) # computes the horizontal environments of the already sampled peps
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logψ, h_envs_r, h_envs_l) # compute the local energy
    grad = @timeit timer "log_gradients" get_Ok(peps, env_top, env_down, S, h_envs_r, h_envs_l, logψ; Ok) # compute the gradient

    return grad, E_loc, logψ, S, logpc
end

