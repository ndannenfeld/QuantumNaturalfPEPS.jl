
# Calculates the Energy and Gradient of a given peps and hamiltonian
function Ok_and_Ek(peps, ham_op; timer=TimerOutput(), Ok=nothing, kwargs...)
     
    S, pc, env_top = @timeit timer "sampling" get_sample(peps) # draw a sample
    logψ, env_top, env_down = @timeit timer "vertical_envs" get_logψ_and_envs(peps, S, env_top) # compute the environments of the peps according to that sample
    # TODO: The code would be easier to read if we would have left and right horizontal enviroments instead of everything crammed into an array
    h_envs = @timeit timer "horizontal_envs" get_all_horizontal_envs(peps, env_top, env_down, S) 
    fourb_envs = @timeit timer "4b_envs" get_all_4b_envs(peps, env_top, env_down, S) 
    E_loc = @timeit timer "energy" get_Ek(peps, ham_op, env_top, env_down, S, logψ, h_envs, fourb_envs) # compute the local energy
    grad = @timeit timer "log_gradients" get_Ok(peps, env_top, env_down, S, h_envs, logψ; Ok) # compute the gradient

    return grad, E_loc, logψ, S, pc
end

