function Test_Ok(peps::PEPS, S::Matrix{Int64}, direction; env_top=nothing, env_down=nothing, h_envs=nothing, logψ=nothing, dt=0.01)
    if logψ === nothing || env_top === nothing || env_down === nothing
        logψ, env_top, env_down = get_logψ_and_envs(peps, S)
    end
    if h_envs === nothing
        h_envs = get_all_horizontal_envs(peps, env_top, env_down, S) 
    end

    Ok = get_Ok(peps, env_top, env_down, S, h_envs, logψ)'*direction
    Oknum = numerical_Ok(peps, S, direction, logψ; dt=dt)
    return Ok, Oknum
end

function Test_logψ(peps::PEPS, S)
    logψ, et, ed = get_logψ_and_envs(peps, S)
    logψex = logψ_exact(peps, S)
    return logψ, logψex
end

function Test_Ek(peps::PEPS, ham_op; it=1)
    E = Array{Float64}(undef, it)
    Enum = Array{Float64}(undef, it)

    func = get_logψ_func(peps)
    for i in 1:it
        S = rand([0,1], size(peps)) 
        logψ, env_top, env_down = get_logψ_and_envs(peps, S)
        h_envs = get_all_horizontal_envs(peps, env_top, env_down, S) 
        fourb_envs = get_all_4b_envs(peps, env_top, env_down, S)

        E[i] = get_Ek(peps, ham_op, env_top, env_down, S, logψ, h_envs, fourb_envs)
        Enum[i] = QuantumNaturalGradient.get_Ek(S.+1, ham_op, func)
    end

    return E, Enum
end

function get_logψ_func(peps)
    function logψ_func(sample)
        return QuantumNaturalfPEPS.logψ_exact(peps, sample.-1)
    end
    return logψ_func
end