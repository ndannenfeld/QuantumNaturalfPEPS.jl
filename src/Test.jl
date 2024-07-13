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
