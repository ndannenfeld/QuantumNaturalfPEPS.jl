# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::Real
    function Environment(env, f; normalize=true)
        env = new(env, f)
        if normalize
            normalize!(env)
        end
        return env
    end
    Environment(env; kwargs...) = Environment(env, 0.0; kwargs...)
end
Base.getindex(env::Environment, i::Int) = env.env[i]
Base.reverse(env::Environment) = ReverseEnvironment(env)

struct ReverseEnvironment
    env::Environment
end
Base.getindex(env::ReverseEnvironment, i::Int) = reverse(env).env[end-i+1]
Base.getproperty(x::ReverseEnvironment, y::Symbol) = getproperty(reverse(x), y)
Base.reverse(env::ReverseEnvironment) = getfield(env, :env)

function normalize!(env::Environment)
    for env_i in env.env
        norm = max_norm(env_i)
        env_i ./= norm
        env.f += log(norm)
    end
    return env
end

# Computes the environments and log(<ψ|S>)
function get_logψ_and_envs(peps::PEPS, S::Array{Int64,2}, env_top=Array{Environment}(undef, size(S,1)-1);
                           alg="densitymatrix", cutoff=1e-13, kwargs...)
    
    overwrite = true    # if env_top is given and the bond dimension is sufficient, we do not need to calculate it again
    if isdefined(env_top, 1) && maxlinkdim(env_top[1].env) >= peps.contract_dim
        overwrite = false
    end
    
    env_down = Array{Environment}(undef, size(S,1)-1)

    peps_projected = get_projected(peps, S)
    
    if overwrite
        env_top[1] = generate_env_row(peps_projected[1,:], peps.contract_dim; alg, cutoff)
    end
    env_down[1] = generate_env_row(peps_projected[size(S,1), :], peps.contract_dim; alg, cutoff)
    
    # for every row we calculate the environments once from the top down and once from the bottom up
    for i in 2:size(S,1)-1
        i_prime = size(S,1)+1-i 
        if overwrite
            env_top[i] = generate_env_row(peps_projected[i,:], peps.contract_dim, env_row_above=env_top[i-1]; alg, cutoff)
        end
        env_down[i] = generate_env_row(peps_projected[i_prime,:], peps.contract_dim, env_row_above=env_down[i-1]; alg, cutoff)
    end
    
    # once we calculated all environments we calculate <ψ|S> using the environments
    return get_logψ(env_top, env_down; kwargs...), env_top, env_down
end

# calculates the environments for a given row and contracts that with env_row_above
function generate_env_row(peps_projected, contract_dim; env_row_above=nothing, alg="densitymatrix", cutoff=1e-13)
    norm_shift = 0
    if env_row_above === nothing
        peps_projected = MPS(peps_projected)
    else
        peps_projected = contract(MPO(peps_projected), env_row_above.env; maxdim=contract_dim, alg, cutoff)
        norm_shift = env_row_above.f
    end

    return Environment(peps_projected, norm_shift; normalize=true)
end

function get_logψ(env_top::Vector{Environment}, env_down::Vector{Environment}; pos=Int(ceil(length(env_top)/2)))
    ψS = contract(env_top[pos].env.*env_down[end-pos+1].env)[1]
    logψS = log(Complex(ψS))
    # TODO: Don't you want to use this instead here?
    #logψS = _log_or_not_dot(env_top[pos].env, env_down[end-pos+1].env, true; dag=false)
    return logψS + env_top[pos].f + env_down[end-pos+1].f
end

function logψ_exact(peps, sample)
    proj = get_projected(peps, sample)
    con = contract_peps_exact(proj)
    return log(Complex(con))
end

function get_all_horizontal_envs(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, all_horizontal_envs::Array{ITensor}=Array{ITensor}(undef, size(peps,1), 2, size(peps, 2)-1))
    for i in 1:size(peps,1)
        get_horizontal_envs!(peps, env_top, env_down, S, i, @view all_horizontal_envs[i,:,:])
    end
    return all_horizontal_envs
end

function get_horizontal_envs(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs=Matrix{ITensor}(undef, 2,size(peps, 2)-1))
    get_horizontal_envs!(peps, env_top, env_down,S,i,horizontal_envs)
    return horizontal_envs
end

# this function computes the horizontal environments for a given row
function get_horizontal_envs!(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs)
    peps_i = get_projected(peps, S, i, :)    #contract the row with S
    
    # now we loop through every site and compute the environments (once from the right and once from the left) by MPO-MPS contraction.
    if i == 1
        contract_recursiv!(horizontal_envs, peps_i, env_down[end].env)
    elseif i == size(peps, 1)
        contract_recursiv!(horizontal_envs, peps_i, env_top[end].env)
    else
        contract_recursiv!(horizontal_envs, env_top[i-1].env, peps_i; c=env_down[end-i+1].env)
    end
end

function contract_recursiv!(h_envs, a, b; c=ones(length(a)), d=ones(length(a)))
    h_envs[1,end] = a[end]*b[end]*c[end]*d[end]
    h_envs[2,1] = a[1]*b[1]*c[1]*d[1]
    for j in length(a)-1:-1:2
        h_envs[1,j-1] = h_envs[1,j]*a[j]*b[j]*c[j]*d[j]
    end
    for j in 2:length(a)-1
        h_envs[2,j] = h_envs[2,j-1]*a[j]*b[j]*c[j]*d[j]
    end
end

function get_all_4b_envs(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, all_4b_envs::Array{ITensor}=Array{ITensor}(undef, size(peps,1)-1, 2, size(peps, 2)-1))
    for i in 1:size(peps,1)-1
        get_4b_envs!(peps, env_top, env_down, S, i, @view all_4b_envs[i,:,:])
    end
    return all_4b_envs
end

function get_4b_envs(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, fourb_envs=Matrix{ITensor}(undef, 2,size(peps, 2)-1))
    get_4b_envs!(peps, env_top, env_down,S,i,fourb_envs)
    return fourb_envs
end

function get_4b_envs!(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, fourb_envs)
    peps_i = get_projected(peps, S, i, :)  
    peps_j = get_projected(peps, S, i+1, :)

    if i == 1
        contract_recursiv!(fourb_envs, peps_i, peps_j, c=env_down[end-1].env)
    elseif i == size(peps, 1)-1
        contract_recursiv!(fourb_envs, env_top[end-1].env, peps_i, c=peps_j)
    else
        contract_recursiv!(fourb_envs, env_top[i-1].env, peps_i, c=peps_j, d=env_down[end-i].env)
    end
end