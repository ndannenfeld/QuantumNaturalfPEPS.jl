# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::ComplexF64 # TODO: Why is this complex?
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
function get_logψ_and_envs(peps::PEPS, S::Array{Int64,2}, Env_top=Array{Environment}(undef, size(S,1)-1))
    
    overwrite = true    # if Env_top is given and the bond dimension is sufficient, we do not need to calculate it again
    if isdefined(Env_top, 1) && maxlinkdim(Env_top[1].env) >= peps.contract_dim
        overwrite = false
    end
    
    Env_down = Array{Environment}(undef, size(S,1)-1)
    out = Array{ComplexF64}(undef, 2)
    
    if overwrite
        Env_top[1] = generate_env_row(peps[1], S[1,:], 1, peps.contract_dim)
    end
    Env_down[1] = generate_env_row(peps[size(S,1)], S[size(S,1),:], size(S,1), peps.contract_dim)
    
    # for every row we calculate the environments once from the top down and once from the bottom up
    for i in 2:size(S,1)-1
        i_prime = size(S,1)+1-i 
        
        if overwrite
            Env_top[i] = generate_env_row(peps[i], S[i,:], i, peps.contract_dim, env_row_above = Env_top[i-1])
        end
        Env_down[i] = generate_env_row(peps[i_prime], S[i_prime,:], i_prime, peps.contract_dim, env_row_above = Env_down[i-1])
    end
    
    # once we calculated all environments we calculate <ψ|S> using the environments
    out[1] = contract(Env_top[end].env.*MPS([peps[size(S,1),j]*ITensor([(S[end,j]+1)%2, S[end,j]], inds(peps[size(S,1),j],"phys_$(j)_$(size(S,1))")) for j in 1:size(S,2)]))[1] * exp(Env_top[end].f)
    out[2] = contract((MPS([peps[1,j]*ITensor([(S[1,j]+1)%2, S[1,j]], inds(peps[1,j],"phys_$(j)_$(1)")) for j in 1:size(S,2)])).*Env_down[end].env)[1] * exp(Env_down[end].f)
    
    return mean(log.(Complex.((out)))), Env_top, Env_down
end

# calculates the environments for a given row and contracts that with env_row_above
function generate_env_row(peps_row, S_row, i, contract_dim; env_row_above=nothing)
    env = [peps_row[j]*ITensor([(S_row[j]+1)%2, S_row[j]], inds(peps_row[j],"phys_$(j)_$(i)")) for j in 1:length(S_row)]
    norm_shift = 0
    if env_row_above === nothing
        env = MPS(env)
    else
        env = apply(MPO(env), env_row_above.env, maxdim=contract_dim)
        norm_shift = env_row_above.f
    end

    return Environment(env, norm_shift; normalize=true)
end