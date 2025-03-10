# calculates the gradient: d(<ψ|S>)/d(Θ_ij) / <ψ|S>
function get_Ok(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment},
                logψ::Number, h_envs_r, h_envs_l, i::Int64, j::Int64)
    Ok_Tensor = 1
    f = 0
    if j != size(peps, 2)
        Ok_Tensor *= h_envs_r[i,j]
    end
    if i != 1
        Ok_Tensor *= env_top[i-1].env[j]
        f += env_top[i-1].f
    end
    if i != size(peps, 1)
        Ok_Tensor *= env_down[end-i+1].env[j]
        f += env_down[end-i+1].f
    end
    if j != 1
        Ok_Tensor *= h_envs_l[i,j-1]
    end
    g = exp(f - logψ)
    # if the tensor is real, we only want the real part of the gradient
    if isreal(Ok_Tensor) 
        g = real(g)
    end
    Ok_Tensor *= g
    @assert eltype(Ok_Tensor) === eltype(peps) "The gradient is $(eltype(Ok_Tensor)) but the PEPS is $(eltype(peps))"
    return Ok_Tensor
end


# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::AbstractPEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number;
                h_envs_r=nothing, h_envs_l=nothing, Ok=nothing, mask=peps.mask)
    if Ok === nothing
        Ok = Vector{eltype(peps)}(undef, length(peps))
    end
    if h_envs_r === nothing || h_envs_l === nothing
        h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, S) # computes the horizontal environments of the already sampled peps
    end

    pos = 1
    shift = 0
    
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        if mask[i,j] != 0

            # Get the projected Ok tensor
            Ok_Tensor = get_Ok(peps, env_top, env_down, logψ, h_envs_r, h_envs_l, i, j)

            # Reshape the tensor to a vector to obtain the gradient
            shift = prod(dim.(inds(Ok_Tensor)))
            loc_dim = dim(siteind(peps, i,j))
            
            # loop through every possible sample
            for spin in 0:loc_dim-1
                # if we get to the actually sampled value write the tensor in the gradient, else fill with zeros
                if S[i,j] == spin
                    # Write in Gradient
                    x = @view Ok[pos+spin:loc_dim:pos+loc_dim*shift-1]
                    permute_reshape_and_copy!(x, Ok_Tensor, linkinds(peps, i, j))
                else
                    # Fill with zeros instead
                    Ok[pos+spin:loc_dim:pos+loc_dim*shift-1] .= 0
                end
            end
            pos = pos+loc_dim*shift
        end
    end
    return Ok
end

function numerical_Ok(peps::AbstractPEPS, S::Matrix{Int64}, direction, logψ; dt=0.01)
    p2 = deepcopy(peps)
    θ = vec(peps)
    f = 0
    for i in [-1,1]
        write!(p2, θ + i*dt*direction)
        x = get_projected(p2, S)
        f += i*contract_peps_exact(x)
    end
    return f/(2*dt) *convert_if_real(exp(-logψ))
end