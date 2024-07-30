# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, h_envs_r::Array{ITensor}, h_envs_l::Array{ITensor}, logψ::Number; Ok=nothing)
    if Ok === nothing
        Ok = Vector{eltype(peps)}(undef, length(peps))
    end
    
    pos = 1
    shift = 0
    
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            Ok_Tensor = 1
            f = 0
            if j != size(peps,2)
                Ok_Tensor *= h_envs_r[i,j]
            end
            if i != 1
                Ok_Tensor *= env_top[i-1].env[j]
                f += env_top[i-1].f
            end
            if i != size(peps,1)
                Ok_Tensor *= env_down[end-i+1].env[j]
                f += env_down[end-i+1].f
            end
            if j != 1
                Ok_Tensor *= h_envs_l[i,j-1]
            end
            g = exp(f - logψ)
            g = convert_if_real(g)
            Ok_Tensor *= g
            @assert eltype(Ok_Tensor) === eltype(peps) "The gradient is $(eltype(Ok_Tensor)) but the PEPS is $(eltype(peps))"

            # lastly we reshape the tensor to a vector to obtain the gradient
            shift = prod(dim.(inds(Ok_Tensor)))
            loc_dim = dim(siteind(peps, i,j))
            
            # loop through every possible sample
            for spin in 0:loc_dim-1
                # if we get to the actually sampled value write the tensor in the gradient, else fill with zeros
                if S[i,j] == spin
                    # Write in Gradient
                    x = @view Ok[pos+spin:loc_dim:pos+loc_dim*shift-1]
                    permute_reshape_and_copy!(x, Ok_Tensor, linkinds(peps,i,j))
                else
                    # Fill with zeros instead
                    Ok[pos+spin:loc_dim:pos+loc_dim*shift-1] .= 0
                end
            end
            pos = pos+2*shift
        end
    end
    return Ok
end

function numerical_Ok(peps::PEPS, S::Matrix{Int64}, direction, logψ; dt=0.01)
    p2 = deepcopy(peps)
    θ = flatten(peps)
    f = 0
    for i in [-1,1]
        write!(p2, θ + i*dt*direction)
        x = get_projected(p2, S)
        f += i*contract_peps_exact(x)
    end
    return f/(2*dt) *convert_if_real(exp(-logψ))
end