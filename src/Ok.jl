# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, h_envs::Array{ITensor}, logψ::Number; Ok=nothing)
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
                Ok_Tensor *= h_envs[i,1,j]
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
                Ok_Tensor *= h_envs[i,2,j-1]
            end
            g = exp(f - logψ)
            g = convert_if_real(g)
            Ok_Tensor *= g
            @assert eltype(Ok_Tensor) === eltype(peps) "The gradient is $(eltype(Ok_Tensor)) but the PEPS is $(eltype(peps))"

            # lastly we reshape the tensor to a vector to obtain the gradient
            # TODO: Does this still work for phys_dim!=2? Possible bug waiting to happen, use the permute_and_copy! function I added to tensor_ops.jl
            shift = prod(dim.(inds(Ok_Tensor)))
            if S[i,j] == 1
                # Fill with zeros instead
                Ok[pos:pos+shift-1] .= 0
                pos = pos+shift
                # TODO Reduce copys by not using Array, make sure that the indices are in the same order as the corresponding peps tensor
                
                x = @view Ok[pos:pos+shift-1]
                permute_reshape_and_copy!(x, Ok_Tensor, linkinds(peps,i,j))
            else
                x = @view Ok[pos:pos+shift-1]
                permute_reshape_and_copy!(x, Ok_Tensor, linkinds(peps,i,j))
                pos = pos+shift
                Ok[pos:pos+shift-1] .= 0
            end
            pos = pos+shift
        end
    end
    return Ok
end