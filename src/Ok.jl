# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number; Ok::Vector=complex(zeros(length(peps))))
    pos = 1
    shift = 0
    for i in 1:size(peps, 1)
        for j in 1:size(peps, 2)
            # peps_S gives an array of ITensors wich are peps[i,y]*S[i,y] for all tensors in the row except for i,j
            peps_S = [(j_p != j ? peps[i,j_p]*ITensor([(S[i,j_p]+1)%2, S[i,j_p]], inds(peps[i,j_p], "phys_$(j_p)_$(i)")) : 1) for j_p in 1:size(S,2)]
            if i == 1
                # we get the differential tensor if we contract the environments with the peps_S
                Ok_Tensor = exp(env_down[end].f - logψ)*contract(env_down[end].env .* peps_S)
            elseif i == size(peps, 1)
                Ok_Tensor = exp(env_top[end].f - logψ)*contract(env_top[end].env .* peps_S)
            else
                Ok_Tensor = exp(env_top[i-1].f + env_down[end-i+1].f - logψ)*contract(env_top[i-1].env .* peps_S .* env_down[end-i+1].env)
            end

            # lastly we reshape the tensor to a vector to obtain the gradient
            # TODO: Does this still work for phys_dim!=2? Possible bug waiting to happen, let's discuss!
            shift = prod(dim.(inds(Ok_Tensor)))
            if S[i,j] == 1
                # Fill with zeros instead
                Ok[pos:pos+shift-1] .= 0
                pos = pos+shift
                # TODO Reduce copys by not using Array, make sure that the indices are in the same order as the corresponding peps tensor
                Ok[pos:pos+shift-1] = reshape(Array(Ok_Tensor, inds(Ok_Tensor)), :)
            else
                Ok[pos:pos+shift-1] = reshape(Array(Ok_Tensor, inds(Ok_Tensor)), :)
                pos = pos+shift
                Ok[pos:pos+shift-1] .= 0
            end
            pos = pos+shift
        end
    end
    return Ok
end