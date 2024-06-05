

"""
dest = zeros(3, 3, 4)
dest2 = @view dest[:, :, 1:2]
permute_and_copy!(dest2, t, (ind[1], ind[3], ind[2]))
dest
"""
permute_and_copy!(dest, tensor::ITensor, target_indices) = permute_and_copy!(dest, tensor.tensor, target_indices)

function permute_and_copy!(dest, tensor::NDTensors.DenseTensor, target_indices)
    perm = NDTensors.getperm(target_indices, inds(tensor))
    s = reshape(tensor.storage, size(tensor))
    return permutedims!(dest, s, perm)
end

