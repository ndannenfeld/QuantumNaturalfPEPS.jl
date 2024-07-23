

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

max_norm(x::ITensor) = maximum(abs, x.tensor)

permute_reshape_and_copy!(dest, tensor::ITensor, target_indices) = permute_reshape_and_copy!(dest, tensor.tensor, target_indices)

function permute_reshape_and_copy!(dest, tensor::NDTensors.DenseTensor, target_indices)
    d = complex(zeros(dims(target_indices)...))
    perm = NDTensors.getperm(target_indices, inds(tensor))
    s = reshape(tensor.storage, size(tensor))
    permutedims!(d, s, perm)
    dest .= reshape(d, :)
end

function random_unitary(::Type{ElT}, i1::Vector{Index{Int64}}, i2::Vector{Index{Int64}}) where ElT<:Number
    t = NDTensors.random_unitary(ElT, dim(i1), dim(i2))
    return ITensor(t, i1..., i2...)
end

random_unitary(i1::Vector{Index{Int64}}, i2::Vector{Index{Int64}}) = random_unitary(Float64, i1, i2)

"""
    subsinds(T, indx_orig::Index, indx_tar::Index)
Substitute the index `indx_orig` by `indx_tar` in the tensor `T`.
"""
function subsinds(T, indx_orig::Index, indx_tar::Index)
  inds_ = collect(inds(T))
  i = findfirst(x -> x == indx_orig, inds_)
  if i === nothing
    @warn "Index $indx_orig not found in tensor $T"
    return T
  end
  inds_[i] = indx_tar
  return ITensors.setinds(T, inds_)
end

function ITensors.combiner(is::ITensors.Indices; target_ind=nothing, kwargs...)
    tags = get(kwargs, :tags, "CMB,Link")
      if target_ind === nothing
        target_ind = Index(prod(dims(is)), tags)
      end
    new_is = (target_ind, is...)
    return itensor(ITensors.Combiner(), new_is)
end