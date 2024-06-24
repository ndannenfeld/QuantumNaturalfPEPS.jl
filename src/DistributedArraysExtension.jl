"""
Fetches a distributed slice of a distributed array
"""
function fetch_distributedslice(d::DArray, i1, i2)
    rs = [remotecall(d->d.localpart[i1:i2, :], w, d) for w in d.pids]
    
    M = Matrix{eltype(d)}(undef, i2-i1+1, size(d, 2))
    for (r, idx, pid) in zip(rs, d.indices, d.pids)
        dest = @view M[:, idx[2]]
        Distributed.Serialization.register_deserialization_array!(pid, dest)
        fetch(r)
    end
    return M
end

"""
Fetches a distributed slice of a distributed array and multiplies it with its transpose
"""
function fetch_and_multiply(d, i1, i2)
    M = fetch_distributedslice(d, i1, i2)
    return M' * M
end

"""
Calculates d' * d where d is a distributed array
"""
function self_mul_transpose(d::DArray)
    rs = Future[]
    i1 = 1
    nr_k = size(d, 1)
    nr_k_i = nr_k ÷ length(d.pids)
    for w in d.pids
        i2 = i1 + nr_k_i -1
        push!(rs, Distributed.remotecall(fetch_and_multiply, w, d, i1, i2))
        i1 = i2 + 1
    end

    sum_ = zeros(eltype(d), size(d, 2), size(d, 2))
    for r in rs
        sum_ .+= fetch(r)
    end
    return sum_
end

getFuture(x) = remotecall(()-> x, myid())