import Serialization
import Serialization: serialize, deserialize, AbstractSerializer
@eval Serialization memory = Dict()
"""
If you register an array with `register_memory!(worker_id, array)`, then the next time an array gets deserialized with the same type and dimensions, the array that was registered will be overwritten with the deserialized array. This is useful to not have to allocate memory for the same array multiple times.
"""
@eval Serialization function register_memory!(worker_id::Int, x)
    etly = eltype(Oi)
    dims = size(Oi)
    Serialization.memory[(worker_id, elty, dims)] = x
end

@eval Serialization function retrieve_if_in_memory(s::AbstractSerializer, elty, dims)
    if hasfield(s, :pid)
        id = (s.pid, elty, dims)
        if id in keys(Serialization.memory)
            A = Serialization.memory[id]
            delete!(Serialization.memory, id)
            return A
        end
    end
    return Array{elty}(undef, dims)
end


function Serialization.deserialize_array(s::AbstractSerializer)
    slot = s.counter; s.counter += 1
    d1 = deserialize(s)
    if isa(d1, Type)
        elty = d1
        d1 = deserialize(s)
    else
        elty = UInt8
    end
    if isa(d1, Int32) || isa(d1, Int64)
        if elty !== Bool && isbitstype(elty)
            a = Vector{elty}(undef, d1)
            s.table[slot] = a
            return read!(s.io, a)
        end
        dims = (Int(d1),)
    elseif d1 isa Dims
        dims = d1::Dims
    else
        dims = convert(Dims, d1::Tuple{Vararg{OtherInt}})::Dims
    end
    if isbitstype(elty)
        n = prod(dims)::Int
        local A
        if elty === Bool && n > 0
            A = Array{Bool, length(dims)}(undef, dims)
            i = 1
            while i <= n
                b = read(s.io, UInt8)::UInt8
                v = (b >> 7) != 0
                count = b & 0x7f
                nxt = i + count
                while i < nxt
                    A[i] = v
                    i += 1
                end
            end
        else
            ########## Added code
            A = Serialization.retrieve_if_in_memory(s, elty, dims)
            ##########
            read!(s.io, A)
        end
        s.table[slot] = A
        return A
    end
    A = Array{elty, length(dims)}(undef, dims)
    s.table[slot] = A
    sizehint!(s.table, s.counter + div(length(A)::Int,4))
    deserialize_fillarray!(A, s)
    return A
end