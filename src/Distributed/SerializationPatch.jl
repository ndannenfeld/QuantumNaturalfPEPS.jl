Serialization = Distributed.Serialization
deserialize, AbstractSerializer = Serialization.deserialize, Serialization.AbstractSerializer


"""
Example:
QuantumNaturalfPEPS.patch_serialization()
O = zeros(2, 2)
worker = 2
Oi = @view O[:, 1:1]
Distributed.Serialization.register_deserialization_array!(worker, Oi)
r = Distributed.remotecall(() -> ones(2, 1), worker)
fetch(r)
@show O
# O = [1.0 0.0; 1.0 0.0]
"""
function patch_serialization()
    """
    If you register an array with `register_deserialization_array!(worker_id, array)`, then the next time an array gets deserialized with the same type and dimensions, the array that was registered will be overwritten with the deserialized array. This is useful to not have to allocate memory for the same array multiple times.
    """
    @eval Serialization function register_deserialization_array!(worker_id::Int, x)
        if !isdefined(Serialization, :deserialization_array)
            @eval Serialization deserialization_array = Dict()
            @eval Serialization deserialization_array_lock = ReentrantLock()
        end
        elty = eltype(x)
        dims = size(x)
        lock(Serialization.deserialization_array_lock) do
            Serialization.deserialization_array[(worker_id, elty, dims)] = x
        end
    end

    @eval Serialization function deregister_deserialization_array!(worker_id::Int, x)
        elty = eltype(x)
        dims = size(x)
        lock(Serialization.deserialization_array_lock) do
            delete!(Serialization.deserialization_array, (worker_id, elty, dims))
        end
    end

    @eval Serialization function retrieve_if_in_memory(s::AbstractSerializer, elty, dims)
        if hasproperty(s, :pid) && isdefined(Serialization, :deserialization_array)
            id = (s.pid, elty, dims)
            if id in keys(Serialization.deserialization_array)
                A = Serialization.deserialization_array[id]
                lock(Serialization.deserialization_array_lock) do
                    delete!(Serialization.deserialization_array, id)
                end
                return A
            end
        end
        return Array{elty, length(dims)}(undef, dims)
    end

    @eval Serialization function deserialize_array(s::AbstractSerializer)
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
                ########## Modified Code
                #a = Vector{elty}(undef, d1)
                a = retrieve_if_in_memory(s, elty, (d1,))
                ##########
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
                ########## Modified Code
                # A = Array{Bool, length(dims)}(undef, dims)
                A = retrieve_if_in_memory(s, Bool, dims)
                ##########
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
                ########## Modified Code
                # A = Array{elty, length(dims)}(undef, dims)
                A = retrieve_if_in_memory(s, elty, dims)
                ##########
                read!(s.io, A)
            end
            s.table[slot] = A
            return A
        end
        ########## Modified Code
        #A = Array{elty, length(dims)}(undef, dims)
        A = retrieve_if_in_memory(s, elty, dims)
        ##########
        s.table[slot] = A
        sizehint!(s.table, s.counter + div(length(A)::Int,4))
        deserialize_fillarray!(A, s)
        return A
    end
end