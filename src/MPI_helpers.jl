
struct MPIFuture
    req::MPI.Request
    f::Future
    data::Any
end
function Base.wait(f::MPIFuture)
    wait(f.f)
    MPI.Wait(f.req)
    return nothing
end

function Base.fetch(f::MPIFuture)
    wait(f)
    return f.data
end


function run_and_send(func, tag, dest, args...; kwargs...)
    out = func(args...; kwargs...)
    MPI.Isend(out, MPI.COMM_WORLD; tag, dest=dest - 1)
    return nothing
end

function remotecall_mpi!(recv_mesg, func, w, args...; kwargs...)
    tag = rand(1:100000)
    my_id = myid()
    r = remotecall((args...; kwargs...) -> run_and_send(func, tag, my_id), w, args...; kwargs...)
    rreq = MPI.Irecv!(recv_mesg, MPI.COMM_WORLD; source=w-1, tag)
    return MPIFuture(rreq, r, recv_mesg)
end

remotecall_fetch_mpi!(recv_mesg, func, w, args...; kwargs...) = fetch(remotecall_mpi!(recv_mesg, func, w, args...; kwargs...))
remotecall_wait_mpi!(recv_mesg, func, w, args...; kwargs...) = wait(remotecall_mpi!(recv_mesg, func, w, args...; kwargs...))
# MPI.Initialized()