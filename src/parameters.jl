bc_type = Base.Broadcast.Broadcasted{Broadcast.Style{Parameters{T}}, Nothing, F, Tuple{Parameters{T}, O}} where T <:AbstractPEPS where O where F
function Base.materialize!(dest::Parameters{<:AbstractPEPS}, bc::bc_type)
    # TODO: Avoid copying the data
    θ = convert(Vector, bc.args[1].obj)
    bc = Base.Broadcast.Broadcasted((bc.f), (θ, bc.args[2]))
    θdot = Base.materialize(bc)
    write!(dest.obj, θdot)
    dest
end

LinearAlgebra.norm(p::Parameters{<:AbstractPEPS}) = norm(norm.(all_params(p.obj)))