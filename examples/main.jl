include("../mod/DS_PEPS.jl")
using .DS_PEPS
using ITensors

Lx = 5
Ly = 5

psi = init_PEPS(Lx,Ly,2,3)
S = [rand(2) for i=1:Lx, j=1:Ly]

E, f, out = peps_product(S, psi, 10, 1e-8)
out_exact = inner_peps(psi, init_Product_State(Lx,Ly,2,S))
diff = differentiate(E, f, psi, S, 2,4)

println("<psi|S> ≈ $(out)")
println("<psi|S> = $(out_exact)")
println("Indices Tensor_ij = $(inds(psi[2,4]))")
println("Indices dpsi/dTensor_ij = $(inds(diff))")
