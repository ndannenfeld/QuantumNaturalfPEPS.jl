function init_PEPS(Lx::Int64, Ly::Int64, phys_dim::Int64, b_dim::Int64)
    tensors = Array{ITensor}(undef, Ly,Lx)

    # initializing bonds
    indices = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        indices[i] = Index(b_dim, "ind_$(i)")
    end
    for i in 1:Ly
        for j in 1:Lx
            phys_ind = Index(phys_dim, "phys_$(i)_$(j)")
            if i == 1
                if j == 1
                    peps_tensor = randomITensor(indices[1], indices[Ly*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(indices[Lx-1], indices[Ly*(Lx-1)+Lx], phys_ind)
                else
                    peps_tensor = randomITensor(indices[j-1],indices[j],indices[Ly*(Lx-1)+j], phys_ind)
                end
            elseif i == Ly
                if j == 1
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-2)*Lx+1], indices[(Ly-1)*(Lx-1)+1], phys_ind)
                elseif j == Lx
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-1)*Lx], indices[Ly*(Lx-1)], phys_ind)
                else
                    peps_tensor = randomITensor(indices[Ly*(Lx-1)+(Ly-2)*Lx+j], indices[(Ly-1)*(Lx-1)+j-1], indices[(Ly-1)*(Lx-1)+j], phys_ind)
                end
            elseif j == 1
                peps_tensor = randomITensor(indices[Ly*(Lx-1)+(i-2)*Lx+1], indices[Ly*(Lx-1)+(i-1)*Lx+1], indices[(i-1)*(Lx-1)+1], phys_ind)
            elseif j == Lx
                peps_tensor = randomITensor(indices[Ly*(Lx-1)+(i-1)*Lx], indices[Ly*(Lx-1)+(i)*Lx], indices[i*(Lx-1)], phys_ind)
            else
                peps_tensor = randomITensor(indices[(i-1)*(Lx-1)+j-1], indices[(i-1)*(Lx-1)+j], indices[Ly*(Lx-1)+(i-2)*Lx+j], indices[Ly*(Lx-1)+(i-1)*Lx+j], phys_ind)
            end

            tensors[i,j] = peps_tensor
            
        end
    end
    
    return tensors
end

function init_Product_State(Lx::Int64, Ly::Int64, phys_dim::Int64, S = nothing)
    if S === nothing
        S = [rand(phys_dim) for i=1:Lx, j=1:Ly]
    end

    product_state = Array{ITensor}(undef, Ly,Lx)

    for i in 1:Ly
        for j in 1:Lx
            product_state[i,j] = ITensor(S[i,j], Index(phys_dim, "phys_$(i)_$(j)")) 
        end
    end

    return product_state
end

function peps_product(S::Array{Array{Float64,1}, 2}, peps::Array{ITensor, 2}, mdim::Int64 = 10, co::Float64 = 1e-6)
    E = Array{MPS,2}(undef, size(S,2)-1, 2)
    f = Array{Float64}(undef, size(S,2)-1, 2)
    out = Array{Float64}(undef, 2)
    for q = 1:2
        for i in 1:size(S,2)
            q == 1 ? i_prime = i : i_prime = size(S,1)+1-i 
            if i == 1
                E[i,q] = MPS([peps[i_prime,j]*ITensor(S[i_prime,j], inds(peps[i_prime,j],"phys_$(i_prime)_$(j)")) for j in 1:size(S,1)])
                f[i,q] = log(norm(E[i,q]))
                E[i,q] /= exp(f[i,q])
            elseif i == size(S,2)
                out[q] = inner(E[i-1,q],MPS([peps[i_prime,j]*ITensor(S[i_prime,j], inds(peps[i_prime,j],"phys_$(i_prime)_$(j)")) for j in 1:size(S,1)])) * exp(f[i-1,q])
            else
                # In-place function apply! not found ?
                E[i,q] = apply(MPO([peps[i_prime,j]*ITensor(S[i_prime,j], inds(peps[i_prime,j],"phys_$(i_prime)_$(j)")) for j in 1:size(S,1)]), E[i-1,q], maxdim=mdim, cutoff=co)
                f[i,q] = log(norm(E[i,q]))+f[i-1,q]
                E[i,q] /= exp(f[i,q]-f[i-1,q])
            end
        end
    end
    return E, f, out
end

function inner_peps(psi::Array{ITensor}, psi2::Array{ITensor})
    x = 1
    for i in 1:size(psi,1)
        for j in 1:size(psi,2)
            x *= conj(psi[i,j])*psi2[i,j]*delta(inds(psi[i,j], "phys_$(i)_$(j)"), inds(psi2[i,j], "phys_$(i)_$(j)"))
        end
    end
    return x[1]
end

function differentiate(E::Array{MPS,2}, f::Array{Float64,2}, peps::Array{ITensor,2}, S::Array{Array{Float64,1},2}, i::Int64, j::Int64)
    if i == 1
        return exp(f[end,2])*contract(E[end,2].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)])
    elseif i == size(S,2)
        return exp(f[end,1])*contract(E[end,1].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)])
    end   
    return exp(f[i-1,1])*exp(f[end-i+1,2])*contract(E[i-1,1].*(E[end-i+1,2].*[(j_p != j ? peps[i,j_p]*ITensor(S[i,j_p], inds(peps[i,j_p],"phys_$(i)_$(j_p)")) : 1) for j_p in 1:size(S,1)]))
end
