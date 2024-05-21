# the entries of the environments are all normalized by the absolute value of the biggest entrie of the first ITensor
# to get the true environtment: contract with the MPS and afterwards multiply by exp(f)
mutable struct Environment
    env::MPS
    f::ComplexF64
    Environment(env, f) = new(env, f)
end

mutable struct PEPS
    tensors::Matrix{ITensor}
    double_layer_envs::Vector{Environment}
    norm::Float64
    bond_dim::Integer
    sample_dim::Integer
    contract_dim::Integer
    double_contract_dim::Integer
    PEPS(tensors, bond_dim; norm=0, sample_dim=bond_dim, contract_dim=3*bond_dim, double_contract_dim=2*bond_dim) = new(tensors, [Environment(MPS(), 1) for i in 1:size(tensors)[1]-1], norm, bond_dim, sample_dim, contract_dim, double_contract_dim)
end

Base.size(peps::PEPS) = size(peps.tensors)
Base.getindex(peps::PEPS, i::Int) = peps.tensors[i, :]
Base.getindex(peps::PEPS, i::Int, j::Int) = peps.tensors[i, j] # You can use peps[i, j]
Base.setindex!(peps::PEPS, v, i::Int, j::Int) = (peps.tensors[i, j] = v)

function flatten(peps::PEPS) # Flattens the tensors into a vector
    θ = ComplexF64[]
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            append!(θ, reshape(Array(peps[i,j], inds(peps[i,j])), :))
        end
    end
    return θ
end

Base.length(peps::PEPS) = length(flatten(peps)) # length(θ)

function write!(peps::PEPS, θ::Vector{ComplexF64}) # Writes the vector θ into the tensors.
    pos = 1
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            shift = prod(dim.(inds(peps[i,j])))
            peps[i,j] *= im
            peps[i,j][:] = reshape(θ[pos:(pos+shift-1)], dim.(inds(peps[i,j])))
            pos += shift
        end
    end
end

# initializes a PEPS
function PEPS(Lx::Int64, Ly::Int64, phys_dim::Int64, bond_dim::Int64)
    tensors = Array{ITensor}(undef, Lx,Ly)

    # initializing bond indices
    indices = Array{Index{Int64}}(undef, (2*Lx*Ly - Lx - Ly))
    for i in 1:(2*Lx*Ly - Lx - Ly)
        indices[i] = Index(bond_dim, "ind_$(i)")
    end
    
    # filling the matrix of tensors with random ITensors wich share the same indices with their neighbours
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
            
            tensors[j,i] = peps_tensor
            
        end
    end
    
    return PEPS(tensors, bond_dim)
end

# Computes the environments and log(<ψ|S>)
function get_logψ_and_envs(peps::PEPS, S::Array{Int64,2}, Env_top=Array{Environment}(undef, size(S,1)-1))
    
    overwrite = true    # if Env_top is given and the bond dimension is sufficient, we do not need to calculate it again
    if isdefined(Env_top, 1) && maxlinkdim(Env_top[1].env) >= peps.contract_dim
        overwrite = false
    end
    
    Env_down = Array{Environment}(undef, size(S,1)-1)
    out = Array{ComplexF64}(undef, 2)
    
    if overwrite
        Env_top[1] = generate_env_row(peps[1], S[1,:], 1, peps.contract_dim)
    end
    Env_down[1] = generate_env_row(peps[size(S,1)], S[size(S,1),:], size(S,1), peps.contract_dim)
    
    # for every row we calculate the environments once from the top down and once from the bottom up
    for i in 2:size(S,1)-1
        i_prime = size(S,1)+1-i 
        
        if overwrite
            Env_top[i] = generate_env_row(peps[i], S[i,:], i, peps.contract_dim, env_row_above = Env_top[i-1])
        end
        Env_down[i] = generate_env_row(peps[i_prime], S[i_prime,:], i_prime, peps.contract_dim, env_row_above = Env_down[i-1])
    end
    
    # once we calculated all environments we calculate <ψ|S> using the environments
    out[1] = contract(Env_top[end].env.*MPS([peps[size(S,1),j]*ITensor([(S[end,j]+1)%2, S[end,j]], inds(peps[size(S,1),j],"phys_$(j)_$(size(S,1))")) for j in 1:size(S,2)]))[1] * exp(Env_top[end].f)
    out[2] = contract((MPS([peps[1,j]*ITensor([(S[1,j]+1)%2, S[1,j]], inds(peps[1,j],"phys_$(j)_$(1)")) for j in 1:size(S,2)])).*Env_down[end].env)[1] * exp(Env_down[end].f)
    
    return mean(log.(Complex.((out)))), Env_top, Env_down
end

# calculates the environments for a given row and contracts that with env_row_above
function generate_env_row(peps_row, S_row, i, contract_dim; env_row_above=nothing)
    Env = [peps_row[j]*ITensor([(S_row[j]+1)%2, S_row[j]], inds(peps_row[j],"phys_$(j)_$(i)")) for j in 1:length(S_row)]
    norm_shift = 0
    if env_row_above == nothing
        Env = MPS(Env)
    else
        Env = apply(MPO(Env), env_row_above.env, maxdim=contract_dim)
        norm_shift = env_row_above.f
    end
    
    normE = maximum(abs.(reshape(Array(Env[1], inds(Env[1])), :)))
    return Environment(Env./normE, length(Env)*log(normE)+norm_shift)
end

# returns the exact inner product of 2 peps (only used for testing purposes)
function inner_peps(psi::PEPS, psi2::PEPS)
    x = 1
    for i in 1:size(psi)[1]
        for j in 1:size(psi)[2]
            x *= psi[i,j]*psi2[i,j]*delta(inds(psi[i,j], "phys_$(j)_$(i)"), inds(psi2[i,j], "phys_$(j)_$(i)"))
        end
    end
    return x[1]
end

# calculates the gradient: d(<ψ|S>)/d(Θ) / <ψ|S>
function get_Ok(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number; Ok::Vector=complex(zeros(length(peps))))
    pos = 1
    shift = 0
    for i in 1:size(peps)[1]
        for j in 1:size(peps)[2]
            # peps_S gives an array of ITensors wich are peps[i,y]*S[i,y] for all tensors in the row except for i,j
            peps_S = [(j_p != j ? peps[i,j_p]*ITensor([(S[i,j_p]+1)%2, S[i,j_p]], inds(peps[i,j_p], "phys_$(j_p)_$(i)")) : 1) for j_p in 1:size(S,2)]
            if i == 1
                # we get the differential tensor if we contract the environments with the peps_S
                Ok_Tensor = exp(env_down[end].f - logψ)*contract(env_down[end].env .* peps_S)
            elseif i == size(peps)[1]
                Ok_Tensor = exp(env_top[end].f - logψ)*contract(env_top[end].env .* peps_S)
            else
                Ok_Tensor = exp(env_top[i-1].f + env_down[end-i+1].f - logψ)*contract(env_top[i-1].env .* peps_S .* env_down[end-i+1].env)
            end

            # lastly we reshape the tensor to a vector to obtain the gradient
            shift = prod(dim.(inds(Ok_Tensor)))
            if S[i,j] == 1
                Ok[pos:pos+shift-1] = zeros(shift)
                pos = pos+shift
                Ok[pos:pos+shift-1] = reshape(Array(Ok_Tensor, inds(Ok_Tensor)), :)
            else
                Ok[pos:pos+shift-1] = reshape(Array(Ok_Tensor, inds(Ok_Tensor)), :)
                pos = pos+shift
                Ok[pos:pos+shift-1] = zeros(shift)
            end
            pos = pos+shift
        end
    end
    return Ok
end

# renames the indices changing_inds to new ones and stores them in indices_outer
function rename_indices!(ket, indices_outer, changing_inds)
    for i in 1:length(indices_outer)
        indices_outer[i] = Index(dim(changing_inds[i]), "Ket_$(i)")
        ket[i] = ket[i]*delta(changing_inds[i], indices_outer[i])
    end  
end

# renames all inner indices in a MPO/MPS
function rename_indices!(ket)
    indices_inner = Array{Index}(undef, length(ket)-1)
    for i in 1:length(indices_inner)
        indices_inner[i] = Index(maxlinkdim(ket), "Ket_inner_$(i)")
        old_ind = commoninds(ket[i], ket[i+1])

        ket[i] = ket[i]*delta(indices_inner[i], old_ind)
        ket[i+1] = ket[i+1]*delta(indices_inner[i], old_ind)
    end
end

# calculates a double layer environment row by contracting a peps_row with itself along the physical indices
# also combines the outgoing indices of the double layer
function generate_double_layer_env_row(peps_row, peps_row_above, contract_dim)
    indices_outer = Array{Index}(undef, length(peps_row))
    
    bra = conj(MPO(peps_row))
    ket = MPO(peps_row)
     
    rename_indices!(ket)
    com_inds = commoninds.(peps_row, peps_row_above)
    rename_indices!(ket, indices_outer, com_inds)

    E_mpo = contract(bra,ket,maxdim=contract_dim)
    
    E_mps = MPS((E_mpo.*combiner.(indices_outer, com_inds, tags="-1")).data)
    
    normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    return Environment(E_mps./normE, length(bra)*log(normE))
end

function generate_double_layer_env_row(peps_row, peps_row_above, peps_row_below, peps_double_env, contract_dim)
    C = Array{Array{ITensor}}(undef, 2)
    indices_outer = Array{Index}(undef, length(peps_row))

    bra = conj(MPO(peps_row))
    ket = MPO(peps_row)
    
    rename_indices!(ket, indices_outer, commoninds.(peps_row, peps_row_above))
    C[1] = combiner.(indices_outer, commoninds.(peps_row, peps_row_above), tags="$(-1)")
    rename_indices!(ket, indices_outer, commoninds.(peps_row, peps_row_below))
    C[2] = combiner.(indices_outer, commoninds.(peps_row, peps_row_below), tags="$(1)")

    rename_indices!(ket)

    E_mpo = contract(bra,ket,maxdim=contract_dim)
    E_mpo = E_mpo.*C[1]
    E_mpo = E_mpo.*C[2]

    E_mps = apply(E_mpo.*delta.(reduce(vcat, collect.(inds.(E_mpo, "1"))), reduce(vcat, collect.(inds.(peps_double_env.env, "-1")))), peps_double_env.env, maxdim=contract_dim)

    normE = maximum(abs.(reshape(Array(E_mps[1], inds(E_mps[1])), :)))
    return Environment(E_mps./normE, length(bra)*log(normE)+peps_double_env.f)
end

# calculates the field double_layer_envs and the norm of peps
function update_double_layer_envs!(peps::PEPS)
    indices_outer = Array{Index}(undef, size(peps)[2])
        
    # for every row we calculate the double layer environment
    peps.double_layer_envs[end] = generate_double_layer_env_row(peps[size(peps)[1]], peps[size(peps)[1]-1], peps.double_contract_dim)
    for i in size(peps)[1]-1:-1:2
        peps.double_layer_envs[i-1] = generate_double_layer_env_row(peps[i], peps[i-1], peps[i+1], peps.double_layer_envs[i], peps.double_contract_dim)
    end

    # We also calculate the (log-)norm of the peps as it is used in get_sample to calculate p_c
    E_mpo = generate_double_layer_env_row(peps[1], peps[2], peps.double_contract_dim)
    E_mpo.env = E_mpo.env .*delta.(reduce(vcat, collect.(inds.(E_mpo.env, "-1"))), reduce(vcat, collect.(inds.(peps.double_layer_envs[1].env, "-1"))))
    
    for i in 1:length(E_mpo.env)
        E_mpo.env[i] = (E_mpo.env[i]*peps.double_layer_envs[1].env[i])
    end

    peps.norm = real(log(Complex(contract(E_mpo.env)[1])))+(peps.double_layer_envs[1].f + E_mpo.f)    
end

# calculates the bra and the ket layer and applies (if available) already sampled rows (from above)
function get_bra_ket!(peps, row, indices_outer, env_top=nothing)
    bra = MPO(peps[row])
    ket = copy(bra)

    if row != 1      
        bra = apply(bra, env_top[row-1].env, maxdim=peps.sample_dim)
        ket = apply(ket, env_top[row-1].env, maxdim=peps.sample_dim)
    end
    rename_indices!(ket)
    
    if row != size(peps)[1]
        rename_indices!(ket, indices_outer, commoninds.(peps[row], peps[row+1]))
    end
    return conj(bra), ket
end

# calculates the unsampled contractions along a row (from right to left the sites are contracted along the physical Index)
function calculate_E!(bra, ket, peps, row, E, indices_outer)
    if row != size(peps)[1]
        C = combiner(indices_outer[end], commoninds(peps[row,size(peps)[2]], peps[row+1,size(peps)[2]]), tags = "1")
        E[end] = contract(bra[end]*ket[end]*C*delta(inds(peps.double_layer_envs[row].env[end], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[end])

        for i in size(peps)[2]-1:-1:2
            C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
            E[i-1] = contract(E[i]*bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i])
        end
    else
        E[end] = contract(bra[end]*ket[end])
        for i in size(peps)[2]-1:-1:2
            E[i-1] = contract(E[i]*bra[i]*ket[i])
        end
    end
end

# returns the 2x2 matrix P_S which is needed to sample from. Also updates sigma (used to store the contraction of already sampled sites from the left edge to the current site)
function get_PS(bra, ket, peps, row, i, E, indices_outer, sigma)
    ket[i] = delta(inds(ket[i], "phys_$(i)_$(row)"), Index(2, "ket_phys"))*ket[i]
    if row != size(peps)[1]
        C = combiner(indices_outer[i], commoninds(peps[row,i], peps[row+1,i]), tags = "1")
        sigma_1 = bra[i]*ket[i]*C*delta(inds(peps.double_layer_envs[row].env[i], "-1")[1], inds(C)[1])*peps.double_layer_envs[row].env[i]
    else
        sigma_1 = bra[i]*ket[i]
    end

    if i == 1
        P_S = contract(E[i]*sigma_1)
    elseif i == size(peps)[2]
        P_S = contract(sigma*sigma_1)
    else
        P_S = contract(sigma*E[i]*sigma_1)
    end  
    return P_S, sigma_1
end

# samples from P_S and updates pc
function sample_PS!(P_S, pc)
    p0 = abs(P_S[1,1])
    p1 = abs(P_S[2,2])
    
    @assert imag(P_S[1,1]) < real(P_S[1,1])*10^(-6)
    if rand() < p0/(p0+p1)
        pc += log(p0/(p0+p1))
        return p0/(p0+p1), 0, pc
    else
        pc += log(p1/(p0+p1))
        return p1/(p0+p1), 1, pc
    end
end

# after the sampling of the current site, it is fixed and its contraction with the aleady sampled sites is stored in sigma
function update_sigma(sigma, sigma_1, S, i, row, norm_factor)
    s = (sigma_1*sigma)* ITensor([(S+1)%2, S], inds(sigma_1, "phys_$(i)_$(row)"))
    return (s)* ITensor([(S+1)%2, S], inds(sigma_1, "ket_phys")) / norm_factor
end

# generates a sample of a given peps along with pc and the top environments
function get_sample(peps::PEPS)
    S = Array{Int64}(undef, size(peps))
    
    indices_inner = Array{Index}(undef, size(peps)[2]-1)
    indices_outer = Array{Index}(undef, size(peps)[2])
    
    E = Array{ITensor}(undef, size(peps)[2]-1)
    
    env_top = Array{Environment}(undef, size(peps)[1]-1)
    
    P_S = ITensor()
    
    psi_S = 0
    pc = 0
    # we loop through every row
    for row in 1:size(peps)[1]
        sigma = 1
        bra, ket = get_bra_ket!(peps, row, indices_outer, env_top)
        
        # we then calculate the unsampled environment (in one row)
        calculate_E!(bra, ket, peps, row, E, indices_outer)

        # then we loop through the different sites in one row
        for i in 1:size(peps)[2]
            
            # calculate the 2x2 matrix from which we sample
            P_S, sigma_1 = get_PS(bra, ket, peps, row, i, E, indices_outer, sigma)
            
            # sample from P_S
            norm_factor, S[row,i], pc = sample_PS!(P_S, pc)
            
            # store the contraction of sampled sites in sigma
            sigma = update_sigma(sigma, sigma_1, S[row,i], i, row, norm_factor)                        
        end
        
        # the sampled bra is used to generate the top environments
        bra = bra.*[ITensor([(S[row,i]+1)%2, S[row,i]], inds(bra[i], "phys_$(i)_$(row)")) for i in 1:size(peps)[2]]
            
        if row != size(peps)[1]
            norm_bra = maximum(abs.(reshape(Array(bra[1], inds(bra[1])), :)))
            if row == 1
                env_top[row] = Environment(MPS(bra.data)./norm_bra, length(bra)*log(norm_bra)) 
            else
                env_top[row] = Environment(bra./norm_bra, length(bra)*log(norm_bra)+env_top[row-1].f)
            end
        else
            psi_S = 2*real(log(Complex(contract(bra)[1])) + env_top[row-1].f)
        end
    end
    
    return S, pc,psi_S ,env_top
end

# function that sorts the dictionary of flipped spin terms into the categories horizontal,vertical and 4body terms. Also sorts so that the row-number gets bigger
# vertical=false sorts all vertical terms into the 4body array
function sort_dict(terms; vertical=true)
    hor = Vector{Any}()
    vert = Vector{Any}()
    four = Vector{Any}()
    
    # loop through every term
    for key in keys(terms)
        if length(key) == 1     # one spin flip is considered a horizontal
            insert(hor, key)
        else
            if key[1][1][1] == key[2][1][1]  #same row
                insert(hor, key)
            elseif key[1][1][2] == key[2][1][2]  #same column
                if vertical
                    insert(vert, key)
                else
                    insert(four, key)
                end
            else
                insert(four, key)
            end
        end
    end
    return hor,vert,four
end
   
# inserts the element x into the array arr at the desired position
function insert(arr, x)
    if isempty(arr)
        push!(arr,x)
        return
    end
    x_min = x[1][1][1]
    if length(x) == 2
        xmin = minimum([x_min, x[2][1][1]])
    end
    
    for i in 1:length(arr)
        if arr[i][1][1][1] >= x_min
            insert!(arr,i,x)
            return
        end
    end
    push!(arr,x)
end

# this function computes the horizontal environments for a given row
function get_horizontal_envs!(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs::Matrix{MPS})
    peps_i = peps[i].*[ITensor([(S[i,k]+1)%2, S[i,k]], inds(peps[i,k], "phys_$(k)_$(i)")) for k in 1:size(peps)[2]]     #contract the row with S
    
    # now we loop through every site and compute the environments (once from the right and once from the left) by MPO-MPS contraction.
    if i == 1
        horizontal_envs[1,end] = MPS([peps_i[end],env_down[end].env[end]])
        horizontal_envs[2,1] = MPS([peps_i[1],env_down[end].env[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([peps_i[j],env_down[end].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([peps_i[j],env_down[end].env[j]]), horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    elseif i == size(peps)[1]
        horizontal_envs[1,end] = MPS([env_top[end].env[end], peps_i[end]])
        horizontal_envs[2,1] = MPS([env_top[end].env[1], peps_i[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[end].env[j], peps_i[j]]),horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([env_top[end].env[j], peps_i[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    else
        horizontal_envs[1,end] = MPS([env_top[i-1].env[end],peps_i[end],env_down[end-i+1].env[end]])
        horizontal_envs[2,1] = MPS([env_top[i-1].env[1],peps_i[1],env_down[end-i+1].env[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[i-1].env[j],peps_i[j],env_down[end-i+1].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([env_top[i-1].env[j],peps_i[j],env_down[end-i+1].env[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    end
end

# same as above but for non-horizontal components
function get_4body_envs!(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs::Matrix{MPS})
    peps_i = peps[i].*[ITensor([(S[i,k]+1)%2, S[i,k]], inds(peps[i,k], "phys_$(k)_$(i)")) for k in 1:size(peps)[2]]
    peps_j = peps[i+1].*[ITensor([(S[i+1,k]+1)%2, S[i+1,k]], inds(peps[i+1,k], "phys_$(k)_$(i+1)")) for k in 1:size(peps)[2]]
    
    if i == 1
        horizontal_envs[1,end] = MPS([peps_i[end],peps_j[end],env_down[end-1].env[end]])
        horizontal_envs[2,1] = MPS([peps_i[1],peps_j[1],env_down[end-1].env[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([peps_i[j],peps_j[j],env_down[end-1].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([peps_i[j],peps_j[j],env_down[end-1].env[j]]), horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    elseif i == size(peps)[1]-1
        horizontal_envs[1,end] = MPS([env_top[end-1].env[end], peps_i[end], peps_j[end]])
        horizontal_envs[2,1] = MPS([env_top[end-1].env[1], peps_i[1], peps_j[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[end-1].env[j], peps_i[j], peps_j[j]]),horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([env_top[end-1].env[j], peps_i[j], peps_j[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    else
        horizontal_envs[1,end] = MPS([env_top[i-1].env[end],peps_i[end],peps_j[end],env_down[end-i].env[end]])
        horizontal_envs[2,1] = MPS([env_top[i-1].env[1],peps_i[1],peps_j[1],env_down[end-i].env[1]])
        for j in size(peps)[2]-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[i-1].env[j],peps_i[j],peps_j[j],env_down[end-i].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps)[2]-1
            horizontal_envs[2,j] = apply(MPO([env_top[i-1].env[j],peps_i[j],peps_j[j],env_down[end-i].env[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    end
end

# a function that computes the contraction of the PEPS with one/two flipped spin(s) at a position specified in key
function get_4body_term(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, key, h_envs)
    con = 1
    f = 0
    
    x = [key[1][1][1], key[2][1][1]]
    y = [key[1][1][2], key[2][1][2]]
    
    for i in 1:2
        con = contract(con*peps[x[i],y[i]]*ITensor([S[x[i],y[i]], (S[x[i],y[i]]+1)%2], inds(peps[x[i],y[i]], "phys_$(y[i])_$(x[i])")))
        if y[1] != y[2]
            con = contract(con*peps[x[i],y[(i%2)+1]]*ITensor([(S[x[i],y[(i%2)+1]]+1)%2, S[x[i],y[(i%2)+1]]], inds(peps[x[i],y[(i%2)+1]], "phys_$(y[(i%2)+1])_$(x[i])")))
        end
    end
        
    if minimum(x) != 1
        con = contract(con*env_top[minimum(x)-1].env[minimum(y)])
        if y[1] != y[2]
            con = contract(con*env_top[minimum(x)-1].env[maximum(y)])
        end
        f += env_top[minimum(x)-1].f
    end
        
    if maximum(x) != size(peps)[1]
        con = contract(con*env_down[end-maximum(x)+1].env[minimum(y)])
        if y[1] != y[2]
            con = contract(con*env_down[end-maximum(x)+1].env[maximum(y)])
        end
        f += env_down[end-maximum(x)+1].f
    end
           
    if minimum(y) != 1
        con = contract(con*contract(h_envs[2,minimum(y)-1]))
    end
    if maximum(y) != size(peps)[2]
        con = contract(con*contract(h_envs[1,maximum(y)]))
    end
        
    return con[1], f
end

# same as get_4body_term but for horizontal terms
function get_term(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, key, h_envs)
    f = 0
    
    x = key[1][1][1]
    y = [key[1][1][2]]
    
    flip = peps[x,y[1]]*ITensor([S[x,y[1]], (S[x,y[1]]+1)%2], inds(peps[x,y[1]], "phys_$(y[1])_$(x)"))
    if x != size(peps)[1]
        flip = flip*env_down[end-x+1].env[y[1]]
        f += env_down[end-x+1].f
    end
    if x != 1
        flip = env_top[x-1].env[y[1]]*flip
        f += env_top[x-1].f
    end
    
    if length(key) == 2
        push!(y,key[2][1][2])
        flip = flip*(peps[x,y[2]]*ITensor([S[x,y[2]], (S[x,y[2]]+1)%2], inds(peps[x,y[2]], "phys_$(y[2])_$(x)")))
        if x != size(peps)[1]
            flip = flip*env_down[end-x+1].env[y[2]]
        end
        if x != 1
            flip = env_top[x-1].env[y[2]]*flip
        end
    end
        
    if minimum(y) != 1
        flip = flip*contract(h_envs[2,minimum(y)-1])
    end
    if maximum(y) != size(peps)[2]
        flip = flip*contract(h_envs[1,maximum(y)])
    end
       
    return contract(flip)[1], f
end

# computes the local energy <S|H|ψ>/<S|ψ>
function get_Ek(peps::PEPS, ham::OpSum, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, logψ::Number)
    hilbert = reshape(siteinds("S=1/2", size(peps)[1]* size(peps)[2]), size(peps)[1], size(peps)[2])
    ham_op = QuantumNaturalGradient.TensorOperatorSum(ham, hilbert)
    terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, S.+1; get_flip_sites=true)
    
    h_envs = Matrix{MPS}(undef, 2,size(peps)[2]-1)
    row = 0
    Ek = 0
      
    # deals with the term with no flipped spin
    if haskey(terms, Any[])
        Ek += terms[Any[]]
        delete!(terms, Any[])
    end
    
    # sorts the dictionary into the different categories
    horizontal,vertical,fourBody = sort_dict(terms, vertical=false)

    # loop through every horizontal components
    for key in horizontal
        if key[1][1][1] != row  # because they are ordered we only need to calculate the horizontal environments once for every row
            row = key[1][1][1]
            get_horizontal_envs!(peps,env_top, env_down, S, row, h_envs)
        end

        # calculate the Energy contribution of the specific term and add it to the total Ek
        Ek_i, f = get_term(peps, env_top, env_down, S, key, h_envs)
        Ek += (Ek_i)*exp(f-logψ)*terms[key]     # abs??   
    end
    
    # same for non-horizontal terms
    for key in fourBody
        if minimum([key[1][1][1],key[2][1][1]]) != row  
            row = minimum([key[1][1][1],key[2][1][1]])
            get_4body_envs!(peps,env_top, env_down, S, row, h_envs)
        end
        
        Ek_i, f = get_4body_term(peps, env_top, env_down, S, key, h_envs)
        Ek += (Ek_i)*exp(f-logψ)*terms[key]     # abs??   
    end
    
    return Ek
end