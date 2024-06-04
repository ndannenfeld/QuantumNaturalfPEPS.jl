
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
    peps_i = peps[i].*[ITensor([(S[i,k]+1)%2, S[i,k]], inds(peps[i,k], "phys_$(k)_$(i)")) for k in 1:size(peps, 2)]     #contract the row with S
    
    # now we loop through every site and compute the environments (once from the right and once from the left) by MPO-MPS contraction.
    if i == 1
        horizontal_envs[1,end] = MPS([peps_i[end],env_down[end].env[end]])
        horizontal_envs[2,1] = MPS([peps_i[1],env_down[end].env[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([peps_i[j],env_down[end].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
            horizontal_envs[2,j] = apply(MPO([peps_i[j],env_down[end].env[j]]), horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    elseif i == size(peps, 1)
        horizontal_envs[1,end] = MPS([env_top[end].env[end], peps_i[end]])
        horizontal_envs[2,1] = MPS([env_top[end].env[1], peps_i[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[end].env[j], peps_i[j]]),horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
            horizontal_envs[2,j] = apply(MPO([env_top[end].env[j], peps_i[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    else
        horizontal_envs[1,end] = MPS([env_top[i-1].env[end],peps_i[end],env_down[end-i+1].env[end]])
        horizontal_envs[2,1] = MPS([env_top[i-1].env[1],peps_i[1],env_down[end-i+1].env[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[i-1].env[j],peps_i[j],env_down[end-i+1].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
            horizontal_envs[2,j] = apply(MPO([env_top[i-1].env[j],peps_i[j],env_down[end-i+1].env[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    end
end

# same as above but for non-horizontal components
function get_4body_envs!(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, i::Int64, horizontal_envs::Matrix{MPS})
    peps_i = peps[i].*[ITensor([(S[i,k]+1)%2, S[i,k]], inds(peps[i,k], "phys_$(k)_$(i)")) for k in 1:size(peps, 2)]
    peps_j = peps[i+1].*[ITensor([(S[i+1,k]+1)%2, S[i+1,k]], inds(peps[i+1,k], "phys_$(k)_$(i+1)")) for k in 1:size(peps, 2)]
    
    if i == 1
        horizontal_envs[1,end] = MPS([peps_i[end],peps_j[end],env_down[end-1].env[end]])
        horizontal_envs[2,1] = MPS([peps_i[1],peps_j[1],env_down[end-1].env[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([peps_i[j],peps_j[j],env_down[end-1].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
            horizontal_envs[2,j] = apply(MPO([peps_i[j],peps_j[j],env_down[end-1].env[j]]), horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    elseif i == size(peps, 1)-1
        horizontal_envs[1,end] = MPS([env_top[end-1].env[end], peps_i[end], peps_j[end]])
        horizontal_envs[2,1] = MPS([env_top[end-1].env[1], peps_i[1], peps_j[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[end-1].env[j], peps_i[j], peps_j[j]]),horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
            horizontal_envs[2,j] = apply(MPO([env_top[end-1].env[j], peps_i[j], peps_j[j]]),horizontal_envs[2,j-1], maxdim = peps.contract_dim)
        end
    else
        horizontal_envs[1,end] = MPS([env_top[i-1].env[end],peps_i[end],peps_j[end],env_down[end-i].env[end]])
        horizontal_envs[2,1] = MPS([env_top[i-1].env[1],peps_i[1],peps_j[1],env_down[end-i].env[1]])
        for j in size(peps, 2)-1:-1:2
            horizontal_envs[1,j-1] = apply(MPO([env_top[i-1].env[j],peps_i[j],peps_j[j],env_down[end-i].env[j]]), horizontal_envs[1,j], maxdim = peps.contract_dim)
        end
        for j in 2:size(peps, 2)-1
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
        
    if maximum(x) != size(peps, 1)
        con = contract(con*env_down[end-maximum(x)+1].env[minimum(y)])
        if y[1] != y[2]
            con = contract(con*env_down[end-maximum(x)+1].env[maximum(y)])
        end
        f += env_down[end-maximum(x)+1].f
    end
           
    if minimum(y) != 1
        con = contract(con*contract(h_envs[2,minimum(y)-1]))
    end
    if maximum(y) != size(peps, 2)
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
    if x != size(peps, 1)
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
        if x != size(peps, 1)
            flip = flip*env_down[end-x+1].env[y[2]]
        end
        if x != 1
            flip = env_top[x-1].env[y[2]]*flip
        end
    end
        
    if minimum(y) != 1
        flip = flip*contract(h_envs[2,minimum(y)-1])
    end
    if maximum(y) != size(peps, 2)
        flip = flip*contract(h_envs[1,maximum(y)])
    end
       
    return contract(flip)[1], f
end

# computes the local energy <sample|H|ψ>/<sample|ψ>
function get_Ek(peps::PEPS, ham_op::TensorOperatorSum, env_top::Vector{Environment}, env_down::Vector{Environment}, sample::Matrix{Int64}, logψ::Number)
    terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, sample .+ 1; get_flip_sites=true)
    
    h_envs = Matrix{MPS}(undef, 2,size(peps, 2)-1)
    row = 0
    Ek = 0
      
    # deals with the term with no flipped spin
    if haskey(terms, Any[])
        Ek += terms[Any[]]
        delete!(terms, Any[])
    end
    
    # sorts the dictionary into the different categories
    horizontal, vertical, fourBody = sort_dict(terms, vertical=false)

    # loop through every horizontal components
    for key in horizontal
        if key[1][1][1] != row  # because they are ordered we only need to calculate the horizontal environments once for every row
            row = key[1][1][1]
            get_horizontal_envs!(peps,env_top, env_down, sample, row, h_envs)
        end

        # calculate the Energy contribution of the specific term and add it to the total Ek
        Ek_i, f = get_term(peps, env_top, env_down, sample, key, h_envs)
        Ek += (Ek_i)*exp(f-logψ)*terms[key]     # abs??   
    end
    
    # same for non-horizontal terms
    for key in fourBody
        if minimum([key[1][1][1],key[2][1][1]]) != row  
            row = minimum([key[1][1][1],key[2][1][1]])
            get_4body_envs!(peps,env_top, env_down, sample, row, h_envs)
        end
        
        Ek_i, f = get_4body_term(peps, env_top, env_down, sample, key, h_envs)
        Ek += Ek_i * exp(f - logψ)*terms[key]     # abs??   
    end
    
    return Ek
end