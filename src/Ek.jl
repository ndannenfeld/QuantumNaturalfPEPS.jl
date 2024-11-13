
# function that sorts the dictionary of flipped spin Ek_terms into the categories horizontal,vertical and 4body Ek_terms. Also sorts so that the row-number gets bigger
# vertical=false sorts all vertical Ek_terms into the 4body array
function sort_dict(Ek_terms; vertical=true)
    hor = Vector{Any}()
    vert = Vector{Any}()
    four = Vector{Any}()
    other = Vector{Any}()
    
    # loop through every term
    for key in keys(Ek_terms)
        if key == ()
            # do nothing
        elseif length(key) == 1     # one spin flip is considered a horizontal
            insert(hor, key)
        elseif length(key) == 2
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
        else # TODO: check here wheter it is a 4-body term
            insert(four, key)
        # else
        # insert(other, key)
        end
    end
    return hor, vert, four, other
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



# a function that computes the contraction of the PEPS with flipped spins at a position specified in key
function get_4body_term(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, key, fourb_envs_r, fourb_envs_l)
    con = 1
    f = 0
    
    x = map(t -> t[1][1], key)
    y = map(t -> t[1][2], key)
    
    for i in 1:length(x)
        con = get_flipped(peps, S, x[i], y[i])*con
    end

    unflipped_spins = setdiff([(xi, yi) for xi in unique(x), yi in unique(y)], zip(x, y))
    for comb in unflipped_spins
        con = get_projected(peps, S, comb[1], comb[2])*con
    end

    if minimum(y) != 1
        con = con*fourb_envs_l[minimum(y)-1]
    end
        
    if minimum(x) != 1
        con = con*env_top[minimum(x)-1].env[minimum(y)]
        if minimum(y) != maximum(y)
            con = con*env_top[minimum(x)-1].env[maximum(y)]
        end
        f += env_top[minimum(x)-1].f
    end
        
    if maximum(x) != size(peps, 1)
        con = con*env_down[end-maximum(x)+1].env[minimum(y)]
        if minimum(y) != maximum(y)
            con = con*env_down[end-maximum(x)+1].env[maximum(y)]
        end
        f += env_down[end-maximum(x)+1].f
    end
    
    if maximum(y) != size(peps, 2)
        con = con*fourb_envs_r[maximum(y)]
    end

    c = con[]
    if c < 0
        c = complex(c)
    end
    logψ_flipped = log(c) + f
    
    return logψ_flipped
end

# same as get_4body_term but for horizontal Ek_terms
function get_term(peps::PEPS, env_top::Vector{Environment}, env_down::Vector{Environment}, S::Matrix{Int64}, key, h_envs_r, h_envs_l)
    f = 0
    
    x = key[1][1][1]
    y = [key[1][1][2]] # TODO: waht is x, y?
    
    #flip = peps[x,y[1]]*ITensor([S[x,y[1]], (S[x,y[1]]+1)%2], siteind(peps,x,y[1]))
    flip = get_projected(peps, (S.+1).%2, x, y[1])
    if minimum(y) != 1
        flip = flip*h_envs_l[minimum(y)-1]
    end
    
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
        #flip = flip*(peps[x,y[2]]*ITensor([S[x,y[2]], (S[x,y[2]]+1)%2], siteind(peps,x,y[2])))
        flip = flip*get_projected(peps, (S.+1).%2, x, y[2])
        if x != size(peps, 1)
            flip = flip*env_down[end-x+1].env[y[2]]
        end
        if x != 1
            flip = env_top[x-1].env[y[2]]*flip
        end
    end
        
    if maximum(y) != size(peps, 2)
        flip = flip*h_envs_r[maximum(y)]
    end
    c = contract(flip)[]
    if c < 0
        c = complex(c)
    end
    logψ_flipped = log(c) + f
       
    return logψ_flipped
end

# computes the local energy <sample|H|ψ>/<sample|ψ>
function get_logψ_flipped(peps::PEPS, Ek_terms, env_top::Vector{Environment}, env_down::Vector{Environment}, sample::Matrix{Int64}, logψ::Number, h_envs_r::Array{ITensor}, h_envs_l::Array{ITensor}; fourb_envs_r=nothing, fourb_envs_l=nothing, logψ_flipped=nothing)
    
    if logψ_flipped === nothing
        logψ_flipped = Dict{Any, Number}()
    end
      
    # deals with the term with no flipped spin
    if haskey(Ek_terms, ())
        logψ_flipped[()] = logψ
    end
    
    # sorts the dictionary into the different categories
    horizontal, vertical, fourBody, other = sort_dict(Ek_terms, vertical=false)

    # loop through every horizontal components
    for key in horizontal 
        # calculate the Energy contribution of the specific term and add it to the total Ek
        if !haskey(logψ_flipped, key)
            logψ_flipped[key] = get_term(peps, env_top, env_down, sample, key, h_envs_r[key[1][1][1], :], h_envs_l[key[1][1][1], :])
        end
    end
    
    # same for non-horizontal Ek_terms
    if !isempty(fourBody)
        if fourb_envs_r === nothing || fourb_envs_l === nothing 
            fourb_envs_r, fourb_envs_l = get_all_4b_envs(peps, env_top, env_down, sample)
        end
        for key in fourBody
            if !haskey(logψ_flipped, key)
                row_values = map(t -> t[1][1], key)
                upperrow = minimum(row_values)
                logψ_flipped[key] = get_4body_term(peps, env_top, env_down, sample, key, fourb_envs_r[upperrow, :], fourb_envs_l[upperrow, :])
            end
        end
    end

    if !isempty(other)
        @warn "Only nearest and next nearest neighbour interactions are efficiently supported. Note that if the opertor is in he computational basis, any interaction length is possible."
        # TODO: fix the pseudocode below
        #for key in other
        #    if !haskey(logψ_flipped, key)
        #       sample_flipped = sample
        #       logψ_flipped[key] = get_logpsi(sample_flipped)
        #    end
        #end
    end

    return logψ_flipped
end

function get_Ek(peps::PEPS, ham::OpSum, sample; kwargs...)
    hilbert = siteinds(peps)
    ham_op = TensorOperatorSum(ham, hilbert)
    return get_Ek(peps, ham_op, sample; kwargs...)
end

function get_Ek(peps::PEPS, ham_op::TensorOperatorSum, sample; kwargs...)
    # get the environment tensors
    logψ, env_top, env_down = get_logψ_and_envs(peps, sample) # compute the environments of the peps according to that sample
    h_envs_r, h_envs_l = get_all_horizontal_envs(peps, env_top, env_down, sample) # computes the horizontal environments of the already sampled peps
    return get_Ek(peps, ham_op, env_top, env_down, sample, logψ, h_envs_r, h_envs_l; kwargs...)
end


function get_Ek(peps::PEPS, ham_op::TensorOperatorSum, env_top::Vector{Environment}, env_down::Vector{Environment}, sample::Matrix{Int64}, logψ::Number, h_envs_r::Array{ITensor}, h_envs_l::Array{ITensor}; fourb_envs_r=nothing, fourb_envs_l=nothing, logψ_flipped=nothing)
    Ek_terms = QuantumNaturalGradient.get_precomp_sOψ_elems(ham_op, sample .+ 1; get_flip_sites=true)
    
    logψ_flipped = get_logψ_flipped(peps, Ek_terms, env_top, env_down, sample, logψ, h_envs_r, h_envs_l; fourb_envs_r, fourb_envs_l, logψ_flipped)
    
    Ek = 0
    for key in keys(Ek_terms)
        if key == ()
            Ek += Ek_terms[key]
        else
            Ek += Ek_terms[key] * exp(logψ_flipped[key] - logψ)
        end
    end

    return convert_if_real(Ek)
end