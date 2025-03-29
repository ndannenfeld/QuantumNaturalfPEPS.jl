function multiply_spectrum!(peps::AbstractPEPS, spectrum)
    spectrum = sqrt.(spectrum)
    spectrum = diagm(spectrum)
    for i in 1:size(peps, 1), j in 1:size(peps, 2)
        for li in linkinds(peps, i, j)
            t = itensor(spectrum , li, li')
            peps[i, j] = apply(peps[i, j], t)
        end
    end
    return peps
end

function multiply_algebraic_spectrum!(peps::AbstractPEPS, α::Number)
    bs = collect(1:maxbonddim(peps))
    spectrum = bs .^ (-α)
    return multiply_spectrum!(peps, spectrum)
end

function tensor_std(peps::AbstractPEPS)
    mean_, mean_2 = 0, 0
    l = 0
    for ten in peps.tensors
        mean_ += sum(ten.tensor.storage)
        mean_2 += sum(x->x^2, ten.tensor.storage)
        l += length(ITensors.tensor(ten))
    end
    mean_ /= l
    mean_2 /= l
    return sqrt(mean_2 - mean_^2)
end

function shift!(peps::AbstractPEPS, shift::Bool) 
    if shift
        return shift!(peps, 2 * tensor_std(peps) / maxbonddim(peps))
    else
        return peps
    end
end

function shift!(peps::AbstractPEPS, shift::Number)
    for ten in peps.tensors
        ten .+= shift 
    end
    return peps
end