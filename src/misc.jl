function convert_if_real(x::Complex)
    if -1e-10 < imag(x)/(abs(real(x) + 1e-9)) < 1e-10
       return real(x) 
    end
    return x
end
convert_if_real(x::Real) = x