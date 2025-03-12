function convert_if_real(x::Complex)
    cond1 = -1e-10 < imag(x)/(abs(real(x) + 1e-9)) < 1e-10
    cond2 = -1e-14 < imag(x) < 1e-14
    if cond1 || cond2
        return real(x)
    end
    return x
end
convert_if_real(x::Real) = x