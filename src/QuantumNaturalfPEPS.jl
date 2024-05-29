module QuantumNaturalfPEPS

using ITensors
using Statistics
using QuantumNaturalGradient

include("PEPS.jl")

export PEPS
export flatten
export get_logψ_and_envs
export inner_peps
export get_Ok
export update_double_layer_envs!
export get_sample
export get_Ek
export write!
export Ok_and_Ek
export generate_Oks_and_Eks

end
