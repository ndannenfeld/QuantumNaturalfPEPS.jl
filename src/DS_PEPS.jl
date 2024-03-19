module DS_PEPS
    using ITensors
    using Statistics

    include("PEPS.jl")

    export PEPS
    export flatten
    export get_logψ_and_envs
    export inner_peps
    export get_Ok
    export update_double_layer_envs!
    export get_sample
end