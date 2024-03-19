include("../src/DS_PEPS.jl")
using .DS_PEPS
using ITensors

peps = PEPS(2,2,2,3)
update_double_layer_envs!(peps)

N = 5000
S_hist = zeros(16)
pc_hist = zeros(16)
for i in 1:N
    S,pc,env_top = get_sample(peps)
    S_hist[parse(Int, join(string.(reshape(S, :))); base=2) + 1] += 1/N
    pc_hist[parse(Int, join(string.(reshape(S, :))); base=2) + 1] = pc
end

S_real = zeros(16)
for i in 0:15
    BinStr = string(i,base = 2, pad = 4)
    S = zeros(4)
    for j in 1:4
        S[j] = parse(Int,BinStr[j])
    end
    S = reshape(S, 2,2) 
    out, et, ed = get_logψ_and_envs(peps, Int.(S))
    S_real[i+1] = out^2
end
S_real /= norm(S_real, 1);

using Plots
scatter(1:16, S_real, label="real")
plot!(1:16, S_hist, label="sample")

println(S_real)
println(S_hist)