###
# This is an example of how to find the maximum distance between 
# the output of two networks in a given region.
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the networks
network_file_1 = string(@__DIR__, "/../networks/runway_generators/128x2mlp_generator.nnet")
network_file_2 = string(@__DIR__, "/../networks/runway_generators/256x4mlp_generator.nnet")
network_1 = read_nnet(network_file_1)
network_2 = read_nnet(network_file_2)

myrange = -1.73:0.1:1.73
N = length(myrange)
ub = zeros(N, N)
# Define your input region
for i=1:N
    for j=1:N
        state_lb = myrange[[i,j]]
        lbs = [-1.0, -1.0, state_lb...]
        ubs = [1.0, 1.0, (state_lb.+0.1)...]
        input_set = Hyperrectangle(low=lbs, high=ubs)
        println("====================================================, i=$i, j=$j, lbs: $lbs, ubs: $ubs")

        # Solve the problem using 20,000 steps maximum
        params = PriorityOptimizerParameters(max_steps=200000, verbosity=0, stop_gap=0.1)
        time = @elapsed x_star, lower_bound, upper_bound, steps = max_network_difference(network_1, network_2, input_set, params; p=1)
        ub[i,j] = upper_bound
    end
end

using BSON

BSON.@save "ub.bson" ub

ub =BSON.load("ub.bson")[:ub]
using Plots
pgfplotsx()
p = heatmap((10 / 1.73).*myrange, (30/1.73).*myrange, ub, xlabel="\$x\$ (meters)", ylabel="\$\\theta\$ (degrees)", title="Maximum Output Distance")
savefig("maximum_output_distance.tex")
Plots.pgfx_preamble()

using CSV
using DataFrames
CSV.write("ub.csv", DataFrame(ub), writeheader=false)

# Print the results 
println("Elapsed time: ", time)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)

