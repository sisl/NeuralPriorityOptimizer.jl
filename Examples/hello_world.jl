###
# This script demonstrates how to optimize a linear objective over the range of a network
###
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the network
network_file = string(@__DIR__, "/../Networks/GANControl/full_msle_uniform.nnet")
network = read_nnet(network_file)

# Define the coefficients for a linear objective
coeffs = [-0.74; -0.44]
lbs = [-1.0, -1.0, -1.0, -1.00]
ubs = [1.0, 1.0, -0.97, -0.97]
input_set = Hyperrectangle(low=lbs, high=ubs)

# Use default parameters for the optimization 
params = PriorityOptimizerParameters()
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)

# Print your results 
println("Elapsed time: ", time)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)


