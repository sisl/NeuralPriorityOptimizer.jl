###
# This is an example of how to find the maximum distance between 
# the output of two networks in a given region.
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the networks
network_file_1 = string(@__DIR__, "/../networks/GANControl/full_big_uniform.nnet")
network_file_2 = string(@__DIR__, "/../networks/GANControl/full_msle_uniform.nnet")
network_1 = read_nnet(network_file_1)
network_2 = read_nnet(network_file_2)

# Define your input region
lbs = [-1.0, -1.0, -0.93141591135858504, -0.928987424967730113]
ubs = [1.0, 1.0, -0.9, -0.9]
input_set = Hyperrectangle(low=lbs, high=ubs)

# Solve the problem using 20,000 steps maximum
params = PriorityOptimizerParameters(max_steps=200000, verbosity=1)
time = @elapsed x_star, lower_bound, upper_bound, steps = max_network_difference(network_1, network_2, input_set, params; p=1)

# Print the results 
println("Elapsed time: ", time)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)

