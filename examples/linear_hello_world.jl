###
# This script demonstrates how to optimize a linear objective over the range of a network
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the network
network_file = string(@__DIR__, "/../networks/GANControl/full_big_uniform.nnet")
network = read_nnet(network_file)

# Define the coefficients for a linear objective
coeffs = [-0.74; -0.44]
lbs = [0.4, 0.4, -0.93141591135858504, -0.928987424967730113]
ubs = [1.0, 1.0, -0.9, -0.9]
input_set = Hyperrectangle(low=lbs, high=ubs)

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters()
time_priority = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)

# Compare timing to a MIP solver which uses zonotope propagation to compute the bounds for its encoding 
time_mip = @elapsed opt_val = NeuralPriorityOptimizer.mip_linear_value_only(network, input_set, coeffs, true)

# Print your results 
println("Elapsed time with priority optimizer: ", time_priority)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)

println("With MIP with zonotope propagated bounds, value:  ", opt_val, " in time: ", time_mip)

println("Speedup: ", time_mip / time_priority)