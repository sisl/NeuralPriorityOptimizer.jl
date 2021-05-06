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
lbs = [-1.0, -1.0, -0.93141591135858504, -0.928987424967730113]
ubs = [1.0, 1.0, -0.9, -0.9]
input_set = Hyperrectangle(low=lbs, high=ubs)
maximize = true

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters()
time_priority = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params; maximize=maximize)

# Compare timing to a MIP solver which uses zonotope propagation to compute the bounds for its encoding 
#time_mip = @elapsed opt_val_mip = NeuralPriorityOptimizer.mip_linear_value_only(network, input_set, coeffs; maximize=maximize)

# Compare timing to solving MIPs in each cell after uniformly splitting 
splits_per_dim = [3, 3, 1, 1]
time_mip_split = @elapsed opt_val_split = NeuralPriorityOptimizer.mip_linear_split(network, input_set, coeffs, splits_per_dim; maximize=maximize)

# Print your results 
println("Elapsed time with priority optimizer: ", time_priority)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)

#println("With MIP with zonotope propagated bounds, value:  ", opt_val_mip, " in time: ", time_mip)
#println("Speedup: ", time_mip / time_priority)
println("With MIP uniform split, value: ", opt_val_split, " time: ", time_mip_split)
println("Speedup: ", time_mip_split / time_priority)
