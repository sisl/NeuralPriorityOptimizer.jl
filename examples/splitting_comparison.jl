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
ubs = [1.0, 1.0, -0.1, -0.1]
input_set = Hyperrectangle(low=lbs, high=ubs)
maximize = true

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters(stop_frequency=40, verbosity=0, max_steps=5000)
time_largest_interval = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params; maximize=maximize)
time_gradient = @elapsed x_star_gradient, lower_bound_gradient, upper_bound_gradient, steps_gradient = optimize_linear_gradient_split(network, input_set, coeffs, params; maximize=maximize)

# Print your results 
println()
println("Time priority: ", time_largest_interval)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)
println()

println("Time gradient split: ", time_gradient)
println("Interval: ", [lower_bound_gradient, upper_bound_gradient])
println("Steps: ", steps_gradient)