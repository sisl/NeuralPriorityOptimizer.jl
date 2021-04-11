###
# This script demonstrates how to optimize a linear objective over the range of a network
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the network
network_file = string(@__DIR__, "/../Networks/GANControl/full_big_uniform.nnet")
network = read_nnet(network_file)

# Define the coefficients for a linear objective
coeffs = [-0.74; -0.44]
lbs = [0.7, 0.7, -0.93141591135858504, -0.928987424967730113]
ubs = [1.0, 1.0, -0.9, -0.9]
input_set = Hyperrectangle(low=lbs, high=ubs)

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters()
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)

# Print your results 
println("Elapsed time: ", time)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)


