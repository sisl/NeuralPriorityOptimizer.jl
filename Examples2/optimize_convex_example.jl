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

# Setup your problem 
lbs = [0.0, 0.0, -0.93141591135858504, -0.928987424967730113]
ubs = [1.0, 1.0, -0.9, -0.9]
input_set = Hyperrectangle(low=lbs, high=ubs)
params = PriorityOptimizerParameters(max_steps=3000) # default parameters except max_steps

### First try with a linear function and compare it to the linear stand-alone
coeffs = [-0.74, -0.44]
obj_fcn = x -> coeffs' * x

# Print the results with the convex wrapper
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_convex_program(network, input_set, obj_fcn, params)
println("---Maximizing c^T y over the range---")
println("Elapsed time convex wrapper: ", time)
println("Interval convex wrapper: ", [lower_bound, upper_bound])
println("Steps convex wrapper: ", steps)
println()

# Now compare to the results from the linear stand-alone wrapper 
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)
println("Elapsed time linear wrapper: ", time)
println("Interval linear wrapper: ", [lower_bound, upper_bound])
println("Steps linear wrapper: ", steps)
println()

### Now try with a distance function from a point and compare it to the project_onto_range wrapper
point = [10.0, 20.0]
p = 2
obj_fcn = x -> norm(x - point, p)

# Print the results with the convex wrapper.
# NOTE: we must tell it to minimize the distance by setting maximize=false
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_convex_program(network, input_set, obj_fcn, params; maximize=false)
println("---Minimizing ||y - point||_p over the range---")
println("Elapsed time convex wrapper: ", time)
println("Interval convex wrapper: ", [lower_bound, upper_bound])
println("Steps convex wrapper: ", steps)
println()

# Now compare to the results from the linear stand-alone wrapper 
# this already solves a convex program at each step so we expect the runtime to be similar
time = @elapsed x_star, lower_bound, upper_bound, steps = project_onto_range(network, input_set, point, p, params)
println("Elapsed time project onto range wrapper: ", time)
println("Interval project onto range wrapper: ", [lower_bound, upper_bound])
println("Steps project onto range wrapper: ", steps)
println()

### Now show that it will work with another convex function that isn't captured by the other wrappers 
# for example minimizing the -log of the first output plus the square of the second or something like that 
obj_fcn = x -> -log(x[1]) + square(x[2])#-log(x[1]) + square(x[2])
time = @elapsed x_star, lower_bound, upper_bound, steps = optimize_convex_program(network, input_set, obj_fcn, params; maximize=false)
println("---Minimizing -log(y[1]) + y[2]^2 over the range---")
println("Elapsed time convex wrapper: ", time)
println("Interval convex wrapper: ", [lower_bound, upper_bound])
println("Steps convex wrapper: ", steps)
println()
