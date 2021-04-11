###
# This script demonstrates how to check whether the output of a network is contained within a polytope
# we use ACASXu property 1: see https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/properties/acas_property_1.txt
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

function test_acas_network(index1, index2, property_index)
    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network
    network_file = string(@__DIR__, "/../Networks/ACASXu/", network_name)
    network = read_nnet(network_file)

    # Define your input and output sets 
    input_set, output_set = get_acas_sets(property_index)

    # Solve the problem 
    params = PriorityOptimizerParameters(max_steps=100000, stop_frequency=40)
    if output_set isa HalfSpace || output_set isa AbstractPolytope
        println("Checking if contained within polytope")
        time = @elapsed x_star, lower_bound, upper_bound, steps = contained_within_polytope(network, input_set, output_set, params)
    elseif output_set isa PolytopeComplement
        time = @elapsed x_star, lower_bound, upper_bound, steps = reaches_polytope(network, input_set, output_set.P, params, p=2)
    else
        @assert false "Haven't implemented reach polytope yet"
    end
    # Print your results 
    println("Elapsed time: ", time)
    println("Interval: ", [lower_bound, upper_bound])
    println("Steps: ", steps)

    return lower_bound, upper_bound, time
end

full_time = @elapsed begin
bounds_and_time = Array{Any, 2}(undef, 5, 9)
property_index = 2
for i = 1:5
    for j = 1:9
        println("Network ", (i, j))
        bounds_and_time[i, j] = test_acas_network(i, j, property_index)
        println()
    end
end
end

for i = 1:5
    for j = 1:9
        println("Network: ", (i, j), "  property: ", property_index, "   bounds and time: ", bounds_and_time[i, j])
    end
end
println("Full time: ", full_time)
