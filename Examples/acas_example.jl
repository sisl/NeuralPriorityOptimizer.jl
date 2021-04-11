###
# This script demonstrates how to check whether the output of a network is contained within a polytope
# we use ACASXu property 1: see https://github.com/NeuralNetworkVerification/Marabou/blob/master/resources/properties/acas_property_1.txt
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

function test_acas_network(index1, index2, property_index; max_steps = 20000)
    network_name = string("ACASXU_experimental_v2a_", index1, "_", index2, ".nnet")
    # Read in the network
    network_file = string(@__DIR__, "/../Networks/ACASXu/", network_name)
    network = read_nnet(network_file)

    # Define your input and output sets 
    input_set, output_set = get_acas_sets(property_index)

    # Solve the problem 
    params = PriorityOptimizerParameters(max_steps=max_steps, stop_frequency=40)
    if output_set isa HalfSpace || output_set isa AbstractPolytope
        println("Checking if contained within polytope")
        time = @elapsed x_star, lower_bound, upper_bound, steps = contained_within_polytope(network, input_set, output_set, params)
    elseif output_set isa PolytopeComplement
        time = @elapsed x_star, lower_bound, upper_bound, steps = reaches_polytope(network, input_set, output_set.P, params, p=Inf)
    else
        @assert false "Haven't implemented reach polytope yet"
    end
    # Print your results 
    println("Elapsed time: ", time)
    println("Interval: ", [lower_bound, upper_bound])
    println("Steps: ", steps)

    return lower_bound, upper_bound, time, steps
end

max_steps = 10000
properties_to_test = 4

full_time = @elapsed begin
    lower_bounds = Array{Float64, 3}(undef, 4, 5, 9)
    upper_bounds = Array{Float64, 3}(undef, 4, 5, 9)
    times = Array{Float64, 3}(undef, 4, 5, 9)
    steps = Array{Integer, 3}(undef, 4, 5, 9)
    for property_index = 1:properties_to_test
        for i = 1:1
            for j = 1:9
                println("Network ", (i, j))
                lower_bounds[property_index, i, j], upper_bounds[property_index, i, j], times[property_index, i, j], steps[property_index, i, j] = test_acas_network(i, j, property_index; max_steps=max_steps)
                println()
            end
        end
    end
end


# Nicely formatted printout of the tests
for property_index = 1:properties_to_test
    for i = 1:1
        for j = 1:9
            println("Property: ", property_index, "   Network: ", (i, j))
            println("    bounds: ", (lower_bounds[property_index, i, j], upper_bounds[property_index, i, j]), "  time: ", times[property_index, i, j])
            if property_index == 1
                if lower_bounds[property_index, i, j] > 0
                    println("    SAT")
                elseif upper_bounds[property_index, i, j] <= NeuralVerification.TOL[]
                    println("    UNSAT")
                else
                    println("    Inconclusive")
                end
            end
            if property_index in [2, 3, 4]
                if upper_bounds[property_index, i, j] <= NeuralVerification.TOL[]
                    println("    SAT")
                elseif lower_bounds[property_index, i, j] > 0 
                    println("    UNSAT")
                else 
                    println("    Inconclusive")
                end
            end
        end
    end
end
println("Full time: ", full_time)
