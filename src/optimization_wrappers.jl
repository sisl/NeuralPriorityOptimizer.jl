"""
    project_onto_range(network, input_set, y₀, p, params)

Project a point (with the projection defined by the p norm) y₀ onto the range of a network given an input_set. We frame this as 
solving the optimization problem
inf_x,y ||y_0 - y||_p s.t. y = network(x), x in input_set. We return the optimal 
x, lower bound on the objective value, upper bound on the objective value, and the number of steps 
that the solver took. 
"""
function project_onto_range(network, input_set, y₀, p, params; solver=Ai2z())
    approximate_optimize_cell = cell -> dist_zonotope_point(forward_network(solver, network, cell), y₀, p)
    achievable_value = cell -> norm(y₀ - NeuralVerification.compute_output(network, cell.center), p)
    return general_priority_optimization(input_set, approximate_optimize_cell, achievable_value, params, false)
end

"""
    optimize_linear(network, input_set, coeffs, params; maximize=true)

Optimize a linear function on the output of a network. This returns the optimal input, lower bound on the objective value,
    upper bound on the objective value, and the number of steps that the solver took.  
"""
function optimize_linear(network, input_set, coeffs, params; maximize=true, solver=Ai2z())
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    achievable_value = cell -> compute_linear_objective(network, cell.center, coeffs)
    return general_priority_optimization(input_set, approximate_optimize_cell, achievable_value, params, maximize)
end

"""
    reaches_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)

Checks whether a polytope is reachable by finding the distance between the range and the polytope. 
This is minimizing a function which is 0 in the polytope and equal to the minimum p-norm distance 
to the polytope outside of the polytope. 
"""
function reaches_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)
    A, b = tosimplehrep(polytope)
    underestimate_cell = cell -> dist_zonotope_polytope(forward_network(solver, network, cell), A, b, p)
    achievable_value = cell -> dist_polytope_point(A, b, NeuralVerification.compute_output(network, cell.center), p)
    return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false, 0.0) # 0.0 is the upper_bound_threshold
end

"""
    contained_within_polytope(network, input_set, polytope, params; solver=Ai2z())

Checks whether the output of the network is contained with a polytope. 
"""
function contained_within_polytope(network, input_set, polytope, params; solver=Ai2z())
    A, b = tosimplehrep(polytope)
    overestimate_cell = cell -> max_polytope_violation(forward_network(solver, network, cell), A, b)
    achievable_value = cell -> max_polytope_violation(NeuralVerification.compute_output(network, cell.center), A, b)
    return general_priority_optimization(input_set, overestimate_cell, achievable_value, params, true, -Inf)
end

"""
    max_network_difference(network1, network2, input_set; solver=Ai2z(), p=2)

Find the maximum distance (under a p-norm) between the output of two networks over the input set. 
"""
function max_network_difference(network1, network2, input_set, params; solver=Ai2z(), p=2)
    # An overapproximation of the maximum distance using hyperrectangles since 
    # we can analytically find that maximum distance. 
    overestimate_cell = cell -> 
    begin
        hyperrectangle_reach_1 = overapproximate(forward_network(solver, network1, cell), Hyperrectangle)
        hyperrectangle_reach_2 = overapproximate(forward_network(solver, network2, cell), Hyperrectangle)
        return max_dist(hyperrectangle_reach_1, hyperrectangle_reach_2, p)
    end

    # The distance between the output from each network at the center of the cell 
    achievable_value = cell -> norm(NeuralVerification.compute_output(network1, cell.center) - NeuralVerification.compute_output(network2, cell.center))

    return general_priority_optimization(input_set, overestimate_cell, achievable_value, params, true, -Inf)
end