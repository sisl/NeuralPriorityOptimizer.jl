"""
    project_onto_range(network, input_set, y₀, p, params)

Project a point (with the projection defined by the p norm) y₀ onto the range of a network given an input_set. We frame this as 
solving the optimization problem
inf_x,y ||y_0 - y||_p s.t. y = network(x), x in input_set. We return the optimal 
x, lower bound on the objective value, upper bound on the objective value, and the number of steps 
that the solver took. 
"""
function project_onto_range(network, input_set, y₀, p, params; solver=Ai2z())
    approximate_optimize_cell = cell -> dist_to_zonotope_p(forward_network(solver, network, cell), y₀, p)
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