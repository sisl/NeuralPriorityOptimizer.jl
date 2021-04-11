# Solve the distance for an arbitrary p-norm 
function dist_to_zonotope_p(reach, point, p)
    G = reach.generators
    c = reach.center
    n, m = size(G)
    x = Variable(m)
    obj = norm(G * x + c - point, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0])
    solve!(prob, Mosek.Optimizer(LOG=0))
    @assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

# Check whether a point is included in the output of a network
# Right now this finds the negative distance of the point to the network:
#  inf_x,y ||y_0 - y||_p s.t. y = network(x), lb <= x <= ub. It will actually solve 
# max_x,y -||y_0 - y||_p s.t. y = network(x), lb <= x <= ub, so the return value is negative of what you ight expect   
function inclusion_wrapper(network, lbs, ubs, y₀, p; n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    evaluate_objective = x -> -norm(y₀ - NeuralVerification.compute_output(network, x), p)
    approximate_optimize_cell = cell -> -dist_to_zonotope_p(forward_network(solver, network, cell), y₀; p=p)
    return general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end

function linear_opt_wrapper(network, lbs, ubs, coeffs; n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits=0)
    evaluate_objective = x -> compute_objective(network, x, coeffs)
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    return general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = n_steps, solver=solver, early_stop=early_stop, stop_freq=stop_freq, stop_gap=stop_gap, initial_splits=initial_splits)
end