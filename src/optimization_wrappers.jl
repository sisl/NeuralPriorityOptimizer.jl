
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
    achievable_value = cell -> cell.center, norm(y₀ - NeuralVerification.compute_output(network, cell.center), p)
    return general_priority_optimization(input_set, approximate_optimize_cell, achievable_value, params, false) # TODO: should there be early stopping here?
end

"""
    optimize_linear(network, input_set, coeffs, params; maximize=true)

Optimize a linear function on the output of a network. This returns the optimal input, lower bound on the objective value,
    upper bound on the objective value, and the number of steps that the solver took.  
"""
function optimize_linear(network, input_set, coeffs, params; maximize=true, solver=Ai2z())
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    achievable_value = cell -> (cell.center, compute_linear_objective(network, cell.center, coeffs))
    return general_priority_optimization(input_set, approximate_optimize_cell, achievable_value, params, maximize)
end

function optimize_linear_gradient_split(network, input_set::Hyperrectangle, coeffs, params; maximize=true, solver=Ai2z())
    approximate_optimize_cell = cell -> ρ(coeffs, forward_network(solver, network, cell))
    #achievable_value = cell -> (cell.center, compute_linear_objective(network, cell.center, coeffs))
    achievable_value = cell -> begin 
                                    grad = get_chained_gradient(network, cell.center, coeffs)
                                    x = σ(vec(grad), cell)
                                    return x, compute_linear_objective(network, x, coeffs)
                               end
    # split_cell = cell -> begin 
    #                         grad = get_chained_gradient(network, cell.center, coeffs)
    #                         priorities = abs.(vec(grad)) .* radius_hyperrectangle(cell)
    #                         return split_hyperrectangle(cell, argmax(priorities))
    #                      end
    return general_priority_optimization(input_set, approximate_optimize_cell, achievable_value, params, maximize)
end

"""
    optimize_convex_program(network, input_set, convex_fcn, evaluate_convex_fcn, params; maximize=true, solver=Ai2z())

If minimizing (maximize=false) minimize a convex function over the range of the network
If maximizing (maximize=true) maximize a concave function over the range of the network

Called optimize convex program since it can be either convex or concave depending on max. vs. min.

obj_fcn should map a list of variables the length of the dimension of the zonotope to 
a convex or concave expression (defined using Convex.jl disciplined convex programming 
building blocks) depending on whether minimizing or maximizing. It will also be used to evaluate 
the function 
"""
function optimize_convex_program(network, input_set, obj_fcn, params; maximize=true, solver=Ai2z())
    approximate_optimize_cell = cell -> convex_program_over_zonotope(forward_network(solver, network, cell), obj_fcn, maximize)
    # Get an achievable value by evaluating the convex function for the centerpoint in the cell 
    x = Variable(length(network.layers[end].bias)) # make your variable just once
    achievable_value = cell -> begin
        x.value = NeuralVerification.compute_output(network, cell.center)
        return cell.center, evaluate(obj_fcn(x))[1]
    end
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
    achievable_value = cell -> (cell.center, dist_polytope_point(A, b, NeuralVerification.compute_output(network, cell.center), p))
    return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false; bound_threshold_approximate=0.0) # if we ever show that we must have a distance > 0, then we know we can't reach the polytope 
end

function reaches_polytope_binary(network, input_set, polytope, params; solver=Ai2z(), eager=false)
    # first get a zonotope overapproximation of the polytope we'll use to filter out 
    # clear lack of intersections?
    # for now hyperrectangle overapprox? or how to quickly filter out ones that don't intersect? 
    
    # have it be 1 if no intersection, 0 if intersection, and try to minimize 
    # so an objective function that's 0 inside the polytope and 1 outside
    # TODO: add comment for change to non-eager 
    halfspaces = polytope.constraints
    A, b = tosimplehrep(polytope)
    underestimate_cell = cell -> begin
                                    reach = forward_network(solver, network, cell)
                                    if !quick_check_disjoint(reach, halfspaces)
                                        if eager
                                            return 0.0
                                        else 
                                            return convert(Int, check_disjoint(reach, A, b))
                                        end
                                    else 
                                        return 1.0
                                    end
                                 end
    achievable_value = cell -> (cell.center, convert(Int, !(compute_output(network, cell.center) ∈ polytope)))
    
    return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false; bound_threshold_approximate=0.5, bound_threshold_realizable=0.5) # if ever show distance lower bound is above 0.5 (will mean it's 1) then there's no intersection left, so return. If we ever get a concrete value below 0.5 it will mean it's 0, which means we actually reached it. 
end

function reaches_obtuse_polytope(network, input_set, polytope, params; solver=Ai2z())
    underestimate_cell = cell -> begin
				    # could do zonotope too I think but the # vertices grows as 2^num generators so that worries me
				    reach = overapproximate(forward_network(solver, network, cell), Hyperrectangle)
				    vertex_intersects = false
				    # if zonotope use vertices_list(reach; apply_convex_hull=false)
				    for vertex in vertices_list(reach)
                        if vertex in polytope
                            vertex_intersects = true
                        end
				    end
				    return vertex_intersects ? 0.0 : 1.0
				 end
    achievable_value = cell -> (cell.center, convert(Int, !(compute_output(network, cell.center) in polytope)))
    return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false; bound_threshold_approximate = 0.5, bound_threshold_realizable=0.5)
end

# Minimize the indicator function which is 1 when a matrix is PSD and 0 otherwise.
function range_is_psd(network, input_set, params; solver=Ai2z())
	underestimate_cell = cell -> begin 
                                    reach = overapproximate(forward_network(solver, network, cell), Hyperrectangle)
                                    for vertex in vertices_list(reach)
                                        if !isposdef!(reshape_vec_to_matrix(vertex))
                                            return 0.0
                                        end
                                    end
                                    return 1.0
					             end
	achievable_value = cell -> (cell.center, convert(Int, isposdef(reshape_vec_to_matrix(compute_output(network, cell.center)))))
	return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false; bound_threshold_approximate = 0.5, bound_threshold_realizable=0.5)
end

"""
    distance_to_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)

Projects the range of a network onto a polytope. This is the same as reaches_polytope except it won't 
stop early if it proves that the distance of the projection is > 0. 

Note that if you want the point on the polytope closest to the range of the network 
you would have to take the optimal input which you get back from calling this function,
pass it through the network, then project that point 
onto the polytope by solving a linear program min._y ||y_opt - y|| s.t. Ay <= b. where y_opt is 
the output of the network closest to the polytope. 
"""
function distance_to_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)
    A, b = tosimplehrep(polytope)
    underestimate_cell = cell -> dist_zonotope_polytope(forward_network(solver, network, cell), A, b, p)
    achievable_value = cell -> (cell.center, dist_polytope_point(A, b, NeuralVerification.compute_output(network, cell.center), p))
    return general_priority_optimization(input_set, underestimate_cell, achievable_value, params, false) # if we ever show that we must have a distance > 0, then we know we can't reach the polytope 
end

"""
    contained_within_polytope(network, input_set, polytope, params; solver=Ai2z())

Checks whether the output of the network is contained with a polytope. 
"""
function contained_within_polytope(network, input_set, polytope, params; solver=Ai2z())
    A, b = tosimplehrep(polytope)
    overestimate_cell = cell -> max_polytope_violation(forward_network(solver, network, cell), A, b)
    achievable_value = cell -> (cell.center, max_polytope_violation(NeuralVerification.compute_output(network, cell.center), A, b))
    return general_priority_optimization(input_set, overestimate_cell, achievable_value, params, true; bound_threshold_realizable=0.0) # if we ever find a concrete value > 0 then return
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
    achievable_value = cell -> (cell.center, norm(NeuralVerification.compute_output(network1, cell.center) - NeuralVerification.compute_output(network2, cell.center), p))

    return general_priority_optimization(input_set, overestimate_cell, achievable_value, params, true)
end

"""
    optimize_linear_with_buffer(network, input_set, coeffs, buffer, params; maximize=true, solver=Ai2z())

solve the optimization problem: 
min. c^T y 
s.t. y = network_two(network_one(x) + z))
     x in input_set 
     z in buffer 

This adds in the "buffer" to the output manifold of network_one, 
and then optimizes a linear function over the resulting output manifold 
of network_two. It uses a mip solver for the second network 
in case the output of the first network is high dimensional. 
"""
function optimize_linear_with_buffer(network_one, network_two, input_set, coeffs, buffer, params; maximize=true, solver=Ai2z())
    overestimate_cell = cell -> begin                                               
                                    reach_one = forward_network(solver, network_one, cell)
                                    buffered_reach = concretize(reach_one ⊕ buffer)
                                    opt_val = mip_linear_value_only(network_two, buffered_reach, coeffs; maximize=maximize)
                                    return opt_val
                                end
    achievable_value = cell ->  begin
                                    out_1 = compute_output(network_one, cell.center)
                                    buffered_output = translate(buffer, out_1)
                                    opt_val = mip_linear_value_only(network_two, buffered_output, coeffs; maximize=maximize)
                                    return cell.center, opt_val
                                end

    return general_priority_optimization(input_set, overestimate_cell, achievable_value, params, maximize)
end
