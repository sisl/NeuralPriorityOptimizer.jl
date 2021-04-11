"""
    elem_basis(i, n)
    
Returns a vector corresponding to the ith elementary basis in dimension n. 
"""
elem_basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

"""
    compute_linear_objective(network, x, coeffs)

Helper function to compute a linear objective given a network, input, and coefficients. This function 
    just passes the input through the network then dots the output with the coefficients.
"""
compute_linear_objective(network, x, coeffs) = dot(coeffs, NeuralVerification.compute_output(network, x))

"""
    split_cell(cell::Hyperrectangle)
Split a hyperrectangle into multiple hyperrectangles. We currently pick the largest dimension 
and split along that. 
"""

function split_cell(cell::Hyperrectangle)
    lbs, ubs = low(cell), high(cell)
    largest_dimension = argmax(ubs .- lbs)
    # have a vector [0, 0, ..., 1/2 largest gap at largest dimension, 0, 0, ..., 0]
    delta = elem_basis(largest_dimension, length(lbs)) * 0.5 * (ubs[largest_dimension] - lbs[largest_dimension])
    cell_one = Hyperrectangle(low=lbs, high=(ubs .- delta))
    cell_two = Hyperrectangle(low=(lbs .+ delta), high=ubs)
    return [cell_one, cell_two]
end

"""
    split_multiple_times(cell, n)

Helper function to split a cell multiple times. It applies split_cell n times 
    resulting in a queue with n+1 cells. The first cell to be split is the last to be split again 
    since we use a queue (so we don't just repeatedly split the same cell).
"""
function split_multiple_times(cell::Hyperrectangle, n)
    q = Queue{Hyperrectangle}()
    enqueue!(q, cell)
    for i = 1:n
        new_cells = split_cell(dequeue!(q))
        enqueue!(q, new_cells[1])
        enqueue!(q, new_cells[2])
    end
    return q
end

"""
    dist_to_zonotope_p(zonotope::Zonotope, point, p)

    A helper function which finds the distance for an arbitrary p-norm norm between a 
        zonotope and a point. This is defined as 
    inf_x ||x - point||_p s.t. x in zonotope
"""
function dist_zonotope_point(zonotope::Zonotope, point, p)
    G = zonotope.generators
    c = zonotope.center
    n, m = size(G)
    x = Variable(m)
    obj = norm(G * x + c - point, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0])
    solve!(prob, Mosek.Optimizer(LOG=0))
    @assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

"""
    dist_to_zonotope_p(zonotope::Zonotope, polytope, p)

    A helper function which finds the distance for an arbitrary p-norm norm between a 
        zonotope and a polytope. This is defined as 
    inf_x,y ||x - y||_p s.t. x in zonotope and y in polytope 
"""
function dist_zonotope_polytope(zonotope::Zonotope, A, b, p)
    G = zonotope.generators
    c = zonotope.center
    n, m = size(G)
    x = Variable(m) # points in the zonotope
    y = Variable(size(A, 2)) # points in the polytope 
    obj = norm(G * x + c - y, p)
    prob = minimize(obj, [x <= 1.0, x >= -1.0, A*y <= b])
    solve!(prob, Mosek.Optimizer(LOG=0))
    @assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval <= NeuralVerification.TOL[] ? 0.0 : prob.optval
end

"""
    dist_polytope_point(A, b, point, p)

    A helper function which finds the distance for an arbitrary p-norm norm between a 
        polytope and a point. This is defined as 
    inf_x ||x - point||_p s.t. x in polytope
"""
function dist_polytope_point(A, b, point, p)
    x = Variable(size(A, 2))
    obj = norm(x - point, p)
    prob = minimize(obj, [A * x <= b])
    solve!(prob, Mosek.Optimizer(LOG=0))
    @assert prob.status == OPTIMAL "Solve must result in optimal status"
    return prob.optval
end

"""
    max_polytope_violation(zonotope::Zonotope, polytope)

Compute the maximum single violation of the linear constraints describing a polytope 
over a zonotope. Imagine the polytope is described by Ax <= b. We maximize a_i^T x - b for
x the zonotope where a_i is the ith row of A. By taking the maximum of this with 0, we get 
the possible violation of each constraint. If the max polytope violation is > 0, then some constraint 
can be violated, which means that the zonotope is not contained within the polytope. Otherwise, the 
zonotope is contained within the polytope since none of the polytope constraints can be violated. 
"""
function max_polytope_violation(zonotope::Zonotope, A, b)
    max_violation = -Inf
    for i = 1:size(A, 1)
        cur_violation = max(ρ(A[i, :], zonotope) - b[i], 0)
        if cur_violation > max_violation
            max_violation = cur_violation # update if need be
        end
    end
    return max_violation
end

"""
    max_polytope_violation(point::Vector{Float64}, A, b)

Compute the maximum violation of the constraints for a polytope described by Ax <= b. 
"""
max_polytope_violation(point::Vector{Float64}, A, b) = max(maximum(A * point - b), 0.0)

"""
    get_acas_sets(property_number)

Get the input and output sets for acas under the standard definition of a problem 
    as trying to show x in X implies y in Y. This returns the input and output sets X, Y.
    Taken from https://github.com/NeuralNetworkVerification/Marabou/tree/master/resources/properties

"""
function get_acas_sets(property_number)
    if property_number == 1
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = HalfSpace([1.0, 0.0, 0.0, 0.0, 0.0], 3.9911256459)
    elseif property_number == 2
        input_set = Hyperrectangle(low=[0.6, -0.5, -0.5, 0.45, -0.5], high=[0.6798577687, 0.5, 0.5, 0.5, -0.45])
        output_set = PolytopeComplement(HPolytope([-1.0 1.0 0.0 0.0 0.0; -1.0 0.0 1.0 0.0 0.0; -1.0 0.0 0.0 1.0 0.0; -1.0 0.0 0.0 0.0 1.0], [0.0; 0.0; 0.0; 0.0]))
    elseif property_number == 3
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.4933803236, 0.3, 0.3], high=[-0.2985528119, 0.0095492966, 0.5, 0.5, 0.5])
        output_set = PolytopeComplement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))

    elseif property_number == 4
        input_set = Hyperrectangle(low=[-0.3035311561, -0.0095492966, 0.0, 0.3181818182, 0.0833333333], high=[-0.2985528119, 0.0095492966, 0.0, 0.5, 0.1666666667])
        output_set = PolytopeComplement(HPolytope([1.0 -1.0 0.0 0.0 0.0; 1.0 0.0 -1.0 0.0 0.0; 1.0 0.0 0.0 -1.0 0.0; 1.0 0.0 0.0 0.0 -1.0], [0.0; 0.0; 0.0; 0.0]))
    else
        @assert false "Unsupported property number"
    end 

    return input_set, output_set
end