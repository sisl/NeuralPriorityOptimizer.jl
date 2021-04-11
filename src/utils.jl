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