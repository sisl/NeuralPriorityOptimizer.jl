# Use a priority based approach to split your space 
# General to any objective function passed in as well as an evaluate objective 
# The function approximate_optimize_cell takes in a cell and then does an approximate optimization over it 
# The function evaluate_objective takes in a point in the input space and evaluates the objective 
# This optimization strategy then uses these functions to provide bounds on the maximum objective
function general_priority_optimization(lbs, ubs, approximate_optimize_cell, evaluate_objective;  n_steps = 1000, solver=Ai2z(), early_stop=true, stop_freq=200, stop_gap=1e-4, initial_splits = 0)
    start_cell = Hyperrectangle(low=lbs, high=ubs)
    initial_cells = split_multiple_times(start_cell, initial_splits)
    println("Done with initial splits")
    # Create your queue, then add your original new_cells 
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
    [enqueue!(cells, cell, approximate_optimize_cell(cell)) for cell in initial_cells] # add with priority
    best_lower_bound = -Inf
    best_x = nothing

    # For n_steps dequeue a cell, split it, and then 
    for i = 1:n_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        dequeue!(cells)
        
        # Early stopping
        if early_stop
            if i % stop_freq == 0
                lower_bound = evaluate_objective(cell.center)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                println("i: ", i)
                println("lower bound: ", lower_bound)
                println("best lower bound: ", best_lower_bound)
                println("value: ", value)
                if (value .- lower_bound) <= stop_gap
                    return best_x, best_lower_bound, value, i
                end
                println("max radius: ", max(radius(cell)))
            end
        end

        new_cells = split_cell(cell)
        # Enqueue each of the new cells
        for new_cell in new_cells
            # If you've made the max objective cell tiny
            # break (otherwise we end up with zero radius cells)
            if max(radius(new_cell) < TOL[])
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                lower_bound = evaluate_objective(cell.center)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                return best_x, best_lower_bound, value, i 
            end
            new_value = approximate_optimize_cell(new_cell)
            enqueue!(cells, new_cell, new_value)
        end
    end
    # The largest value in our queue is the approximate optimum 
    cell, value = peek(cells)
    lower_bound = evaluate_objective(cell.center)
    if lower_bound > best_lower_bound
        best_lower_bound = lower_bound
        best_x = cell.center
    end
    return best_x, best_lower_bound, value, n_steps
end