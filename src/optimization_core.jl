""" 
    struct PriorityOptimizerParameters
Define a struct which holds all the parameters for the priority optimizer 
  steps: the maximum number of steps to take before returning the best bounds so far
  early_stop: whether to use evaluate_objective occasionally to narrow your optimality gap and potentially return early.
  stop_frequency: how often to check if you should return early
  stop_gap: optimality gap that you need to be beneath in order to return early 
  initial_splits: a number of times to split the original hyperrectangle before doing any analysis. 
"""
@with_kw struct PriorityOptimizerParameters
    max_steps::Int = 1000
    early_stop::Bool = true
    stop_frequency::Int = 200
    stop_gap::Float64 = 1e-4
    initial_splits::Int = 0
end

"""
    general_priority_optimization(start_cell::Hyperrectangle, approximate_optimize_cell, achievable_value, params::PriorityOptimizerParameters)

Use a priority based approach to split your space and optimize an objective function. We assume we are maximizing our objective. 
General to any objective function passed in as well as an evaluate objective 
The function overestimate_cell takes in a cell and returns an overestimate of the objective value. 
The function achievable_value takes in the input cell and finds an achievable objective value in that cell. 
This optimization strategy then uses these functions to provide bounds on the maximum objective

This function returns the best input found, a lower bound on the optimal value, an upper bound on the optimal value, and the number of steps taken.
"""
function general_priority_optimization(start_cell::Hyperrectangle, overestimate_cell, achievable_value, params::PriorityOptimizerParameters)
    initial_cells = split_multiple_times(start_cell, params.initial_splits)
    println("Done with initial splits")
    # Create your queue, then add your original new_cells 
    cells = PriorityQueue(Base.Order.Reverse) # pop off largest first 
    [enqueue!(cells, cell, overestimate_cell(cell)) for cell in initial_cells] # add with priority
    best_lower_bound = -Inf
    best_x = nothing

    # For n_steps dequeue a cell, split it, and then 
    for i = 1:params.max_steps
        cell, value = peek(cells) # peek instead of dequeue to get value, is there a better way?
        @assert value >= best_lower_bound "Our lowest upper bound must be greater than the highest achieved value"
        dequeue!(cells)
        
        # Early stopping
        if params.early_stop
            if i % params.stop_frequency == 0
                lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                println("i: ", i)
                println("lower bound: ", lower_bound)
                println("best lower bound: ", best_lower_bound)
                println("value: ", value)
                if (value .- lower_bound) <= params.stop_gap
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
            if max(radius(new_cell) < NeuralVerification.TOL[])
                # Return a concrete value and the upper bound from the parent cell
                # that was just dequeued, as it must have higher value than all other cells
                # that were on the queue, and they constitute a tiling of the space
                lower_bound = achievable_value(cell)
                if lower_bound > best_lower_bound
                    best_lower_bound = lower_bound
                    best_x = cell.center
                end
                return best_x, best_lower_bound, value, i 
            end
            new_value = overestimate_cell(new_cell)
            enqueue!(cells, new_cell, new_value)
        end
    end
    # The largest value in our queue is the approximate optimum 
    cell, value = peek(cells)
    lower_bound = achievable_value(cell)
    if lower_bound > best_lower_bound
        best_lower_bound = lower_bound
        best_x = cell.center
    end
    return best_x, best_lower_bound, value, params.max_steps
end

"""
    general_priority_optimization(start_cell::Hyperrectangle, overestimate_cell, evaluate_objective, params::PriorityOptimizerParameters; maximize=true)

Wrapper to the general priority optimization so we can handle maximization and minimization.  
TODO: is this way of wrapping the functions inefficient?

If maximize is true, then relaxed_optimize_cell must map a cell to an overestimate of the objective value in that cell.
If maximize is false (so we are minimizing), then relaxed_optimize_cell must map a cell to an underestimate of the objective value in that cell. 
In both cases, evaluate_objective must map from a cell to an achievable objective value within that cell. 

This function returns the best input found, a lower bound on the optimal value, an upper bound on the optimal value, and the number of steps taken.
"""
function general_priority_optimization(start_cell::Hyperrectangle, relaxed_optimize_cell, evaluate_objective, params::PriorityOptimizerParameters, maximize)
    if maximize
        return general_priority_optimization(start_cell, relaxed_optimize_cell, evaluate_objective, params)
    else 
        overestimate_cell = cell -> -relaxed_optimize_cell(cell)
        neg_evaluate_objective = cell -> -evaluate_objective(cell)
        x, lower, upper, steps = general_priority_optimization(start_cell, overestimate_cell, neg_evaluate_objective, params)
        return x, -upper, -lower, steps
    end
end