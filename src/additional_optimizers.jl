using NeuralVerification: init_vars, BoundedMixedIntegerLP, encode_network!, _ẑᵢ₊₁, TOL

function fgsm(network, x, lbs, ubs, coeffs; step_size=0.1)
    grad_full = NeuralVerification.get_gradient(network, x)
    grad = grad_full' * coeffs
    return clamp.(x + grad * step_size, lbs, ubs)
end

function pgd(network, x, lbs, ubs, coeffs; step_size=0.01, iterations=600)
    cur_x = x
    for i = 1:iterations
        cur_x = fgsm(network, cur_x, lbs, ubs, coeffs; step_size=step_size)
    end
    return cur_x
end

function repeated_pgd(network, x, lbs, ubs, coeffs; step_size=0.01, pgd_iterations=25, samples=50)
    best_x = x
    best_y = compute_objective(network, x, coeffs)
    for i = 1:samples
        rand_x = rand(length(x)) .* (ubs - lbs) .+ lbs
        cur_x = pgd(network, rand_x, lbs, ubs, coeffs; step_size=step_size, iterations=pgd_iterations)
        y = compute_objective(network, cur_x, coeffs)
        if y > best_y
            best_x, best_y = cur_x, y
        end
    end
    return best_x
end

function hookes_jeeves(network, x, lbs, ubs, coeffs, α, ϵ, γ=0.5)
    f(in) = compute_objective(network, in, coeffs)
    y, n = f(x), length(x)
    while α > ϵ
        improved = false 
        x_best, y_best = x, y
        for i in 1:n 
            for sgn in (-1, 1)
                x_prime = clamp.(x + sgn*α*elem_basis(i, n), lbs, ubs)
                y_prime = f(x_prime)
                if y_prime > y_best
                    x_best, y_best, improved = x_prime, y_prime, true
                end
            end
        end
        x, y = x_best, y_best
        
        if !improved 
            α *= γ
        end
    end
    return x
end

"""

objective_threshold gives a value above which 
"""
function mip_linear_value_only(network, input_set::Union{Hyperrectangle, Zonotope}, coeffs; maximize=true, obj_threshold=(maximize ? -Inf : Inf), timeout=1500.0, outputflag=0, threads=1)
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=outputflag, Threads=threads, TimeLimit=timeout))
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = bounds
    model[:before_act] = true

    # Add the input constraint 
    NeuralVerification.add_set_constraint!(model, input_set, first(z))

    # Encode the network as an MIP
    encode_network!(model, network, BoundedMixedIntegerLP())
    obj = coeffs'*last(z)
    if maximize 
        @objective(model, Max, obj)
        if obj_threshold != -Inf
            @constraint(model, obj >= obj_threshold)
        end
    else
        @objective(model, Min, obj)
        if obj_threshold != Inf 
            @constraint(model, obj >= obj_threshold)
        end
    end

    # Set lower and upper bounds 
    #for the first layer it's special because it has no ẑ
    set_lower_bound.(z[1], low(bounds[1]))
    set_upper_bound.(z[1], high(bounds[1]))
    for i = 2:length(z)-1
        # Set lower and upper bounds for the intermediate layers
        ẑ_i =  _ẑᵢ₊₁(model, i-1)
        z_i = z[i]
        # @constraint(model, ẑ_i .>= low(bounds[i])) These empirically seem to slow it down?
        # @constraint(model, ẑ_i .<= high(bounds[i]))
        z_low = max.(low(bounds[i]), 0.0)
        z_high = max.(high(bounds[i]), 0.0)
        set_lower_bound.(z_i, z_low)
        set_upper_bound.(model[:z][i], z_high)
    end
    # Set lower and upper bounds for the last layer special because 
    # it has no ReLU
    set_lower_bound.(z[end], low(bounds[end]))
    set_upper_bound.(z[end], high(bounds[end]))
    
    optimize!(model)
    if termination_status(model) == OPTIMAL
        return objective_value(model)
    elseif termination_status(model) == INFEASIBLE
        @warn "Infeasible result, did you have an output threshold? If not, then it should never return infeasible"
        return maximize ? -Inf : Inf  
    else
        @assert false "Non optimal result"
    end
end

"""
    cell_to_subcell(cell, index, cells_per_dim)
Get a sub-cell from a hyperrectangular cell given an index. So if we were in two dimensions 
with a cell with low = [-1, -1] and high = [1, 1], and cells_per_dim = [2, 2]
and index = [1, 1] we would get low=[-1.0, -1.0], high=[0.0, 0.0], the bottom left quadrant.
"""
function cell_to_subcell(cell, index, cells_per_dim)
    lbs, ubs = low(cell), high(cell)
    subcell_lbs = (ubs .- lbs) .* ((index .- 1) ./ cells_per_dim) .+ lbs
    subcell_ubs = (ubs .- lbs) .* (index ./ cells_per_dim) .+ lbs
    return Hyperrectangle(low=subcell_lbs, high=subcell_ubs)
end

function cell_to_all_subcells(cell, cells_per_dim)
    cells = Array{Hyperrectangle}(undef, cells_per_dim...)
    for index in CartesianIndices(Tuple(cells_per_dim))
        index_as_vector = [index[i] for i =1:length(index)]
        cells[index] = cell_to_subcell(cell, index_as_vector, cells_per_dim) 
    end
    return cells
end

"""
    mip_linear_split(network, input_set, coeffs, maximize, cells_per_dim)

Solve a linear optimization problem by splitting the space into cells and 
solving a MIP on each cell 
splits gives the number of splits in each dimension
"""
function mip_linear_split(network, input_set, coeffs, cells_per_dim; maximize=true, threads=8)
    indices = CartesianIndices(Tuple(cells_per_dim))
    optima = (maximize == true ? -Inf : Inf) * ones(cells_per_dim...)
    for index in indices 
        best_so_far = maximize ? maximum(optima) : minimum(optima)
        println("Starting query for index: ", index)
        println("best so far: ", best_so_far)
        index_as_vector = [index[i] for i =1:length(index)]
        subcell = cell_to_subcell(input_set, index_as_vector, cells_per_dim)
        optima[index] = mip_linear_value_only(network, subcell, coeffs; maximize=maximize, obj_threshold=best_so_far, threads=threads)   
    end
    return maximize ? maximum(optima) : minimum(optima)
end