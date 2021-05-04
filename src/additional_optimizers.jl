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

function mip_linear_value_only(network, input_set::Union{Hyperrectangle, Zonotope}, coeffs, maximize)
    # Get your bounds
    bounds = NeuralVerification.get_bounds(Ai2z(), network, input_set; before_act=true)

    # Create your model
    model = Model(with_optimizer(Gurobi.Optimizer, OutputFlag=1, Threads=8, TimeLimit=300.0))
    z = init_vars(model, network, :z, with_input=true)
    δ = init_vars(model, network, :δ, binary=true)
    # get the pre-activation bounds:
    model[:bounds] = bounds
    model[:before_act] = true

    # Add the input constraint 
    NeuralVerification.add_set_constraint!(model, input_set, first(z))

    # Encode the network as an MIP
    encode_network!(model, network, BoundedMixedIntegerLP())
    if maximize 
        @objective(model, Max, coeffs'*last(z))
    else
        @objective(model, Min, coeffs'*last(z))
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
    else 
        @assert false "Non optimal result"
    end
end