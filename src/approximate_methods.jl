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