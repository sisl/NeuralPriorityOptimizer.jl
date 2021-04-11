elem_basis(i, n) = [k == i ? 1.0 : 0.0 for k in 1:n]

compute_objective(network, x, coeffs) = dot(coeffs, NeuralVerification.compute_output(network, x))