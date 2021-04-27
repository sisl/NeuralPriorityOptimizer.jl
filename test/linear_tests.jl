# Read in the network. Ground truth are from comparing this optimizer's first version, MIPVerify, and Marabou and seeing agreement. 
network_file = string(@__DIR__, "/../networks/GANControl/full_big_uniform.nnet")
network = read_nnet(network_file)

# Define the coefficients for a linear objective
coeffs = [-0.74; -0.44]

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters()

@testset "Linear objective tests" begin 
    # Test 1
    lbs = [-1.0, -1.0, -0.93141591135858504, -0.928987424967730113]
    ubs = [-0.8, -0.8, -0.9, -0.9]
    groundtruth_1 = 9.874803
    input_set = Hyperrectangle(low=lbs, high=ubs)
    x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)
    @test abs(lower_bound - groundtruth_1) <= params.stop_gap
    @test abs(upper_bound - groundtruth_1) <= params.stop_gap

    # Test 2
    lbs = [0.9, 0.9, -0.93141591135858504, -0.928987424967730113]
    ubs = [1.0, 1.0, -0.9, -0.9]
    groundtruth_2 = 12.098409
    input_set = Hyperrectangle(low=lbs, high=ubs)
    x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)
    @test abs(lower_bound - groundtruth_2) <= params.stop_gap
    @test abs(upper_bound - groundtruth_2) <= params.stop_gap

    # Test 3
    lbs = [0.7, 0.7, -0.93141591135858504, -0.928987424967730113]
    ubs = [1.0, 1.0, -0.9, -0.9]
    groundtruth_3 = 12.098409
    input_set = Hyperrectangle(low=lbs, high=ubs)
    x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)
    @test abs(lower_bound - groundtruth_3) <= params.stop_gap
    @test abs(upper_bound - groundtruth_3) <= params.stop_gap
end

