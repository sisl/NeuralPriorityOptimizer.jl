using NeuralPriorityOptimizer
using NeuralPriorityOptimizer: cell_to_all_subcells
using NeuralVerification
using LazySets
using LinearAlgebra
using PGFPlots

# Set to 1 thread 
threads = 1
@assert Threads.nthreads()==threads "Must be 1 thread"
LinearAlgebra.BLAS.set_num_threads(threads)

function time_and_plot(splits_per_dim)
    # Read in the network
    network_file = string(@__DIR__, "/../networks/GANControl/full_big_uniform.nnet")
    network = read_nnet(network_file)

    # Define the coefficients for a linear objective
    coeffs = [-0.74; -0.44]
    maximize = true

    # We'll verify an eighth of the state space
    lbs = [-0.8, -0.8, -1.0, -1.0]
    ubs = [0.8, 0.8, -0.94, -0.94]
    cells_per_dim = [1, 1, 2, 2]
    full_input_region = Hyperrectangle(low=lbs, high=ubs)
    cells = cell_to_all_subcells(full_input_region, cells_per_dim)

    # Parameters for the solvers 
    params = PriorityOptimizerParameters(stop_gap=1e-4, stop_frequency=100, max_steps=200000) # for the priority solver 
    # splits_per_dim = [10, 10, 1, 1] # defined as an argument passed into the function 


    # Now, time the solvers for each cell
    times_priority = zeros(cells_per_dim...)
    times_mip_split = zeros(cells_per_dim...)
    for i in CartesianIndices(cells)
        cell = cells[i]
        # Run the priority optimizer then the mip splitting optimizer
        time_priority = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, cell, coeffs, params; maximize=maximize)
        time_mip_split = @elapsed opt_val_uniform = NeuralPriorityOptimizer.mip_linear_split(network, cell, coeffs, splits_per_dim; maximize=maximize, threads=threads)
        
        println("steps: ", steps)
        println("lower, upper: ", lower_bound, upper_bound)
        println("opt priority: ", lower_bound, " time: ", time_priority)
        println("opt mip split: ", opt_val_uniform, " time: ", time_mip_split)
        @assert abs(opt_val_uniform - lower_bound < 2*params.stop_gap) "Disagreement in the optimal value"
        times_priority[i] = time_priority 
        times_mip_split[i] = time_mip_split 
    end

    # Make a scatterplot comparing the two 
    times_priority = vec(times_priority)
    times_mip_split = vec(times_mip_split)
    plot = Axis(style="black, width=19cm, height=12cm", xlabel="Time Priority Optimizer (s)", ylabel="Time DeepZ + MIP + Splitting (s)", title="Timing comparison on linear optimization problems", axisEqual=true)
    push!(plot, Plots.Scatter(times_priority, times_mip_split))

    save(string("./visualization/plots/comparison_to_mip_split/linear_optimization_vs_uniform_split_", string(splits_per_dim), ".pdf"), plot)
    save(string("./visualization/plots/comparison_to_mip_split/linear_optimization_vs_uniform_split_", string(splits_per_dim), ".pdf"), plot)

    return times_priority, times_mip_split 
end
times_priority_1, times_mip_split_3_3 = time_and_plot([3, 3, 1, 1])
times_priority_2, times_mip_split_5_5 = time_and_plot([5, 5, 1, 1])
times_priority_3, times_mip_split_10_10 = time_and_plot([10, 10, 1, 1])
times_priority_4, times_mip_split_15_15 = time_and_plot([15, 15, 1, 1])

# Now do a cactus plot with these 
include(string(@__DIR__, "/cactus_plot.jl"))
times = [times_mip_split_3_3, times_mip_split_5_5, times_mip_split_10_10, times_mip_split_15_15, times_priority_1]
labels = ["MIP Split 3x3", "MIP Split 5x5", "MIP Split 10x10", "MIP Split 15x15", "Priority Optimizer (ours)"]
styles = ["mark=o, orange", "mark=+, blue", "mark=triangle, red", "mark=square, black", "mark=diamond,teal"]
max_time = 3000000.0
output_file = string(@__DIR__, "/plots/comparison_to_mip_split/splitting_cactus")
clean_and_cactus_plot(times, labels, styles, max_time, output_file; title="MIP Splitting vs. Priority Optimizer")

