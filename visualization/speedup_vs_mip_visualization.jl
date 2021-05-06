###
# This script demonstrates how to optimize a linear objective over the range of a network
###
using PGFPlots

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the network
network_file = string(@__DIR__, "/../networks/CAS/ACASXU_experimental_v2a_1_1.nnet")
network = read_nnet(network_file)

# Define the coefficients for a linear objective
coeffs = [1.0; -1.0; 1.0; -1.0; 1.0]
radii = [0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064, 0.08, 0.10, 0.12, 0.13, 0.135, 0.14, 0.145, 0.15, 0.152, 0.155] #0.032, 0.064] #0.128, 0.14, 0.15]

# Use default parameters for the optimization then solve for each radius
params = PriorityOptimizerParameters()

times_priority = zeros(length(radii))
times_mip = zeros(length(radii))
for (i, radius) in enumerate(radii)
    input_set = Hyperrectangle(0.5*ones(5), radius*ones(5))
    times_priority[i] = @elapsed x_star, lower_bound, upper_bound, steps = optimize_linear(network, input_set, coeffs, params)
    times_mip[i] = @elapsed opt_val = NeuralPriorityOptimizer.mip_linear_value_only(network, input_set, coeffs, true; timeout=1500)
    println("Diff in optimal value: ", lower_bound - opt_val)
end

println("Times priority: ", times_priority)
println("Times mip: ", times_mip)
println("Speedups: ", times_mip ./ times_priority)

# Create the plots

# Both times on a plot vs. radius
plot = Axis(style="black", xlabel="Radius of input perturbation", ylabel="Time", title="Solve time vs. radius")
plot.legendStyle = "at={(1.05,1.0)}, anchor = north west"
push!(plot, Plots.Linear(radii, times_mip, legendentry="MIP", markSize=1.0))
push!(plot, Plots.Linear(radii, times_priority, legendentry="Priority Optimizer", markSize=1.0))
save("./visualization/plots/comparison_to_mip/times_vs_radius.pdf", plot)
save("./visualization/plots/comparison_to_mip/times_vs_radius.tex", plot)

# Speedups vs. radius 
plot = Axis(style="black", xlabel="Radius of input perturbation", ylabel="Speedup", title="Speedup vs. radius", ymode="log")
plot.legendStyle = "at={(1.05,1.0)}, anchor = north west"
push!(plot, Plots.Linear(radii, times_mip ./ times_priority, markSize=1.0))
save("./visualization/plots/comparison_to_mip/speedup_vs_radius.pdf", plot)
save("./visualization/plots/comparison_to_mip/speedup_vs_radius.tex", plot)

# Scatterplot of the two solvers
plot = Axis(style="black", axisEqual=true, xlabel="Time of Priority Optimizer", ylabel="Time of DeepZ + MIP", title="DeepZ + MIP vs. Priority Optimizer")
plot.legendStyle = "at={(1.05,1.0)}, anchor = north east"
push!(plot, Plots.Scatter(times_priority, times_mip, legendentry="DeepZ + MIP", markSize=1.0))
save("./visualization/plots/comparison_to_mip/mip_vs_priority_scatter.pdf", plot)
save("./visualization/plots/comparison_to_mip/mip_vs_priority_scatter.tex", plot)
