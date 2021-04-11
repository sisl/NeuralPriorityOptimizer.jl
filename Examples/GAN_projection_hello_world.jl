###
# This script demonstrates how to project a point onto the range of a network. 
# It demonstrates this using a GAN, so this is asking the question "what is the cloest 
# image my GAN can produce to a given image with the distance defined by some p-norm".
###

# Include the optimizer as well as supporting packages
using NeuralPriorityOptimizer
using NeuralVerification
using LazySets

# Read in the network
gan_network_file = string(@__DIR__, "/../Networks/GANControl/mlp256x4_msle.nnet")
network = read_nnet(gan_network_file)

# Load image data
data_file = string(@__DIR__, "/../Data/AutoTaxiData/Data_3_18_21/SK_DownsampledGANFocusAreaData.h5")
images = h5read(data_file, "y_train")
images = reshape(images, 16*8, :)
labels = h5read(data_file, "X_train")[1:2, :]
labels_scaled = labels ./ [6.366468343804353, 17.248858791583547] 

# Define your query
image_index = 3
state_eps = 1e-4
p = Inf # the order of the norm you're using for the projection. Commonly 1, 2, or Inf

# Check the full latent dimensions, and then a small region around the labeled state 
lbs = [-1.0, -1.0, labels_scaled[1, image_index] - state_eps, labels_scaled[2, image_index] - state_eps]
ubs = [1.0, 1.0, labels_scaled[1, image_index] + state_eps, labels_scaled[2, image_index] + state_eps]

input_set = Hyperrectangle(low=lbs, high=ubs)

# The point (in this case an image) that we'd like to project onto the range of the network 
point = (images[:, image_index] .* 2) .- 1 # rescale image to the range the GAN outputs in

# Use default parameters for the optimization then solve
params = PriorityOptimizerParameters()
time = @elapsed x_star, lower_bound, upper_bound, steps = project_onto_range(network, input_set, point, p, params)

# Print your results 
println()
println("Elapsed time: ", time)
println("Interval: ", [lower_bound, upper_bound])
println("Steps: ", steps)

# Now visualize the projected versus the original image. First plot the generated image 
gen_from_x = (NeuralVerification.compute_output(network, x_star) .+ 1) ./2.0
gen_shaped = reshape(gen_from_x, 16, 8)'
plot(Gray.(gen_shaped), axis = [], title=string("Generated Image ", image_index))

# Now plot the original image 
true_shaped = reshape(images[:, image_index], 16, 8)'
plot(Gray.(true_shaped), axis = [], title=string("True Image ", image_index))


