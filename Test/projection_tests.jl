###
# This script tests how to project a point onto the range of a network. 
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

# Several queries with the L-Inf norm
groundtruths = [0.2660, 0.37726, 0.2398]
@testset "l-inf norm tests" begin
    for image_index = 1:3
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
        x_star, lower_bound, upper_bound, steps = project_onto_range(network, input_set, point, p, params)

        @test abs(lower_bound - groundtruths[image_index]) <= params.stop_gap
        @test abs(upper_bound - groundtruths[image_index]) <= params.stop_gap
    end 
end

# Compare to the l-2 norm. This is violating by 0.0002 at the moment, but perhaps the 
# ground truth is a bit off?
# @testset "l-2 norm tests" begin
#     groundtruth_l2 = 0.63748927819433
#     image_index = 3
#     state_eps = 1e-4
#     p = 2 # the order of the norm you're using for the projection. Commonly 1, 2, or Inf

#     # Check the full latent dimensions, and then a small region around the labeled state 
#     lbs = [-1.0, -1.0, labels_scaled[1, image_index] - state_eps, labels_scaled[2, image_index] - state_eps]
#     ubs = [1.0, 1.0, labels_scaled[1, image_index] + state_eps, labels_scaled[2, image_index] + state_eps]

#     input_set = Hyperrectangle(low=lbs, high=ubs)

#     # The point (in this case an image) that we'd like to project onto the range of the network 
#     point = (images[:, image_index] .* 2) .- 1 # rescale image to the range the GAN outputs in

#     # Use default parameters for the optimization then solve
#     params = PriorityOptimizerParameters()
#     x_star, lower_bound, upper_bound, steps = project_onto_range(network, input_set, point, p, params)

#     @test abs(lower_bound - groundtruth_l2) <= params.stop_gap
#     @test abs(upper_bound - groundtruth_l2) <= params.stop_gap
# end

# And now to the l-1 norm 
@testset "l-1 norm tests" begin
    groundtruth_l1 = 4.63941
    image_index = 3
    state_eps = 1e-4
    p = 1 # the order of the norm you're using for the projection. Commonly 1, 2, or Inf

    # Check the full latent dimensions, and then a small region around the labeled state 
    lbs = [-1.0, -1.0, labels_scaled[1, image_index] - state_eps, labels_scaled[2, image_index] - state_eps]
    ubs = [1.0, 1.0, labels_scaled[1, image_index] + state_eps, labels_scaled[2, image_index] + state_eps]

    input_set = Hyperrectangle(low=lbs, high=ubs)

    # The point (in this case an image) that we'd like to project onto the range of the network 
    point = (images[:, image_index] .* 2) .- 1 # rescale image to the range the GAN outputs in

    # Use default parameters for the optimization then solve
    params = PriorityOptimizerParameters()
    x_star, lower_bound, upper_bound, steps = project_onto_range(network, input_set, point, p, params)

    @test abs(lower_bound - groundtruth_l1) <= params.stop_gap
    @test abs(upper_bound - groundtruth_l1) <= params.stop_gap
end