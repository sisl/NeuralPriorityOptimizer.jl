module NeuralPriorityOptimizer
using NeuralVerification, LazySets, Parameters, DataStructures, LinearAlgebra

include("utils.jl")
include("optimization_core.jl")
include("optimization_wrappers.jl")
include("approximate_methods.jl")
export general_priority_optimization,
       PriorityOptimizerParameters,
       project_onto_range,
       optimize_linear,
       fgsm,
       pgd,
       repeated_pgd,
       hookes_jeeves
end # module
