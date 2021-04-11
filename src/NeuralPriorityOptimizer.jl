module NeuralPriorityOptimizer
using NeuralVerification, LazySets

include("utils.jl")
include("optimization_core.jl")
include("optimization_wrappers.jl")
include("approximate_methods.jl")
export general_priority_optimization,
       inclusion_wrapper,
       linear_opt_wrapper,
       fgsm,
       pgd,
       repeated_pgd,
       hookes_jeeves

end # module
