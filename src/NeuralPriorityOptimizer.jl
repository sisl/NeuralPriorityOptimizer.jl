module NeuralPriorityOptimizer
using NeuralVerification, LazySets, Parameters, DataStructures, LinearAlgebra, HDF5
using NeuralVerification: compute_output
using Convex, Mosek, MosekTools, JuMP, Gurobi
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.INFEASIBLE_OR_UNBOUNDED

include("utils.jl")
include("optimization_core.jl")
include("optimization_wrappers.jl")
include("additional_optimizers.jl")
export general_priority_optimization,
       PriorityOptimizerParameters,
       project_onto_range,
       optimize_linear,
       contained_within_polytope,
       reaches_polytope,
       reaches_polytope_binary,
       max_network_difference,
       optimize_convex_program,
       fgsm,
       pgd,
       repeated_pgd,
       hookes_jeeves,
       get_acas_sets,
       mip_linear_value_only
end # module
