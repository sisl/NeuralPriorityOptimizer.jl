module NeuralPriorityOptimizer
using NeuralVerification, LazySets, Parameters, DataStructures, LinearAlgebra, HDF5
using NeuralVerification: compute_output, get_activation, TOL
using Convex, Mosek, MosekTools, JuMP, Gurobi
import JuMP.MOI.OPTIMAL, JuMP.MOI.INFEASIBLE, JuMP.MOI.INFEASIBLE_OR_UNBOUNDED

#const gurobi_env = Gurobi.Env()

# From https://githubmemory.com/repo/jump-dev/Gurobi.jl/issues/388
const GUROBI_ENV = Ref{Gurobi.Env}()
function __init__()
    global GUROBI_ENV[] = Gurobi.Env()
end

include("utils.jl")
include("optimization_core.jl")
include("optimization_wrappers.jl")
include("additional_optimizers.jl")
export general_priority_optimization,
       PriorityOptimizerParameters,
       project_onto_range,
       optimize_linear,
       optimize_linear_gradient_split,
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
       mip_linear_value_only,
       mip_linear_uniform_split
end # module
