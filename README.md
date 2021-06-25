# NeuralPriorityOptimizer
This library is meant to perform a variety of simple optimization tasks. The wrapper section describes how to run several useful optimization tasks over neural networks. For example, projecting a point onto the range of a network or optimizing a linear or convex objective over the range of a network. Examples of the use of most wrappers can be found [here](https://github.com/castrong/NeuralPriorityOptimizer.jl/blob/main/Examples/) 

<!---A writeup (in progress) describing how the algorithm works can be found in this [overleaf](https://www.overleaf.com/read/qvkssjmbrgyr). Any feedback or clarifying questions would be greatly appreciated! You can post them as issues to this repository or reach me by email at castrong@stanford.edu.  --->

## Quick start
To add this package you must have [the Julia programming language](https://julialang.org/) installed as well as the [Gurobi](https://github.com/jump-dev/Gurobi.jl) and [Mosek](https://github.com/MOSEK/Mosek.jl) packages configured, both of which have free academic licenses. We hope to update the package to run with open-source linear program and convex program solvers that don't require a license, like GLPK and COSMO. Enter the Julia REPL, then type ] to enter the package manager. Then run the following lines:
`pkg> add https://github.com/sisl/NeuralVerification.jl`
`pkg> add https://github.com/sisl/NeuralPriorityOptimizer.jl`

Several examples of loading networks and running simple queries can be found in the `examples` folder. For example, `examples/linear_hello_world.jl` shows how to load a network then optimize a linear function over the range of the network within an input region. Feedforward ReLU networks are the only type of networks currently supported by this tool. A general optimization algorithm and several useful wrappers are implemented and described in the following sections of the README.

## Wrappers
Here's a list of the currently implemented wrappers that are around the core optimizer. All can be found in [this file](https://github.com/castrong/NeuralPriorityOptimizer.jl/blob/main/src/optimization_wrappers.jl).

All wrappers return results in the same format: (1) best input, (2) lower bound on the objective value, (3) upper bound on the objective value, (4) steps taken by the algorithm. To go from the best_input to the corresponding point in the output space you can use the function `NeuralVerification.compute_output(network, best_x)`.

##### `project_onto_range(network, input_set, y₀, p, params; solver=Ai2z())`
This wrapper projects the point y₀ in the output space onto the range of the network.  It returns the input which the network maps to the closest output in the range. This is equivalent solving the problem minimize ||y - y₀||_p s.t. y = network(x), x in input_set. It returns the objective associated with this optimization problem. If the objective is equal to 0, then the point y₀ is contained in the range. If it is greater than 0, then it is not. This approach requires solving a convex program at every step. 

##### `optimize_linear(network, input_set, coeffs, params; maximize=true)`
This wrapper optimizes a linear function over the range of the network. If maximize=true, it solves maximize coeffs^T y s.t. y = network(x), x in input_set. If maximize = false, then it solves minimize coeffs^T y s.t. y = network(x), x in input_set. This approach requires only analytical operations at every step (no LPs or convex programs).

##### `optimize_convex_program(network, input_set, obj_fcn, params; maximize=true, solver=Ai2z())`
This wrapper optimizes a convex function over the range of the network. If maximize is true, then it solves the problem maximize obj_fcn(y) s.t. y = network(x), x in input_set. If maximize is false, then it solves the problem minimize obj_fcn(y) s.t. y = network(x), x in input_set. This approach requires solving a convex program at every step. 

##### `reaches_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)`
This wrapper can be used to check whether a network can reach a polytope when its input is within the input_set. It achieves this by projecting onto the polytope by solving the optimization problem minimize ||y - z||_p s.t. z in polytope, y = f(x), x in input_set. If the objective value is 0, then the polytope is reachable. If the objective value is greater than 0, then the polytope is not reachable. If it ever finds that the lower bound on the objective is greater than 0, it stops early. This approach requires solving a convex program at every step. 

##### `distance_to_polytope(network, input_set, polytope, params; solver=Ai2z(), p=2)`
This wrapper is the same as `reaches_polytope` described above, except it won't stop once it finds that the lower bound on the objective is greater than 0 since here we would like a numerical shortest distance between the range of the network and the polytope.

##### `contained_within_polytope(network, input_set, polytope, params; solver=Ai2z())`
This wrapper can be used to check whether a network's output is contained with a polytope. Suppose our polytope is defined as {x | Ax <= b}. We check whether the output will always be within this polytope by maximizing the violation of any of the polytope constraints: we solve the problem maximize  max(max(Ay - b, 0)) s.t. y = f(x), x in input_set. If this is greater than 0 we must be able to violate some of the polytope's constraints, and so it is not  This approach requires only analytical operations at every step (no LPs or convex programs).

##### `max_network_difference(network1, network2, input_set; solver=Ai2z(), p=2)`
Find the maximum distance between the outputs of two networks under the p-norm. This is done by solving the problem maximize ||y1 - y2||_p s.t. y1 = network1(x), y2 = network2(x), x in input_set.  This approach requires only analytical operations at every step (no LPs or convex programs) by formulating  but may require a lot of splitting to get to a low optimality gap. 

## Optimizer parameters
Here is a description of the parameters you can use to adjust the optimizer. The struct is given as 

@with_kw struct PriorityOptimizerParameters
    max_steps::Int = 1000
    early_stop::Bool = true
    stop_frequency::Int = 200
    stop_gap::Float64 = 1e-4
    initial_splits::Int = 0
    verbosity::Int = 0
end

max_steps gives the maximum number of splits the optimizer will perform before returning the best bounds found so far. early_stop tells it whether to find a concrete value every once in a while to update the optimality gap and potentially return early. stop_frequency dictates how often it should stop splitting to find a concrete value to perform that update and early stop check. stop_gap gives the optimality gap used to return early - if the solver ever goes below this optimality gap then it will return with the current bounds. initial_splits tells it how many times to split the space before evaluating any of the cells. verbosity = 0 will tell it to not print while solving while verbosity = 1 will tell it to print some information while solving. 

## General Function
The general function can be found in [optimization_core.jl](https://github.com/castrong/NeuralPriorityOptimizer.jl/blob/main/src/optimization_core.jl) Throughout this description we will assume without loss of generality that we would like to maximize an objective. At a high level, the algorithm can maximize an arbitrary objective functions as long as you have two components:

(1) A function which maps a hyperrectangle cell in the input space to an overestimate of the objective. In order for the optimizer to converge to the true objective given infinite time, this overestimate should be tight in the limit as the volume of the hyperrectangle goes to 0 (although I have not formally proven this). Call this function `overestimate_cell`. 

(2) A function which maps a hyperrectangle cell in the input space to an achievable value and returns that value and the associated input in the cell. Call this function `achievable_value`

The algorithm has the high level structure:
(1) Start with a hyperrectangular input cell that you're trying to optimize over
(2) Run `overestimate_cell` on this cell, and add the cell to a priority queue with that value.
(3) Pull the cell with highest priority off of the queue. If early stopping is on, check for a concrete value, update your optimality gap. Return early if the optimality gap is low enough. 
(4) Split the cell you pulled off of the priority queue, run `overestimate_cell` on each sub-cell, then push them to the priority queue with their overestimates.
(5) Repeat steps (3) and (5), splitting cells and overestimating them and occasionally finding a concrete value, until either you reach the maximum number of steps or the optimality gap is small enough and you return.

The core function has the signature `general_priority_optimization(start_cell::Hyperrectangle, overestimate_cell, achievable_value, params::PriorityOptimizerParameters, lower_bound_threshold, upper_bound_threshold)`. `start_cell` gives your original cell ,`overestimate_cell` is a function which maps from a hyperrectangular cell to an overestimate of the objective, `achievable_value` is a function which maps from a hyperrectangular cell to an achievable objective value and the corresponding input, `params` which gives the solver parameters described in the Optimizer Parameters section of this README, `lower_bound_threshold` which gives a value which if the lower bound is ever above `lower_bound_threshold` the solver will return, and `upper_bound_threshold` which gives a value which if the upper bound is ever below `upper_bound_threshold` the solver will return.

There is also a wrapper to this core function which allows for maximization and minimization with the signature `general_priority_optimization(start_cell::Hyperrectangle, relaxed_optimize_cell, evaluate_objective, params::PriorityOptimizerParameters, maximize; bound_threshold_realizable=(maximize ? Inf : -Inf), bound_threshold_approximate=(maximize ? -Inf : Inf))`. For this function `start_cell` is the starting cell, `relaxed_optimize_cell` gives an upper bound for a cell if maximizing and a lower bound for a cell if minimizing, `evaluate_objective` which gives an achievable objective value and the corresponding input, `params` which are again the solver parameters, `maximize` which is true if maximizing and false if minimizing. If minimizing the negative of the objective is maximized with the other core function. Then, `bound_threshold_realizable` if maximizing is used as the `lower_bound_threshold` and if minimizing has its negative used as the `lower_bound_threshold`. This is meant to represent the threshold on the concrete values found by the `achievable_value` function which when you do better than that you can return.   `bound_threshold_approximate` if maximizing is used as the `upper_bound_threshold` and if minimizing has its negative used as the `upper_bound_threshold`. This is meant to represent the threshold on the approximate values found by the `relaxed_optimize_cell` which when you do better than it you can return. These thresholds are useful in wrappers such as `reaches_polytope` which allows it to return after showing that it can reach the polytope. 

## Networks available in the repository 
We have several different types of networks available in the [networks folder](https://github.com/castrong/NeuralPriorityOptimizer.jl/tree/main/networks) of the repository. All networks are provided in .nnet format. This format is described, and helper functions to convert between other commonly used formats are provided in [this repository](https://github.com/sisl/NNet). Feedforward ReLU networks are the only type of networks currently supported by this tool. We have the following folders of networks: 

(1) The folder CAS which contains networks trained to emulate the ACAS Xu tables. See [Policy compression for aircraft collision avoidance systems](https://ieeexplore.ieee.org/document/7778091) for the rationale behind these networks.

(2) The folder AutoTaxi contains networks that map from an image to a control effort for an aircraft taxiing down a runway. See [Validation of Image-Based Neural Network Controllers through Adaptive Stress Testing](https://arxiv.org/pdf/2003.02381.pdf) section IV for description of this use case. The network maps from an image to a crosstrack error and heading error. Then, in this repository the networks have a single output with the last layer representing the coefficients for the proportional controller which take the crosstrack error and heading error and map them to a control effort to steer the plane.

(3) The folder GANControl contains several networks that are either a GAN representing the AutoTaxi images, or concatenate a GAN with one of those AutoTaxi control networks.

(4) The folder MNIST contains several MNIST classification images. See [here](http://www.pymvpa.org/datadb/mnist.html) for a description of the MNIST dataset. 

## Code Structure
There are several key files in the repository. `optimization_core.jl` contains the general optimizer, a wrapper to that optimizer which allows for maximization and minimization, and a struct defining the solver parameters. `optimization_wrappers.jl` contains several wrappers to that general optimizer which perform several useful optimizations. Those are described in the Wrappers section of this README. `utils.jl` contains a variety of utils for the wrapper and optimization core. It has some functions to perform projections, do splits, and more! `approximate_methods.jl` has some approximate optimization functions that haven't been used much or thoroughly tested yet.

## ACAS Xu Benchmarks 
TODO: This section will contain some timing comparisons between different solvers on the ACAS Xu benchmarks with properties 1, 2, 3, and 4. These verification problems can be reformulated as an optimization problem and solved with our wrapper. For the code to do this, see [this file](https://github.com/castrong/NeuralPriorityOptimizer.jl/blob/main/Examples/acas_example.jl)

