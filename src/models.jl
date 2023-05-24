# Optimizers
abstract type AbstractOptimizer end
struct LSTSQ <: AbstractOptimizer end
struct Tikhonov <: AbstractOptimizer 
    lambda_min
    lambda_max

    function Tikhonov(l1,l2)
        new(l1,l2)
    end 
end
struct NN <: AbstractOptimizer end

# Interaction Models
abstract type AbstractInteraction end
abstract type Dyadic <: AbstractInteraction end
abstract type Triadic <: AbstractInteraction end
abstract type Quadadic <: AbstractInteraction end

# Bases types
abstract type AbstractRadialBases end

include("bases.jl")
export RadialBases, RadialCutoffBases, RadialKRBases
export bases, coeff, combined_basis_and_coeff

include("rbf.jl")
export RBFForceField

include("piecewise.jl")
export PiecewiseLinearForceField
export PiecewiseQuadraticForceField
export PiecewiseCubicForceField

include("nn.jl")
export NeuralForceField