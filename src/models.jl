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
export bases, coeff

include("linear.jl")
include("nn.jl")

# include("radial.jl")
# export RadialModel

# include("radial_full.jl")
# export FullRadialModel

# include("additive.jl")
# export AdditiveRadialModel

# export LSTSQ, Tikhonov, NN
# export infer!, radial_bases!, uniform_points, evaluate