abstract type AbstractModel end

abstract type AbstractOptimizer end
struct LSTSQ <: AbstractOptimizer end
struct Tikhonov <: AbstractOptimizer 
    lambda_min
    lambda_max

    function Tikhonov(l1,l2)
        new(l1,l2)
    end 
end

abstract type AbstractInteraction end
struct Dyadic <: AbstractInteraction end
struct Triadic <: AbstractInteraction end
struct Quadadic <: AbstractInteraction end

include("radial.jl")
export RadialModel, infer!, radial_bases!, uniform_points
include("additive.jl")
export AdditiveRadialModel, infer!, evaluate

export LSTSQ, Tikhonov