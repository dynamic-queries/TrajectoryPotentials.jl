mutable struct NeuralForceField
    phi::Any
    rad::AbstractRadialBases
    N::Int
    f::AbstractArray
    
    function NeuralForceField(phi, rad, f)
        N = size(rad.z,3)
        new(phi,rad,N,f)
    end 
end 

