mutable struct NeuralForceField
    phi::Any
    rad::AbstractRadialBases
    N::Int
    
    function NeuralForceField(phi, rad)
        N = size(rad.z,3)
        new(phi,rad,N)
    end 
end 

function (model::NeuralForceField)(x,ps,st)
    N = model.N
    R = combined_basis_and_coeff(model.rad)
    t = (1/N)* (R * model.phi.(r))
end

function Base.show(io::IO, nff::NeuralForceField)
    print(io, "Neural Force Field")
end 

function loss(model,ps,st,data)
    
end