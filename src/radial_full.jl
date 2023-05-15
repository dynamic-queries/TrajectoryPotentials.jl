mutable struct FullRadialModel <: AbstractModel
    z::AbstractArray # size(z) = (input_dims, ntimesteps, nparticles, nics) 
    dz::AbstractArray # size(dz) = (input_dims, ntimesteps, nparticles, nics)
    R::Any
    r::Any
    rdomain::Any
    rbases::Any # Bases functions

    Φ::Any # Inference Operator
    rcoeff::Any # Coefficients for the bases functions

    function RadialModel(z,dz)
        new(z,dz,nothing,nothing,nothing,nothing,nothing,nothing)
    end 
end 

function Base.show(io::IO, rad::FullRadialModel)
    print(io, "Full Radial Model\n")
end 

function radial_pairwise!(rad::FullRadialModel)
    z = rad.z
    s = size(z)
    R = zeros(s[1],s[3],s[3],s[2],s[4])
    r = zeros(s[3],s[3],s[2],s[4])
    for instance=1:s[4]
        for ts=1:s[2]
            for part1=1:s[3]
                for part2=1:s[3]
                        temp = z[:,ts,part1,instance].-z[:,ts,part2,instance]
                        rtemp = norm(temp)
                        if rtemp != 0
                            R[:,part1,part2,ts,instance] .= temp/rtemp
                        else 
                            R[:,part1,part2,ts,instance] .= temp
                        end 
                        r[part1,part2,ts,instance] = rtemp
                    end
                end
            end
        end
    end
    rad.R = R
    rad.r = r
    nothing
end

function radial_bases!(rad::FullRadialModel,γ::Float64,α::Float64)
    r = rad.r
    rdomain = uniform_points(r, α)
    f(x,x0) = exp(-((x-x0)/γ)^2)
    rad.rdomain = rdomain
    rad.rbases = [x->f(x,rad.rdomain[i]) for i=1:length(rad.rdomain)]
    nothing
end 

function (rad::FullRadialModel)()
    radial_pairwise!(rad)
    radial_bases!(rad,γ,α)
    nothing
end