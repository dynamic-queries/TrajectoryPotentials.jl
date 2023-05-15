mutable struct AdditiveRadialModel <: AbstractModel
    z::AbstractArray
    dz::AbstractArray
    R::Any
    r::Any
    rdomain::Any
    rbases::Any

    A::Any # Inference Operator
    B::Any
    rcoeff::Any # Coefficients for the bases functions

    function AdditiveRadialModel(z,dz)
        new(z,dz,nothing,nothing,nothing,nothing,nothing,nothing)
    end 
end 

function Base.show(io::IO, rad::AdditiveRadialModel)
    print(io, "Additive Radial Model\n")
end

function radial_pairwise!(rad::AdditiveRadialModel)
    z = rad.z
    s = size(z)
    R = zeros(s[1],s[3],s[3],s[2],s[4])
    r = zeros(s[3],s[3],s[2],s[4])
    for instance=1:s[4]
        for ts=1:s[2]
            k = 1 
            for part1=1:s[3]
                for part2=1:s[3]
                    temp = z[:,ts,part1,instance].-z[:,ts,part2,instance]
                    R[:,part1,part2,ts,instance] .= temp
                    r[part1,part2,ts,instance] = norm(temp)
                end
            end
        end
    end
    rad.R = R
    rad.r = r
    nothing
end

function radial_bases!(rad::AdditiveRadialModel,γ::Float64,α::Float64)
    r = rad.r
    rdomain = uniform_points(r, α)
    f(x,x0) = exp(-((x-x0)/γ)^2)
    rad.rdomain = rdomain
    rad.rbases = [x->f(x,rad.rdomain[i]) for i=1:length(rad.rdomain)]
    nothing
end

function (rad::AdditiveRadialModel)(γ::Float64,α::Float64)
    @time radial_pairwise!(rad)
    @time radial_bases!(rad,γ,α)
    nothing
end

function infer!(rad::AdditiveRadialModel)
    z = rad.z
    dz = rad.dz
    R = rad.R
    r = rad.r
    rbases = rad.rbases
    rdomain = rad.rdomain

    ndims = size(z,1)
    ics = size(z,4)
    ts = size(z,2)
    np = size(z,3)

    rad.A = zeros(np, length(rbases))
    rad.B = zeros(np)

    for ic = 1:1
        for t = 1:1
            for p = 1:np
                
                rtemp = R[:,:,p,t,ic]
                ftemp = dz[:,t,p,ic]
                Phi = reduce(hcat,[rbases[i].(r[:,p,t,ic]) for i=1:length(rbases)])

                c = rtemp' * rtemp
                rad.A .= rad.A + (1/np).*(c * Phi)
                rad.B .= rad.B + rtemp' * ftemp
            end 
        end 
    end 
    @show cond(Array(rad.A),2) 

    rad.rcoeff = pinv(rad.A)*rad.B
    nothing
end

function evaluate(rad::AdditiveRadialModel, r::Float64)
    k = rad.rcoeff
    rbases = rad.rbases
    
    sol = 0.0
    for i=1:length(k)
        sol += k[i]*rbases[i](r)
    end
    sol
end

function evaluate(rad::AdditiveRadialModel, R::AbstractArray)
    k = rad.rcoeff
    rbases = rad.rbases
    
    sol = zero(R)
    for (i,r) in enumerate(R)
        sol[i] = evaluate(rad, r)
    end 
    sol
end