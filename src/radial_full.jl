mutable struct FullRadialModel <: Dyadic
    z::AbstractArray # size(z) = (input_dims, ntimesteps, nparticles, nics) 
    dz::AbstractArray # size(dz) = (input_dims, ntimesteps, nparticles, nics)
    R::Any
    r::Any
    rdomain::Any
    rbases::Any # Bases functions

    A::Any
    B::Any
    Φ::Any # Inference Operator
    rcoeff::Any # Coefficients for the bases functions

    function FullRadialModel(z,dz)
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

function (rad::FullRadialModel)(γ::Float64,α::Float64)
    radial_pairwise!(rad)
    radial_bases!(rad,γ,α)
    nothing
end

function assemble_band!(rad::FullRadialModel, par::Int, ics::Int, ts::Int)
    r = rad.r[:,:,ts,ics][:]
    bases = rad.rbases
    
    # Assemble rbf bases
    B = spzeros(length(r), length(bases))
    for j=1:length(bases)
        B[:,j] = bases[j].(r)
    end 
    rad.B = B
    
    # Assemble frames
    np = nparticles = size(rad.z,3)
    A = spzeros(size(rad.z,1)*(size(rad.z,3)-1), size(rad.R,2))
    for i=1:nparticles
        A[(i-1)*2+1:i*2,(i-1)*np+1:i*np] = rad.R[:,:,i,ts,ics]
    end 
    rad.A = A

    # Inference matrix
    rad.Φ = (1/np) .* A*B

    # Force term
    rad.f = rad.dz[:,:,ts,ics][:]
    
    nothing
end 

function infer!(rad::FullRadialModel, par::Int, ics::Int, ts::Int, ::LSTSQ)
    assemble_band!(rad, par, ics, ts)
    rad.rcoeff = pinv(Array(rad.Φ))*rad.f
    nothing
end 

function infer!(rad::FullRadialModel,par::Int, ics::Int, ts::Int, reg::Tikhonov)
    assemble_band!(rad, par,ics,ts)
    # Invert with Tikhonov regularization
    prob = setupRegularizationProblem(rad.Φ, 0)
    solution = solve(prob, rad.f,alg=:L_curve, λ₁=reg.lambda_min, λ₂=reg.lambda_max)
    rad.rcoeff = solution.x
    nothing
end