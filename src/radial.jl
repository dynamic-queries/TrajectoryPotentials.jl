mutable struct RadialModel <: AbstractModel
    z::AbstractArray # size(z) = (input_dims, ntimesteps, nparticles, nics) 
    dz::AbstractArray # size(dz) = (input_dims, ntimesteps, nparticles, nics)
    R::Any
    r::Any
    rdomain::Any
    rbases::Any # Bases functions

    A::Any
    B::Any
    Φ::Any # Inference Operator
    f::Any
    rcoeff::Any # Coefficients for the bases functions

    function RadialModel(z,dz)
        new(z,dz,nothing,nothing,nothing,nothing,nothing,nothing)
    end 
end 

function Base.show(io::IO, rad::RadialModel)
    print(io, "Radial Model\n")
end 

function radial_pairwise!(rad::RadialModel)
    z = rad.z
    s = size(z)
    R = zeros(s[1],Int((s[3])*(s[3]-1)/2),s[2],s[4])
    r = zeros(Int((s[3])*(s[3]-1)/2),s[2],s[4])
    for instance=1:s[4]
        for ts=1:s[2]
            k = 1 
            for part1=1:s[3]
                for part2=1:s[3]
                    if part2>part1
                        temp = z[:,ts,part1,instance].-z[:,ts,part2,instance]
                        R[:,k,ts,instance] .= temp/norm(temp)
                        r[k,ts,instance] = norm(temp)
                        k += 1
                    end
                end
            end
        end
    end
    rad.R = R
    rad.r = r
    nothing
end 

function recurse(i::Int, np::Int)
    s = 0
    for j=1:i-1
        s += (np-j)
    end
    s + 1
end

function Base.get(rad::RadialModel, ics::Int, ts::Int, itp::Int)
    R = rad.R
    np = size(rad.z,3)
    R1 = @views R[:,:,ts,ics]
    start = recurse(itp,np)
    l = (np-itp)
    return R1[:,start:start+l-1]
end

function uniform_points(r::AbstractArray, α::Float64)
    rmin = minimum(r)
    rmax = maximum(r)
    rdomain = rmin:(rmax-rmin)/floor(Int,α*length(r)):rmax
    rdomain
end 

function radial_bases!(rad::RadialModel,γ::Float64,α::Float64)
    r = rad.r
    rdomain = uniform_points(r, α)
    f(x,x0) = exp(-((x-x0)/γ)^2)
    rad.rdomain = rdomain
    rad.rbases = [x->f(x,rad.rdomain[i]) for i=1:length(rad.rdomain)]
    nothing
end

function (rad::RadialModel)(α::Float64, γ::Float64)
    radial_pairwise!(rad)
    radial_bases!(rad,γ,α)
    nothing
end

function assemble_block(rad::RadialModel, ics::Int, par::Int)
    
end 

function assemble_band!(rad::RadialModel, ics::Int, ts::Int)
    r = rad.r[:,ts,ics]
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
    for i=1:nparticles-1
        k = get(rad, ics, ts, i)
        s = size(k,2)
        l = (i-1)*2+1
        start = recurse(i,np)
        lon = (np-i)
        A[l:l+1,start:start+lon-1] = k 
    end 
    rad.A = A

    # Inference matrix
    rad.Φ = (1/np) .* A*B

    # Force term
    rad.f = rad.dz[:,ts,:,ics][:][1:end-2]
    nothing
end 

function infer!(rad::RadialModel, ics::Int, ts::Int, ::LSTSQ)
    assemble_band!(rad,ics,ts)
    rad.rcoeff = pinv(Array(rad.Φ)) * rad.f
    nothing
end 

function infer!(rad::RadialModel, ics::Int, ts::Int, reg::Tikhonov)
    assemble_band!(rad,ics,ts)
    # Invert with Tikhonov regularization
    prob = setupRegularizationProblem(rad.Φ, 0)
    solution = solve(prob, rad.f,alg=:L_curve, λ₁=reg.lambda_min, λ₂=reg.lambda_max)
    rad.rcoeff = solution.x
    nothing
end

function evaluate(rad::RadialModel, x::Float64)
    k = rad.rcoeff
    rbases = rad.rbases
    sol = 0.0
    for i=1:length(k)
        sol += k[i]*rbases[i](x)
    end
    sol
end 

function evaluate(rad::RadialModel, R::AbstractArray)
    k = rad.rcoeff
    rbases = rad.rbases
    
    sol = zero(R)
    for (i,r) in enumerate(R)
        sol[i] = evaluate(rad, r)
    end 
    sol
end