mutable struct RadialBases <: AbstractRadialBases
    """Radial Bases

    Bases defining unit vectors along pair-wise displacement vectors of particles.
    """
    z::AbstractArray
    R::Any
    r::Any

    function RadialBases(z::AbstractArray)
        s = size(z)
        ndims, nts, nparticles, nics = s
        R = zeros(ndims,nparticles,nparticles,nts,nics)
        r = zeros(nparticles,nparticles,nts,nics)

        for ic in 1:nics
            for ts in 1:nts
                for i=1:nparticles
                    for j=1:nparticles
                        R[:,i,j,ts,ic] .= z[:,ts,i,ic] .- z[:,ts,j,ic]
                        t = norm(R[:,i,j,ts,ic])
                        if t != 0
                            R[:,i,j,ts,ic] .= R[:,i,j,ts,ic]/t
                            r[i,j,ts,ic] = t
                        end 
                    end 
                end 
            end 
        end 
        new(z,R,r)
    end 
end 

function bases(rad::RadialBases, particle::Int, ts::Int, ic::Int)
    @views rad.R[:,particle,:,ts,ic]
end 

function coeff(rad::RadialBases, particle::Int, ts::Int, ic::Int)
    @views rad.r[particle,:,ts,ic]
end 

function combined_basis_and_coeff(rad::RadialBases)
    natoms = size(rad.z, 3)
    RB = []
    RC = []
    for i=1:natoms
        push!(RB,bases(rad,i,1,1))
        push!(RC,coeff(rad,i,1,1))
    end 

    RBasis = spzeros(2*natoms,natoms*natoms)
    RCoeff = spzeros(natoms*natoms)
    for i=1:natoms
        RBasis[2*(i-1)+1:2*(i),natoms*(i-1)+1:natoms*(i)] .= RB[i]
        RCoeff[natoms*(i-1)+1:natoms*(i)] .= RC[i]
    end 
    RBasis,RCoeff
end 

#---------------------------------------------------------------------------------------#

mutable struct RadialCutoffBases <: AbstractRadialBases
    """Radial CutOff Bases

    Bases defining unit vectors along pair-wise displacement vectors of particles with a cutoff threshold.
    """
    z::AbstractArray
    R::Any
    r::Any
    order::Any

    function RadialCutoffBases(z::AbstractArray)
        s = size(z)
        ndims, nts, nparticles, nics = s
        R = zeros(ndims,nparticles,nparticles,nts,nics)
        r = zeros(nparticles,nparticles,nts,nics)

        for ic in 1:nics
            for ts in 1:nts
                for i=1:nparticles
                    for j=1:nparticles
                        R[:,i,j,ts,ic] .= z[:,ts,i,ic] .- z[:,ts,j,ic]
                        t = norm(R[:,i,j,ts,ic])
                        if t != 0
                            R[:,i,j,ts,ic] .= R[:,i,j,ts,ic]/t
                            r[i,j,ts,ic] = t
                        end 
                    end 
                end 
            end 
        end 
        new(z,R,r)
    end 
end 

function bases(rad::RadialCutoffBases, thres::Float64, particle::Int, ts::Int, ic::Int)
    rtemp = rad.r[particle,:,ts,ic]
    idx = sortperm(rtemp)
    
    idxt = []
    for id in idx
        if rtemp[id] <= thres
            push!(idxt,id)
        end 
    end 

    rad.order = idxt

    @views rad.R[:,particle,idxt,ts,ic]
end 

function coeff(rad::RadialCutoffBases, thres::Float64, particle::Int, ts::Int, ic::Int)
    rtemp = rad.r[particle,:,ts,ic]
    idx = sortperm(rtemp)
    
    idxt = []
    for id in idx
        if abs(rtemp[id]) <= thres
            push!(idxt,id)
        end 
    end 

    @views rad.r[particle,idxt,ts,ic]
end 


function combined_basis_and_coeff(rad::RadialCutoffBases)
    natoms = size(rad.z, 3)
    RB = []
    RC = []
    thres = 0.5
    for i=1:natoms
        push!(RB,bases(rad,thres,i,1,1))
        push!(RC,coeff(rad,thres,i,1,1))
    end

    ncols = 0
    for i=1:natoms
        ncols += size(RB[i],2)
    end 

    RBasis = spzeros(2*natoms,ncols)
    RCoeff = spzeros(ncols)
    RBasis[1:2,1:size(RB[1],2)] .= RB[1]
    k = size(RB[1],2)
    for i=2:natoms 
        k2 = size(RB[i],2)
        RBasis[2*(i-1)+1:2*(i),k+1:k+k2] .= RB[i]
        RCoeff[k+1:k+k2] .= RC[i]
        k += size(RB[i],2)
    end
    RBasis, RCoeff
end 
#---------------------------------------------------------------------------------------#

mutable struct RadialKRBases <: AbstractRadialBases
    """Radial Khatri Rao Bases

    Bases defining unit vectors along uniquely defined pair-wise displacement between particles.
    Is named so, since this resembles the Khatri-Rao product.
    """
    z::AbstractArray
    R::Any
    r::Any

    function RadialKRBases(z::AbstractArray)
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
        new(z,R,r) 
    end 
end 

function recurse(i::Int, np::Int)
    s = 0
    for j=1:i-1
        s += (np-j)
    end
    s + 1
end

function bases(rad::RadialKRBases, particle::Int, ts::Int, ics::Int)
    itp = particle  
    R = rad.R
    np = size(rad.z,3)
    R1 = @views R[:,:,ts,ics]
    start = recurse(itp,np)
    l = (np-itp)
    @views R1[:,start:start+l-1]
end 

function coeff(rad::RadialKRBases, particle::Int, ts::Int, ics::Int)
    itp = particle  
    r = rad.r
    np = size(rad.z,3)
    R1 = @views r[:,ts,ics]
    start = recurse(itp,np)
    l = (np-itp)
    @views R1[start:start+l-1]
end 


function combined_basis_and_coeff(rad::RadialKRBases)
    natoms = size(rad.z, 3)
    RB = []
    RC = []
    thres = 0.5
    for i=1:natoms
        push!(RB,bases(rad,i,1,1))
        push!(RC,coeff(rad,i,1,1))
    end

    ncols = 0
    for i=1:natoms
        ncols += size(RB[i],2)
    end 

    RBasis = spzeros(2*natoms,ncols)
    RCoeff = spzeros(ncols)
    RBasis[1:2,1:size(RB[1],2)] .= RB[1]
    k = size(RB[1],2)
    for i=2:natoms 
        k2 = size(RB[i],2)
        RBasis[2*(i-1)+1:2*(i),k+1:k+k2] .= RB[i]
        RCoeff[k+1:k+k2] .= RC[i]
        k += size(RB[i],2)
    end
    RBasis, RCoeff
end 
#---------------------------------------------------------------------------------------#