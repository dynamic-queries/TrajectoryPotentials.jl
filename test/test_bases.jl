using Plots
using SparseArrays
using TrajectoryPotentials
using LAMMPSParser
using TimeSeriesDerivatives
ENV["GKSwstype"] = "100"

# Soft-bound potential
filename = "examples/soft-bound/dump.dat"
pars = Parser(filename)
x,y,z = get_positions(pars)
vx,vy,vz = get_velocities(pars)
fx,fy,fz = get_forces(pars) 
natoms = pars.nparticles
ntsteps = pars.ntimesteps
dt = 0.001

F_cx = zeros(natoms,ntsteps-8)
F_cy = similar(F_cx)
V_cx = similar(F_cx)
V_cy = similar(F_cx)

for atom in 1:natoms
    tsd_x = TSDerivative(vx[atom,:],dt)
    tsd_x(VIIIc())
    tsd_y = TSDerivative(vy[atom,:],dt)
    tsd_y(VIIIc())

    # Compute forces
    v_cx,f_cx = get_inferables(tsd_x)
    v_cy,f_cy = get_inferables(tsd_y)

    F_cx[atom,:] .= f_cx
    F_cy[atom,:] .= f_cy
    V_cx[atom,:] .= v_cx
    V_cy[atom,:] .= v_cy
end 

V = zeros(size(V_cx,1), size(V_cx,2), 2)
F = zeros(size(F_cx,1), size(F_cx,2), 2)
V[:,:,1] .= x[:,5:end-4]
V[:,:,2] .= y[:,5:end-4]
F[:,:,1] .= F_cx
F[:,:,2] .= F_cy

V = reshape(permutedims(V,(3,2,1)),(2,:,natoms,1))
F = reshape(permutedims(F,(3,2,1)),(2,:,natoms,1))

# Radial map
rad = RadialBases(V)
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

f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_radial.png")

# Sorted Radial Map with cutoff
rad = RadialCutoffBases(V)
RB = []
RC = []
thres = 0.5
for i=1:natoms
    push!(RB,bases(rad,thres,i,1,1))
    push!(RC,coeff(rad,thres,i,1,1))
end

ncols = 0
for i=1:natoms
    global ncols += size(RB[i],2)
end 

RBasis = spzeros(2*natoms,ncols)
RCoeff = spzeros(ncols)
RBasis[1:2,1:size(RB[1],2)] .= RB[1]
k = size(RB[1],2)
for i=2:natoms 
    k2 = size(RB[i],2)
    RBasis[2*(i-1)+1:2*(i),k+1:k+k2] .= RB[i]
    RCoeff[k+1:k+k2] .= RC[i]
    global k += size(RB[i],2)
end 


f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_cutoff_radial.png")


# Radial Map - Unique pairwise
rad = RadialKRBases(V)
RB = []
RC = []
for i=1:natoms
    push!(RB,bases(rad,i,1,1))
    push!(RC,coeff(rad,i,1,1))
end

ncols = 0
for i=1:natoms
    global ncols += size(RB[i],2)
end 

RBasis = spzeros(2*natoms,ncols)
RCoeff = spzeros(ncols)
RBasis[1:2,1:size(RB[1],2)] .= RB[1]
k = size(RB[1],2)
for i=2:natoms 
    k2 = size(RB[i],2)
    RBasis[2*(i-1)+1:2*(i),k+1:k+k2] .= RB[i]
    RCoeff[k+1:k+k2] .= RC[i]
    global k += size(RB[i],2)
end 


f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_kr_radial.png")