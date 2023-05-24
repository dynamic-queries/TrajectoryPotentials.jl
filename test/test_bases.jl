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
RBasis,RCoeff = combined_basis_and_coeff(rad)
f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_radial.png")

# Sorted Radial Map with cutoff
rad = RadialCutoffBases(V)
RBasis,RCoeff = combined_basis_and_coeff(rad)
f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_cutoff_radial.png")


# Radial Map - Unique pairwise
rad = RadialKRBases(V)
RBasis,RCoeff = combined_basis_and_coeff(rad)
f1 = heatmap(Array(RBasis))
f2 = plot(Array(RCoeff))
plot(f1,f2,size=(1000,1000))
savefig("figures/test_kr_radial.png")