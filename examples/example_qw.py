from ledsim import *
import material, solve

# Define the settings used for this simulation.
gridOpts    = GridOpts(dzQuantum=1e-10)
modelOpts   = ModelOpts(quantum=True)
quantumOpts = QuantumOpts(boundaryType='periodic')
solverOpts  = SolverOpts(verboseLevel=5)

# Choose the material for the structure and define active region layers
mat = material.AlGaInN()
bar = Layer(material=mat,thickness=4.5e-9,x=0.00,y=0.00,Nd=5e17*1e6,isQuantum=True)
qw  = Layer(material=mat,thickness=3.0e-9,x=0.00,y=0.22,isQuantum=True)

# Define the layer stack and build the structure
layers = [bar,qw,bar]
s = build(layers,gridOpts=gridOpts,modelOpts=modelOpts,quantumOpts=quantumOpts)

# Solve the Poisson equation with periodic boundary conditions and the
# specified electron sheet charge density (units are #/m3)
Ntarget = 1e20*1e6*1e-9
# cond = solve.solve_equilibrium_local(s,solverOpts=solverOpts)
# cond = solve.solve_poisson(s,solverOpts=solverOpts)
cond = solve.solve_poisson_periodic(s,Ntarget=Ntarget,solverOpts=solverOpts)

# Plot the results
pylab.ion()
cond.plot('bands',figNum=1)
cond.plot('doping',figNum=2)
cond.plot('EcWavefunctions',figNum=3)
cond.plot('EvWavefunctions',figNum=4)
# cond.plot('Spectrum',figNum=5)
pylab.show()