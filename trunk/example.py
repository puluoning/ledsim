from ledsim import *
import material, solve

# Choose the material for the structure.
mat = material.AlGaInN()

# Define some layers used in the LED active region.
nGaN   = Layer(material=mat,thickness=100e-9,x=0.00,y=0.00,Nd=4e18*1e6)
bar    = Layer(material=mat,thickness=4.5e-9,x=0.00,y=0.00,isQuantum=True)
qw     = Layer(material=mat,thickness=3.0e-9,x=0.00,y=0.12,isQuantum=True)
spacer = Layer(material=mat,thickness=10.e-9,x=0.00,y=0.00,isQuantum=True)
ebl    = Layer(material=mat,thickness=30.e-9,x=0.10,y=0.00,Na=4e19*1e6)
pGaN   = Layer(material=mat,thickness=100e-9,x=0.00,y=0.00,Na=4e19*1e6)

# Define the layer stack.
# layers = [nGaN]+[bar,qw]*3+[spacer,ebl,pGaN]
layers = [bar,qw,bar]
# layers = [qw]

# Build the structure
gridOpts    = GridOpts(dzQuantum=1e-10)
modelOpts   = ModelOpts(quantum=True)
quantumOpts = QuantumOpts(boundaryType='periodic')
s = build(layers,gridOpts=gridOpts,modelOpts=modelOpts,quantumOpts=quantumOpts)

# Solve for two conditions, the local equilibrium and the condition at 35A/cm2
solverOpts = SolverOpts(verboseLevel=4)
c1 = solve.solve_equilibrium_local(s,solverOpts=solverOpts)
c2 = solve.solve_poisson_periodic(c1,Ntarget=1e17*1e6*1e-9,solverOpts=solverOpts)

pylab.ion()
c2.plot('bands',figNum=1)
c2.plot('carriers',figNum=2)
c2.plot('EcWavefunctions',figNum=3)
c2.plot('EvWavefunctions',figNum=4)
pylab.show()