from ledsim import *
import material, solve

# Define the settings used for this simulation.
gridOpts    = GridOpts(dzQuantum=1e-10)
modelOpts   = ModelOpts(quantum=False)
quantumOpts = QuantumOpts(boundaryType='Dirichlet')
solverOpts  = SolverOpts(verboseLevel=4)

# Choose the material for the structure and define active region layers
mat = material.AlGaInN()
nGaN   = Layer(material=mat,thickness=100e-9,x=0.00,y=0.00,Nd=4e18*1e6)
bar    = Layer(material=mat,thickness=9.0e-9,x=0.00,y=0.00,Nd=5e17*1e6,isQuantum=True)
qw     = Layer(material=mat,thickness=3.0e-9,x=0.00,y=0.12,isQuantum=True)
spacer = Layer(material=mat,thickness=10.e-9,x=0.00,y=0.00,isQuantum=True)
ebl    = Layer(material=mat,thickness=30.e-9,x=0.10,y=0.00,Na=4e19*1e6,isQuantum=True)
pGaN   = Layer(material=mat,thickness=100e-9,x=0.00,y=0.00,Na=4e19*1e6)

# Define the layer stack.
layers = [nGaN]+[bar,qw]*3+[spacer,ebl,pGaN]
s = build(layers,gridOpts=gridOpts,modelOpts=modelOpts,quantumOpts=quantumOpts)

# Solve using a current boundary condition with the specified target current
# density (units are A/m2)
Jtarget = 35*1e4
cond = solve.bias(s,Jtarget=Jtarget,solverOpts=solverOpts)

cond = cond.offset(0,0,0)
modelOpts.quantum = True

# Plot the results
pylab.ion()
cond.plot('bands',figNum=1)
cond.plot('carriers',figNum=2)
cond.plot('EcWavefunctions',figNum=3)
cond.plot('EvWavefunctions',figNum=4)
pylab.show()