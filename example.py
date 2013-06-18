from ledsim import *
import material, solve

# Choose the material for the structure
mat = material.AlGaInN()

# Define the layer stack
layers = [Layer(mat,thickness=100e-9,x=0.0,y=0.0,Nd=4e18*1e6),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0,Nd=1e18*1e6),
          Layer(mat,thickness=3e-9  ,x=0.0,y=0.1),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0,Nd=1e18*1e6),
          Layer(mat,thickness=3e-9  ,x=0.0,y=0.1),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0,Nd=1e18*1e6),
          Layer(mat,thickness=3e-9  ,x=0.0,y=0.1),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0,Nd=1e18*1e6),
          Layer(mat,thickness=3e-9  ,x=0.0,y=0.1),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0),
          Layer(mat,thickness=3e-9  ,x=0.0,y=0.1),
          Layer(mat,thickness=10e-9 ,x=0.0,y=0.0),
          Layer(mat,thickness=30e-9 ,x=0.1,y=0.0,Na=4e19*1e6),
          Layer(mat,thickness=100e-9,x=0.0,y=0.0,Na=4e19*1e6)]

so = solve.SolverOpts(verboseLevel=3)


s  = build(layers)

c1 = solve.solve_equilibrium_local(s)
c2 = solve.bias(c1,Jtarget=100*1e4)
pylab.ion()

c2.plot('bands',figNum=1)
c2.plot('carriers',figNum=2)
c2.plot('recombination',figNum=3)
c2.plot('currents',figNum=4)

pylab.show()