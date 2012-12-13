import scipy, pylab, material, solve
from ledsim import *

mat = material.AlGaInN()
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

so = SolverOpts(verboseLevel=3)

c1 = build(layers,solverOpts=so)
c2 = solve.bias(c1,Jtarget=1e4,solverOpts=so)

pylab.ion()

c2.plot('bands',figNum=1)
c2.plot('carriers',figNum=2)
c2.plot('recombination',figNum=3)
c2.plot('currents',figNum=4)

pylab.show()