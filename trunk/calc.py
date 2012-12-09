''' Calc module contains useful mathematical methods used throughout ledsim.
'''
import scipy

def diffuse(a,dz,Ld):
  ''' Diffuse the quantity a on the grid specified by dz. The diffusion 
      length is constant and given by Ld.
  '''
  if Ld == 0:
    return a
  else:
    b = scipy.copy(a)
    T = 1.
    D = Ld**2/(4*T)
    D = D*scipy.ones(len(dz))
    rnum   = len(dz)
    dtmax  = 0.25/max(D/dz**2)
    nsteps = max(rnum,scipy.ceil(T/dtmax))
    dt     = T/nsteps;
    ind1   = range(1,rnum)+[rnum-1]
    ind2   = [0]+range(0,rnum-1)
    for ii in scipy.arange(0,nsteps):
      b[1:-1] = (b+dt/(dz*(dz+dz[ind1]))*(D+D[ind1])*(b[ind1]-b)+ \
                   dt/(dz*(dz+dz[ind2]))*(D+D[ind2])*(b[ind2]-b))[1:-1]
    return b