'''
'''
from ledsim import *
import calc

class Wavefunctions():
  
  def __init__(self,cond,bandType,boundaryType='Dirichlet'):
    '''
    '''
    self.cond         = cond
    self.phiOrig      = cond.phi
    self.bandType     = bandType
    self.boundaryType = boundaryType
    self.calc_wavefunctions()
    
  def calc_wavefunctions(self):
    ''' Calculate the wavefunctions for the specified condition.
        for each point in discretized k-space, need to have:
        location of point in k-space
        volume of k-space corresponding to this point
        energies of all relevant wavefunctions for this k-point
        wavefunctions for this k-point
    '''
    def norm_psi(psi):
      return normalize(terminate(psi,self.kpSize,self.boundaryType),\
                       self.kpSize,self.cond.grid.dz[self.cond.grid.qrIndex])
    
    t0 = time.time()
    if self.bandType == 'Ec':
      startVal = self.cond.EcMin
      endVal = self.cond.EcMax+self.cond.quantumOpts.deltaE
      H = self.cond.Hc
      self.degen = self.cond.CHc['degen']
      self.kpSize = self.cond.CHc['kpSize']
      self.thetaPeriod = self.cond.CHc['thetaPeriod']
    elif self.bandType == 'Ev':
      startVal = self.cond.EvMax
      endVal = self.cond.EvMin-self.cond.quantumOpts.deltaE
      H = self.cond.Hv
      self.degen = self.cond.CHv['degen']
      self.kpSize = self.cond.CHv['kpSize']
      self.thetaPeriod = self.cond.CHv['thetaPeriod']
      
    self.k     = []
    self.theta = []
    self.vk    = []
    self.Ek    = []
    self.psi   = []
    self.psisq = []
    
    k = 0
    dk = self.cond.quantumOpts.dk
    done = False
    self.eigsTime = 0
    while not done:
      done = True
      startValNext = scipy.inf if self.bandType == 'Ec' else -scipy.inf 
      for t in ([0] if k == 0 else theta):
        kx = k*scipy.cos(t)
        ky = k*scipy.sin(t)
        Evec,psi,stats = calc.eigs_range(H(kx,ky,self.boundaryType),startVal,endVal,isGetStats=True)
        self.eigsTime += stats['eigsTime']
        if len(Evec) > 0:
          psi,psisq = norm_psi(psi)
          self.k     += [k]
          self.theta += [t]
          self.vk    += [pi*(dk/2)**2 if k == 0 else 2*pi*k*dk*dtheta/self.thetaPeriod]
          self.Ek    += [Evec]
          self.psi   += [psi]
          self.psisq += [psisq]
          
          if self.bandType == 'Ec':
            done = min(Evec) > max(self.Ek[0])+self.cond.quantumOpts.deltaEk
            startValNext = min(startValNext,min(Evec))
          elif self.bandType == 'Ev':
            done = max(Evec) < min(self.Ek[0])-self.cond.quantumOpts.deltaEk
            startValNext = max(startValNext,max(Evec))
      
      k = k+dk
      theta = scipy.linspace(0,self.thetaPeriod/2,self.cond.quantumOpts.thetaRes)
      dtheta = self.thetaPeriod if len(theta) == 1 else theta[1]-theta[0]
      startVal = startValNext
    self.totalTime = time.time()-t0
      
def terminate(psi,kpSize,boundaryType):
  ''' Terminate psi using the appropriate method given the specified boundary
      condition. When psi is calculated, only the values interior to the domain
      are calculated, as boundary values are known given the applied boundary
      condtions. Valid boundary types are:
        0 : Dirichlet (psi=0 at boundaries)
        1 : Neumann   (dpsi/dz=0 at boundaries)
        2 : periodic  (psi[0]=psi[-1])
  '''
  eignum = scipy.shape(psi)[1]
  if boundaryType == 'Dirichlet':
    psiT = scipy.vstack((psi[0:kpSize,:]*0,psi,psi[-kpSize-1:-1,:]*0))
  elif boundaryType == 'Neumann':
    psiT = scipy.vstack((psi[0:kpSize,:],psi,psi[-kpSize-1:-1,:]))
  elif boundaryType == 'periodic':
    psiT = scipy.vstack((psi,psi[0:kpSize,:]))
  return psiT

def get_psisq(psi,kpSize):
  ''' Calculate the probability density of psi.
  '''
  znum   = scipy.shape(psi)[0]/kpSize
  psiShape = scipy.shape(psi)
  eignum = scipy.shape(psi)[1]
  psisq  = scipy.zeros((znum,eignum))
  for ii in range(0,kpSize):
    psisq = psisq+scipy.absolute(psi[scipy.arange(ii,znum*kpSize,kpSize),:])**2
  return psisq

def normalize(psi,kpSize,dz,isGetPsisq=True):
  ''' Normalize psi, so that the probability density integrates to unity. By
      default, |psisq|**2 is also returned.
  '''
  znum   = scipy.shape(psi)[0]/kpSize
  eignum = scipy.shape(psi)[1]
  psisq  = get_psisq(psi,kpSize)
  normFactor = scipy.sqrt(scipy.sum((psisq[:-1,:]+psisq[1:,:])/2*\
                          dz.repeat(eignum).reshape(znum-1,eignum),axis=0))
  return psi/normFactor, psisq/normFactor**2 if isGetPsisq else psi/normFactor

def assemble_hamiltonian(C1,C2,C3,C4,phi,dz,boundaryType='Dirichlet'):
  ''' boundaryType : Dirichlet (psi=0 at boundaries)
      boundaryType : Neumann   (dpsi/dz=0 at boundaries)
      boundaryType : periodic  (psi[0]=psi[-1])
      boundaryType : extend    (includes reference to points outside simulation domain)
      padCount extends the simulation domain by the specified number of gridpoints.
  '''
  def discretize(C1,C2,C3,C4,dz):
    f = scipy.complex128(scipy.zeros((3,len(dz)-1)))
    f[0,:] = -C1[:-1]/dz[:-1]+1j*C2[:-1]/2+1j*C3[:-1]/2
    f[1,:] =  C1[:-1]/dz[:-1]+C1[1: ]/dz[1: ]+ \
              1j*C2[:-1]/2-1j*C2[1: ]/2- \
              1j*C3[:-1]/2+1j*C3[1: ]/2+ \
              C4[:-1]*dz[:-1]/2+C4[1: ]*dz[1: ]/2
    f[2,:] = -C1[1: ]/dz[1: ]-1j*C2[1: ]/2-1j*C3[1: ]/2
    return 2*f/(dz[:-1]+dz[1: ])
 
  kpSize = len(C1[:,0,0])
  if boundaryType == 'periodic':
    qnum = len(phi)-1
    ind = scipy.concatenate((scipy.array([len(dz)-1]),scipy.arange(0,len(dz))))
    pot = calc.stretch(-q*phi[ :-1],kpSize)
  elif boundaryType in ['Dirichlet','Neumann','extend']:
    qnum = len(phi)-2
    ind = scipy.arange(0,len(dz))
    pot = calc.stretch(-q*phi[1:-1],kpSize)
  else:
    raise ValueError, 'Unknown boundaryType!'
  validElement = scipy.sum(C1+C2+C3+C4,2) != 0
  nzmax = scipy.sum(scipy.sum(validElement))*qnum*3+kpSize*qnum
  sm = calc.SparseMaker(nzmax,isComplex=True)
  sm.add_diag(kpSize,0,pot)
  for ii in range(0,kpSize):
    for jj in range(0,kpSize):
      if validElement[ii,jj]:
        f = discretize(C1[ii,jj,ind],C2[ii,jj,ind],C3[ii,jj,ind],C4[ii,jj,ind],dz[ind])
        r = scipy.arange(ii,ii+kpSize*qnum,kpSize)
        c = scipy.arange(jj,jj+kpSize*qnum,kpSize)
        sm.add_elem(r,c+0*kpSize,f[0,:])
        sm.add_elem(r,c+1*kpSize,f[1,:])
        sm.add_elem(r,c+2*kpSize,f[2,:])
  if boundaryType == 'Dirichlet':
    mat = sm.assemble('csr')
    return mat[:,kpSize:-kpSize]
  elif boundaryType == 'Neumann':
    mat = sm.assemble('lil')
    mat[:, 1*kpSize: 2*kpSize] = mat[:, 1*kpSize: 2*kpSize]+mat[:,:kpSize ]
    mat[:,-2*kpSize:-1*kpSize] = mat[:,-2*kpSize:-1*kpSize]+mat[:,-kpSize:]
    return mat[:,kpSize:-kpSize].tocsr()
  elif boundaryType == 'periodic':
    mat = sm.assemble('lil')
    mat[:, 1*kpSize: 2*kpSize] = mat[:, 1*kpSize: 2*kpSize]+mat[:,-kpSize:]
    mat[:,-2*kpSize:-1*kpSize] = mat[:,-2*kpSize:-1*kpSize]+mat[:,:kpSize ]
    return mat[:,kpSize:-kpSize].tocsr()
  else:
    mat = sm.assemble('csr')
    return mat

def get_carriers_quantum(cond,phi,phiN,phiP):
  ''' Calculate the carrier density at each gridpoint in the quantum region.
  '''         
  def fc(Ef,E,phiOrig):
    return 1./(1+scipy.exp((E[:,scipy.newaxis]-Ef-q*(phiOrig[cond.grid.qIndex]-phi))/(kB*cond.modelOpts.T)))
  def fv(Ef,E,phiOrig):
    return 1.-fc(Ef,E,phiOrig)
    
  wf = cond.EcWavefunctions
  Efn = q*(phiN-phi)
  phiOrig = wf.phiOrig
  n = scipy.zeros(len(cond.grid.qIndex))
  for ii in range(len(wf.k)):
    n += scipy.sum(wf.psisq[ii]*wf.vk[ii]*scipy.transpose(fc(Efn,wf.Ek[ii],phiOrig)),1)
  n = n/(4*pi**2)*wf.degen
  
  wf = cond.EvWavefunctions
  Efp = q*(phiP-phi)
  phiOrig = wf.phiOrig
  p = scipy.zeros(len(cond.grid.qIndex))
  for ii in range(len(wf.k)):
    p += scipy.sum(wf.psisq[ii]*wf.vk[ii]*scipy.transpose(fv(Efp,wf.Ek[ii],phiOrig)),1)
  p = p/(4*pi**2)*wf.degen
  return n,p
    
def calc_gain(cond):
  '''
  '''
  return 0,0