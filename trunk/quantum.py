'''
'''
from ledsim import *
import calc

class Wavefunctions():
  
  def __init__(self,cond,bandType,H,boundaryType=0,quantumOpts=opts.QuantumOpts()):
    '''
    '''
    self.phiOrig      = cond.phi
    self.bandType     = bandType
    self.H            = H
    self.quantumOpts  = quantumOpts
    self.boundaryType = boundaryType
    self.calc_wavefunctions(cond)
    
  def calc_wavefunctions(self,cond):
    ''' Calculate the wavefunctions for the specified condition.
    '''
    def norm_psi(psi):
      return normalize(terminate(psi,self.kpSize,self.boundaryType),self.kpSize,cond.grid.dz)

    if self.bandType == 'Ec':
      startVal = cond.EcMin
      endVal   = cond.EcMax+self.quantumOpts.deltaE
    elif self.bandType == 'Ev':
      startVal = cond.EvMax
      endVal   = cond.EvMin-self.quantumOpts.deltaE
    k  = 0
    dk = self.quantumOpts.dk
    done = False
    self.psiDict = {}
    while not done:
      done = True
      startValNext = scipy.inf if self.bandType == 'Ec' else -scipy.inf 
      for t in ([0] if k == 0 else theta):
        Hmat,self.kpSize,self.degen,self.thetaPeriod = self.H(cond,k,t,self.boundaryType)
        Evec,psi = calc.eigs_range(Hmat,startVal,endVal)
        if len(Evec) > 0:
          psi,psisq = norm_psi(psi)
          Vk = pi*(dk/2)**2 if k == 0 else 2*pi*k*dk*dtheta/self.thetaPeriod
          self.psiDict[(k,t)] = {'E':Evec,'psi':psi,'psisq':psisq,'Vk':Vk}
          if self.bandType == 'Ec':
            term = min(Evec)+self.quantumOpts.deltaEk if k == 0 else term
            done = False if min(Evec) < term else done
            startValNext = min(startValNext,min(Evec))
          else:
            term = max(Evec)-self.quantumOpts.deltaEk if k == 0 else term
            done = False if min(Evec) < term else done
            startValNext = max(startValNext,max(Evec))
      k      = k+dk
      theta  = scipy.linspace(0,self.thetaPeriod/2,self.quantumOpts.thetaRes)
      dtheta = self.thetaPeriod if len(theta) == 1 else theta[1]-theta[0]
      startVal = startValNext
      
  def get_carriers(self,cond):
    ''' Calculate the carrier density.
    '''         
    def fc(Ef,E):
      return 1./(1+scipy.exp((E[:,scipy.newaxis]-Ef-q*(self.phiOrig-cond.phi))/(kB*cond.modelOpts.T)))
    def fv(Ef,E):
      return 1.-fc(Ef,E)
    
    n = scipy.zeros(len(cond.grid.qIndex))
    for kpt in self.psiDict.values():
      if self.bandType == 'Ec':
        n += scipy.sum(kpt['psisq']*kpt['Vk']*scipy.transpose(fc(cond.Efn[cond.grid.qIndex],kpt['E'])),1)
      else:
        n += scipy.sum(kpt['psisq']*kpt['Vk']*scipy.transpose(fv(cond.Efp[cond.grid.qIndex],kpt['E'])),1)
    return n/(4*pi**2)*self.degen

def get_Hc1x1(cond,kx,ky,boundaryType=0):
  ''' C1 corresponds to type 1 terms (kz C1 kz)
      C2 corresponds to type 2 terms (kz C2   )
      C3 corresponds to type 3 terms (   C3 kz)
      C4 corresponds to type 4 terms (   C4   )
  '''
  kpSize = 1
  
  qrIndex = cond.grid.qrIndex
  C1 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C2 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C3 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C4 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  
  C1[0,0,:] = hbar**2/(2*cond.meperp)
  C4[0,0,:] = cond.Eref+cond.Eg0+cond.delcr+cond.delso/3+hbar**2/(2*cond.mepara)*(kx**2+ky**2)+ \
              (cond.a1+cond.D1)*cond.epszz+(cond.a2+cond.D2)*(cond.epsxx+cond.epsyy)
  Hc = assemble_hamiltonian(C1[:,:,qrIndex],C2[:,:,qrIndex],\
                            C3[:,:,qrIndex],C4[:,:,qrIndex],cond.phi,cond.grid.dz,boundaryType)
  return Hc, kpSize, 2, pi

def get_Hv3x3(cond,kx,ky,boundaryType=0):
  ''' C1 corresponds to type 1 terms (kz C1 kz)
      C2 corresponds to type 2 terms (kz C2   )
      C3 corresponds to type 3 terms (   C3 kz)
      C4 corresponds to type 4 terms (   C4   )
  '''
  kpSize = 3
  qrIndex = cond.grid.qrIndex
  kt = scipy.sqrt(kx**2+ky**2)
  C1 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C2 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C3 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))
  C4 = scipy.complex128(scipy.zeros((kpSize,kpSize,cond.grid.rnum)))

  C1[0,0,:] = hbar**2/(2*m0)*(cond.A1+cond.A3)
  C1[1,1,:] = hbar**2/(2*m0)*(cond.A1+cond.A3)
  C1[2,2,:] = hbar**2/(2*m0)*cond.A1
  C2[0,2,:] = -1j*hbar**2/(2*m0)*cond.A6*kt/2
  C2[1,2,:] = -1j*hbar**2/(2*m0)*cond.A6*kt/2
  C2[2,0,:] =  1j*hbar**2/(2*m0)*cond.A6*kt/2
  C2[2,1,:] =  1j*hbar**2/(2*m0)*cond.A6*kt/2
  C3[0,2,:] = -1j*hbar**2/(2*m0)*cond.A6*kt/2
  C3[1,2,:] = -1j*hbar**2/(2*m0)*cond.A6*kt/2
  C3[2,0,:] =  1j*hbar**2/(2*m0)*cond.A6*kt/2
  C3[2,1,:] =  1j*hbar**2/(2*m0)*cond.A6*kt/2
  C4[0,0,:] = cond.Eref+cond.delcr+cond.delso/3+hbar**2/(2*m0)*(cond.A2+cond.A4)*kt**2+ \
              (cond.D1+cond.D3)*cond.epszz+(cond.D2+cond.D4)*2*cond.epsxx
  C4[0,1,:] = hbar**2/(2*m0)*cond.A5*kt**2
  C4[1,0,:] = hbar**2/(2*m0)*cond.A5*kt**2
  C4[1,1,:] = cond.Eref+cond.delcr-cond.delso/3+hbar**2/(2*m0)*(cond.A2+cond.A4)*kt**2+ \
              (cond.D1+cond.D3)*cond.epszz+(cond.D2+cond.D4)*2*cond.epsxx
  C4[1,2,:] = scipy.sqrt(2)*cond.delso/3
  C4[2,1,:] = scipy.sqrt(2)*cond.delso/3
  C4[2,2,:] = cond.Eref+hbar**2/(2*m0)*cond.A2*kt**2+ \
              cond.D1*cond.epszz+cond.D2*2*cond.epsxx
  HvU = assemble_hamiltonian( C1[:,:,qrIndex], C2[:,:,qrIndex],\
                              C3[:,:,qrIndex], C4[:,:,qrIndex],cond.phi,cond.grid.dz,boundaryType)
#  HvL = assemble_hamiltonian( C1[:,:,qrIndex],-C2[:,:,qrIndex],\
#                             -C3[:,:,qrIndex], C4[:,:,qrIndex],cond.phi,cond.grid.dz,boundaryType)
#  return HvU, HvL, kpSize
  return HvU, kpSize, 2, pi

def get_Hv6x6(cond,kx,ky,boundaryType=0):
  ''' C1 corresponds to type 1 terms (kz C1 kz)
      C2 corresponds to type 2 terms (kz C2   )
      C3 corresponds to type 3 terms (   C3 kz)
      C4 corresponds to type 4 terms (   C4   )
  '''
  kpSize = 6
  pass

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
  if boundaryType == 0:
    psiT = scipy.vstack((psi[0:kpSize,:]*0,psi,psi[-kpSize-1:-1,:]*0))
  elif boundaryType == 1:
    psiT = scipy.vstack((psi[0:kpSize,:],psi,psi[-kpSize-1:-1,:]))
  elif boundaryType == 2:
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

def assemble_hamiltonian(C1,C2,C3,C4,phi,dz,boundaryType=0):
  ''' boundaryType=0 : Dirichlet (psi=0 at boundaries)
      boundaryType=1 : Neumann   (dpsi/dz=0 at boundaries)
      boundaryType=2 : periodic  (psi[0]=psi[-1])
      boundaryType=3 : extend    (includes reference to points outside simulation domain) 
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
  if boundaryType == 2:
    qnum = len(phi)-1
    ind = scipy.concatenate((scipy.array([len(dz)-1]),scipy.arange(0,len(dz))))
    pot = calc.stretch(-q*phi[ :-1],kpSize)
  elif boundaryType in [0,1,3]:
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
  if boundaryType == 0:
    mat = sm.assemble('csr')
    return mat[:,kpSize:-kpSize]
  elif boundaryType == 1:
    mat = sm.assemble('lil')
    mat[:, 1*kpSize: 2*kpSize] = mat[:, 1*kpSize: 2*kpSize]+mat[:,:kpSize ]
    mat[:,-2*kpSize:-1*kpSize] = mat[:,-2*kpSize:-1*kpSize]+mat[:,-kpSize:]
    return mat[:,kpSize:-kpSize].tocsr()
  elif boundaryType == 2:
    mat = sm.assemble('lil')
    mat[:, 1*kpSize: 2*kpSize] = mat[:, 1*kpSize: 2*kpSize]+mat[:,-kpSize:]
    mat[:,-2*kpSize:-1*kpSize] = mat[:,-2*kpSize:-1*kpSize]+mat[:,:kpSize ]
    return mat[:,kpSize:-kpSize].tocsr()
  else:
    mat = sm.assemble('csr')
    return mat
