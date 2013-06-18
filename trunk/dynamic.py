''' Module for Condition class, which is used to calculate dynamical variables,
    e.g. carrier concentration, which depend upon the electrical bias.
'''
from ledsim import *
import calc, out

class Condition(Access):
  ''' Class for dynamically calculated attributes, i.e. those which depend
      upon the bias. Each condition has an associated structure, and attributes
      of the structure may be referenced directly as though they were
      attributes of the condition.
  '''     
  def __init__(self,struct=None,phi=None,phiN=None,phiP=None,wavefunctions=None):
    ''' Initialize the condition; this requires the structure, the electrostatic
        potential phi, and the quasi-potentials phiN and phiP.
    '''
    self.struct = struct
    self.phi    = phi
    self.phiN   = phiN
    self.phiP   = phiP
    self.wavefunctions = wavefunctions
    self.attrSwitch = \
      {'Efn'         : self.get_band_edges,
       'Efp'         : self.get_band_edges,
       'EcMax'       : self.get_band_edges,
       'EcMin'       : self.get_band_edges,
       'EvMax'       : self.get_band_edges,
       'EvMin'       : self.get_band_edges,
       'Lp'          : self.get_carriers_lr,
       'Ln'          : self.get_carriers_lr,
       'Rp'          : self.get_carriers_lr,
       'Rn'          : self.get_carriers_lr,
       'LNaIonized'  : self.get_carriers_lr,
       'RNaIonized'  : self.get_carriers_lr,
       'LNdIonized'  : self.get_carriers_lr,
       'RNdIonized'  : self.get_carriers_lr,
       'Lprho'       : self.get_rho,
       'Rprho'       : self.get_rho,
       'Lnrho'       : self.get_rho,
       'Rnrho'       : self.get_rho,
       'Lrho'        : self.get_rho,
       'Rrho'        : self.get_rho,
       'prho2D'      : self.get_rho2D,
       'nrho2D'      : self.get_rho2D,
       'rho2D'       : self.get_rho2D,
       'n2D'         : self.get_np2D,
       'p2D'         : self.get_np2D,
       'N'           : self.get_NP,
       'P'           : self.get_NP,
       'Q'           : self.get_NP,
       'nmid'        : self.get_np_mid,
       'pmid'        : self.get_np_mid,
       'muN'         : self.get_mobility,
       'muP'         : self.get_mobility,
       'Dn'          : self.get_mobility,
       'Dp'          : self.get_mobility,
       'cn'          : self.get_c,
       'cp'          : self.get_c,
       'V'           : self.get_iv,
       'J'           : self.get_iv,
       'Jn'          : self.get_iv,
       'Jp'          : self.get_iv,
       'Jleak'       : self.get_iv,
       'Rdef2D'      : self.get_R2D,
       'Rrad2D'      : self.get_R2D,
       'Raug2D'      : self.get_R2D,
       'Rtot2D'      : self.get_R2D,
       'LRdef'       : self.get_recombination_lr,
       'RRdef'       : self.get_recombination_lr,
       'LRrad'       : self.get_recombination_lr,
       'RRrad'       : self.get_recombination_lr,
       'LRaug'       : self.get_recombination_lr,
       'RRaug'       : self.get_recombination_lr,
       'LRtot'       : self.get_recombination_lr,
       'RRtot'       : self.get_recombination_lr,
       'Rdef'        : self.get_R,
       'Rrad'        : self.get_R,
       'Raug'        : self.get_R,
       'Rtot'        : self.get_R,
       'Tndef'       : self.get_nonradiative_lifetimes,
       'Tpdef'       : self.get_nonradiative_lifetimes,
       'Vthn'        : self.get_nonradiative_lifetimes,
       'Vthp'        : self.get_nonradiative_lifetimes}
  
  def __getattribute__(self,attr):
    ''' If the structure object already has the requested attribute, return it;
        otherwise, call __getattr__.
    '''
    if attr in self.attrs():
      return self[attr]
    else:
      return self.__getattr__(attr)
    
  def __getattr__(self,attr): 
    ''' Refer to the structure object for attributes that are not already
        part of the condition. If the structure does not provide a means for
        calculating the attribute, raise an AttributeError.
    '''
    if attr in self.attrSwitch:
      self.attrSwitch[attr]()
      if attr in self.attrs():
        return self[attr]
      else:
        raise AttributeError, attr
    else:
      return self.struct.__getattribute__(attr)
  
  def plot(self,plotType,*args,**kwargs):
    ''' Reference the ledplot method in the out module. It is convenient to call
        ledplot as a method of the condition.
    '''
    out.ledplot(self,plotType,*args,**kwargs)
  
  def valid_attr_list(self):
    ''' Return the list of valid attributes for the condition.
    '''
    return self.attrSwitch.keys()+self.struct.valid_attr_list()
  
  def offset(self,dphi,dphiN,dphiP,keepWavefunctions=True):
    ''' Return a new condition with phi, phiN, and phiP offset by amounts dphi,
        dphiN, and dphiP. This is useful in calculating derivatives and applying
        corrections, e.g. in a solver.
    '''
    if keepWavefunctions:
      return Condition(self.struct,self.phi+dphi,self.phiN+dphiN,self.phiP+dphiP,\
                       self.wavefunctions)
    else:
      return Condition(self.struct,self.phi+dphi,self.phiN+dphiN,self.phiP+dphiP)

  def get_iv(self):
    ''' Get the voltage, total current density, electron and hole current density
        as a function of position, and the total leakage current.
    '''
    self.V = (self.phiN[0]-self.phi[0])-(self.phiN[-1]-self.phi[-1])
    self.J = -q*self.cn[-1]/self.grid.dz[-1]*((self.phiN[-1]-self.phi[-1])-\
                                              (self.phiN[-2]-self.phi[-2]))+\
             -q*self.cp[-1]/self.grid.dz[-1]*((self.phiP[-1]-self.phi[-1])-\
                                              (self.phiP[-2]-self.phi[-2]))
    self.Jn = -q*self.cn/self.grid.dz*((self.phiN[1:]-self.phi[1:])-\
                                       (self.phiN[:-1]-self.phi[:-1]))
    self.Jp = -q*self.cp/self.grid.dz*((self.phiP[1:]-self.phi[1:])-\
                                       (self.phiP[:-1]-self.phi[:-1]))
    if self.Nd[0] > self.Na[0]:
      self.Jleak = self.Jn[-1]+self.Jp[0]
    else:
      self.Jleak = self.Jn[0]+self.Jp[-1]
  
  def get_band_edges(self):
    ''' Calculate the quasi-Fermi level positions.
    '''
    def band_min(E):
      return min(-q*scipy.minimum(self.phi[self.grid.qIndex][:-1],
                    self.phi[self.grid.qIndex][ 1:])+E[self.grid.qrIndex])
    def band_max(E):
      return max(-q*scipy.maximum(self.phi[self.grid.qIndex][:-1],\
                    self.phi[self.grid.qIndex][ 1:])+E[self.grid.qrIndex])
    self.Efn   = q*(self.phiN-self.phi)
    self.Efp   = q*(self.phiP-self.phi)
    self.EcMax = band_max(self.Ec0)
    self.EcMin = band_min(self.Ec0)
    self.EvMax = band_max(self.Ev0[0,:])
    self.EvMin = band_min(self.Ev0[0,:])

  def get_carriers_quantum(self,phi,phiN,phiP):
    ''' Calculate the carrier density at each gridpoint in the quantum region.
    '''         
    def fc(Ef,E):
      return 1./(1+scipy.exp((E[:,scipy.newaxis]-Ef-q*(phiOrig-phi))/(kB*cond.modelOpts.T)))
    def fv(Ef,E):
      return 1.-fc(Ef,E)
    
    Efn = q*(phiN-phi)[self.grid.qIndex]
    phiOrig = self.EcWavefunctions.phiOrig
    n = scipy.zeros(len(self.grid.qIndex))
    for kpt in self.EcWavefunctions.psiDict.values():
      n += scipy.sum(kpt['psisq']*kpt['Vk']*scipy.transpose(fc(Efn,kpt['E'])),1)
    n = n/(4*pi**2)*self.EcWavefunctions.degen
    
    Efp = q*(phiP-phi)[self.grid.qIndex]
    phiOrig = self.EvWavefunctions.phiOrig
    p = scipy.zeros(len(self.grid.qIndex))
    for kpt in self.EvWavefunctions.psiDict.values():
      p += scipy.sum(kpt['psisq']*kpt['Vk']*scipy.transpose(fv(Efp,kpt['E'])),1)
    p = p/(4*pi**2)*self.EvWavefunctions.degen
    return n, p

  def get_carriers(self,phi,phiN,phiP):
    ''' Get the carriers in each region (i.e. between gridpoints) given the
        specified values for phi, phiN, and phiP. Also calculate the ionized
        density of donors and acceptors. Note: the phi, phiN, and phiP for this
        condition are not used. In principle this involves evaluation of the
        Fermi integral. An approximation from the textbook by Pierret is used.
    '''
    kBT = kB*self.modelOpts.T
    etac = (q*phiN-self.Ec0)/kBT
    n    = self.Nc/(scipy.exp(-etac)+3*scipy.sqrt(pi/2)*
                     ((etac+2.13)+(abs(etac-2.13)**2.4+9.6)**(5./12))**(-1.5))
    etav = (self.Ev0-q*phiP)/kBT
    p    = scipy.sum(self.Nv/(scipy.exp(-etav)+3*scipy.sqrt(pi/2)*
                     ((etav+2.13)+(abs(etav-2.13)**2.4+9.6)**(5./12))**(-1.5)),axis=0)
    NdIonized = self.Nd/(1+self.Gd*scipy.exp(( q*phiN-(self.Ec0-self.ENd))/kBT))
    NaIonized = self.Na/(1+self.Ga*scipy.exp((-q*phiP+(self.Ev0[0,:]+self.ENa))/kBT))
    return n, p, NdIonized, NaIonized
  
  def get_carriers_lr(self):
    ''' Calculate the carrier and ionized dopant density to the left and right 
        of each gridpoint, using the get_carriers method.
    '''
    self.Ln = scipy.zeros(self.grid.znum)
    self.Rn = scipy.zeros(self.grid.znum)
    self.Lp = scipy.zeros(self.grid.znum)
    self.Rp = scipy.zeros(self.grid.znum)
    self.LNdIonized = scipy.zeros(self.grid.znum)
    self.RNdIonized = scipy.zeros(self.grid.znum)
    self.LNaIonized = scipy.zeros(self.grid.znum)
    self.RNaIonized = scipy.zeros(self.grid.znum)
    self.Ln[1: ], self.Lp[1: ], self.LNdIonized[1: ], self.LNaIonized[1: ] = \
      self.get_carriers(self.phi[ 1:],self.phiN[ 1:],self.phiP[ 1:])
    self.Rn[:-1], self.Rp[:-1], self.RNdIonized[:-1], self.RNaIonized[:-1] = \
      self.get_carriers(self.phi[:-1],self.phiN[:-1],self.phiP[:-1])
  
  def get_rho(self):
    ''' Calculate the positive, negative, and total charge density to the left 
        and right of each gridpoint.
    '''
    self.Lprho  = q*(self.Lp-self.LNaIonized)
    self.Lnrho  = q*(self.Ln-self.LNdIonized)
    self.Rprho  = q*(self.Rp-self.RNaIonized)
    self.Rnrho  = q*(self.Rn-self.RNdIonized)
    self.Lrho   = self.Lprho-self.Lnrho 
    self.Rrho   = self.Rprho-self.Rnrho

  def get_rho2D(self):
    ''' Get the positive and negative 2D charge density at each gridpoint (by
        integrating over the Voronoi box).
    '''
    self.prho2D = self.Lprho*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.Rprho*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.nrho2D = self.Lnrho*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.Rnrho*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.rho2D  = self.prho2D-self.nrho2D
    
  def get_np2D(self):
    ''' Get the positive and negative 2D electron and hole density at each
        gridpoint (by integrating over the Voronoi box).
    '''
    self.p2D    = self.Lp*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.Rp*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.n2D    = self.Ln*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.Rn*scipy.concatenate([self.grid.dz,[0.0]])/2

  def get_NP(self):
    ''' Calculate the total 2D electron and hole densities as well as total 
        charge density in the structure (by summing the 2D density along its
        entire thickness).
    '''
    self.P = sum(self.p2D)
    self.N = sum(self.n2D)
    self.Q = sum(self.rho2D)
  
  def get_np_mid(self,gdx=1e-3):
    ''' Get the carrier densities between gridpoints using the Scharfetter-
        Gummel discretization method. In the region where direct calculation
        is unreliable, use a multi-term Taylor expansion. For reference see:
        http://www.math.nuk.edu.tw/jinnliu/proj/Device/ScharfGum.pdf
    '''
    xg = (self.phi[1:]-self.phi[:-1])/(kB*self.modelOpts.T/q)
    sr = (xg >= -gdx)*(xg <= gdx)
    gn = scipy.zeros(self.grid.rnum)
    gp = scipy.zeros(self.grid.rnum)
    gn[-sr] = (1-scipy.exp( xg[-sr]/2))/(1-scipy.exp( xg[-sr]))
    gp[-sr] = (1-scipy.exp(-xg[-sr]/2))/(1-scipy.exp(-xg[-sr]))
    gn[ sr] = 0.5-xg[sr]/8+xg[sr]**3/384-xg[sr]**5/15360
    gp[ sr] = 0.5+xg[sr]/8-xg[sr]**3/384+xg[sr]**5/15360
    self.nmid = (1-gn)*self.Rn[:-1]+gn*self.Ln[1: ]
    self.pmid = (1-gp)*self.Lp[1: ]+gp*self.Rp[:-1]
  
  def get_mobility(self):
    ''' Calculate the mobility and diffusivity. Only a basic model at present,
        i.e. muN = muN0 and muP = muP0.
    '''
    self.muN = self.muN0
    self.muP = self.muP0
    
    kBT = kB*self.modelOpts.T
    self.Dn = kBT/q*self.muN
    self.Dp = kBT/q*self.muP   
  
  def get_c(self):
    ''' Calculate the product of carrier density and mobility. This is useful
        for nonequilibrium solvers.
    '''
    self.cn = self.nmid*self.muN
    self.cp = self.pmid*self.muP
  
  def get_recombination(self,phiN,phiP,n,p):
    ''' Get the recombination rates in each region (i.e. between gridpoints) 
        given the specified values for phi, phiN, and phiP. Note: the phi, 
        phiN, and phiP for this condition are not used. 
    '''
    kBT = kB*self.modelOpts.T
    expFactor = 1-scipy.exp(scipy.minimum(phiP-phiN,0.)/(kBT/q))
    Rdef = expFactor*n*p/(p*self.Tndef+n*self.Tpdef)*self.modelOpts.defect
    Rrad = expFactor*n*p*self.B*self.modelOpts.radiative
    Raug = expFactor*n*p*(n*self.Cn+p*self.Cp)*self.modelOpts.auger
    Rtot = Rdef+Rrad+Raug
    return Rdef,Rrad,Raug,Rtot
    
  def get_recombination_lr(self):
    ''' Calculate the various recombination rates to the left and right of each 
        gridpoint, using the get_recombination method.
    '''
    self.LRdef = scipy.zeros(self.grid.znum)
    self.RRdef = scipy.zeros(self.grid.znum)
    self.LRrad = scipy.zeros(self.grid.znum)
    self.RRrad = scipy.zeros(self.grid.znum)
    self.LRaug = scipy.zeros(self.grid.znum)
    self.RRaug = scipy.zeros(self.grid.znum)
    self.LRtot = scipy.zeros(self.grid.znum)
    self.RRtot = scipy.zeros(self.grid.znum)
    self.LRdef[1: ],self.LRrad[1: ],self.LRaug[1: ],self.LRtot[1: ] = \
      self.get_recombination(self.phiN[ 1:],self.phiP[ 1:],self.Ln[ 1:],self.Lp[ 1:])
    self.RRdef[:-1],self.RRrad[:-1],self.RRaug[:-1],self.RRtot[:-1] = \
      self.get_recombination(self.phiN[:-1],self.phiP[:-1],self.Rn[:-1],self.Rp[:-1])
        
  def get_R2D(self):
    ''' Get the integrated recombination rate per unit cross-sectional area at
        each gridpoint (by integrating recombination over the Voronoi box).
    '''
    self.Rdef2D = self.LRdef*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.RRdef*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.Rrad2D = self.LRrad*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.RRrad*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.Raug2D = self.LRaug*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.RRaug*scipy.concatenate([self.grid.dz,[0.0]])/2
    self.Rtot2D = self.LRtot*scipy.concatenate([[0.0],self.grid.dz])/2+ \
                  self.RRtot*scipy.concatenate([self.grid.dz,[0.0]])/2
  
  def get_R(self):
    ''' Calculate the total recombination rate in the structure (by summing the 
        2D rate along its entire thickness).
    '''
    self.Rdef = sum(self.Rdef2D)
    self.Rrad = sum(self.Rrad2D)
    self.Raug = sum(self.Raug2D)
    self.Rtot = sum(self.Rtot2D)
    
  def get_nonradiative_lifetimes(self):
    ''' Get the lifetimes for defect-assisted nonradiative (SRH) recombination.
        See http://www.iue.tuwien.ac.at/phd/ayalew/node72.html for an online
        reference.
    '''
    def m_eff(mx,my,mz):
      m = 3/(1/mx+1/my+1/mz)
      return m if scipy.ndim(m) == 1 else m[0,:] 
             
    kBT = kB*self.modelOpts.T
    self.Vthn  = scipy.sqrt(3*kBT/m_eff(self.mex,self.mey,self.mez))
    self.Vthp  = scipy.sqrt(3*kBT/m_eff(self.mhx,self.mhy,self.mhz))
    self.Tndef = 1/(self.Ndef*self.CCSn*self.Vthn)
    self.Tpdef = 1/(self.Ndef*self.CCSp*self.Vthp)
