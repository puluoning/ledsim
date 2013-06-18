''' Material module which contains material properties.
'''
import scipy, scipy.constants, scipy.linalg

pi       = scipy.pi
q        = scipy.constants.elementary_charge
eV       = scipy.constants.electron_volt
m0       = scipy.constants.electron_mass
kB       = scipy.constants.Boltzmann
hbar     = scipy.constants.hbar
epsilon0 = scipy.constants.epsilon_0

class AlGaInN():
  ''' Class for the AlGaInN material system. The AlGaInN class has the following
      parameters:
        layerAttrs : a collection of attributes that a layer of AlGaInN
          must have (e.g. composition, dopant density, etc.). In a
          structure, each of these will generally vary spatially. A diffusion
          length is given for each attribute, which characterizes the typical 
          length scale for variation in each parameter.
        subAttrs : a collection of attributes that an AlGaInN structure will
          inherit from its substrate. This includes e.g. dislocation density.
        attrSwitch: dictionary which specifies how attributes of the material
          are calculated, e.g. strained bandgap, lattice constants,
          polarization, etc.
        overrideAttrsMethod: material attributes calculated by this method may
          be overridden using the overrideAttrs input argument to __init__.
  '''
  
  def __init__(self,layerAttrs={},subAttrs={},overrideAttrs={}):
    ''' Initialize the material class, modifying layerAttrs and subAttrs as
        specified by the user. overrideAttrs specifies how basic parameters
        may be overridden.
    '''      
    # Set default layerAttrs
    # x : Al mole fraction
    # y : In mole fraction
    # Na : acceptor density
    # Nd : donor density
    # relaxation : degree of strain relaxation in the layer
    # Ndef : point defect (nonradiative center) density in the layer
    self.layerAttrs = \
      {'x'          : {'defaultValue':0.      ,'diffusionLength':2e-10},
       'y'          : {'defaultValue':0.      ,'diffusionLength':2e-10},
       'Na'         : {'defaultValue':0.      ,'diffusionLength':8e-10},
       'Nd'         : {'defaultValue':0.      ,'diffusionLength':8e-10},
       'relaxation' : {'defaultValue':0.      ,'diffusionLength':2e-10},
       'Ndef'       : {'defaultValue':1e17*1e6,'diffusionLength':2e-10}}
    
    # Set default subAttrs
    # phi : 
    # theta :
    self.subAttrs = \
      {'phi'        : {'defaultValue':0.},
       'theta'      : {'defaultValue':0.}}
    
    self.overrideAttrsMethod = self.get_basic_params
    
    # Dictionary that specifies how additional material parameters are
    # calculated.
    self.attrSwitch = \
      {'alc0'       : self.get_basic_params,
       'clc0'       : self.get_basic_params,
       'Eg0'        : self.get_basic_params,
       'delcr'      : self.get_basic_params,
       'delso'      : self.get_basic_params,
       'mepara'     : self.get_basic_params,
       'meperp'     : self.get_basic_params,
       'A1'         : self.get_basic_params,
       'A2'         : self.get_basic_params,
       'A3'         : self.get_basic_params,
       'A4'         : self.get_basic_params,
       'A5'         : self.get_basic_params,
       'A6'         : self.get_basic_params,
       'a1'         : self.get_basic_params,
       'a2'         : self.get_basic_params,
       'D1'         : self.get_basic_params,
       'D2'         : self.get_basic_params,
       'D3'         : self.get_basic_params,
       'D4'         : self.get_basic_params,
       'D5'         : self.get_basic_params,
       'D6'         : self.get_basic_params,
       'C11'        : self.get_basic_params,
       'C12'        : self.get_basic_params,
       'C13'        : self.get_basic_params,
       'C33'        : self.get_basic_params,
       'C44'        : self.get_basic_params,
       'd13'        : self.get_basic_params,
       'd33'        : self.get_basic_params,
       'd15'        : self.get_basic_params,
       'Psp'        : self.get_basic_params,
       'ENd'        : self.get_basic_params,
       'ENa'        : self.get_basic_params,
       'Ga'         : self.get_basic_params,
       'Gd'         : self.get_basic_params,
       'epsilon'    : self.get_basic_params,
       'muN0'       : self.get_basic_params,
       'muP0'       : self.get_basic_params,
       'B'          : self.get_basic_params,
       'Cn'         : self.get_basic_params,
       'Cp'         : self.get_basic_params,
       'Ndef'       : self.get_basic_params,
       'DEdef'      : self.get_basic_params,
       'CCSn'       : self.get_basic_params,
       'CCSp'       : self.get_basic_params,
       'epsxx'      : self.get_strain,
       'epsyy'      : self.get_strain,
       'epszz'      : self.get_strain,
       'epsxz'      : self.get_strain,
       'epsyx'      : self.get_strain,
       'Ppz'        : self.get_polarization,
       'Ptot'       : self.get_polarization,
       'Eg'         : self.get_band_params,
       'Ec0'        : self.get_band_params,
       'Ev0'        : self.get_band_params,
       'mex'        : self.get_band_params,
       'mey'        : self.get_band_params,
       'mez'        : self.get_band_params,
       'mhx'        : self.get_band_params,
       'mhy'        : self.get_band_params,
       'mhz'        : self.get_band_params,
       'med'        : self.get_band_params,
       'mhd'        : self.get_band_params,
       'Nc'         : self.get_band_params,
       'Nv'         : self.get_band_params}
    
    # Apply override attributes. Note: only the attrs set by the method given
    # by overrideAttrsMethod can be overridden. This method accepts only
    # the structure as input, not the substrate.
    self.overrideAttrs = {}
    for attr, method in overrideAttrs.items():
      if attr in self.attrSwitch.keys() and \
        self.attrSwitch[attr] == self.get_basic_params:
        self.overrideAttrs[attr] = overrideAttrs[attr]
      else:
        raise AttributeError, '%s is not an attr valid for override' %(attr)
    
    # Apply layerAttrs specified by user
    for attr, spec in layerAttrs.items():
      for key, value in spec.items():
        if attr in self.layerAttrs.keys() and \
          key in self.layerAttrs[attr].keys() and \
          type(value) == type(self.layerAttrs[attr][key]):
          self.layerAttrs[attr][key] = value
        else:
          raise AttributeError, 'Error applying %s:%s' %(attr,key)
    
    # Apply subAttrs specified by user
    for attr, spec in subAttrs.items():
      for key, value in spec.items():
        if attr in self.subAttrs.keys() and \
          key in self.subAttrs[attr].keys() and \
          type(value) == type(self.subAttrs[attr][key]):
          self.subAttrs[attr][key] = value
        else:
          raise AttributeError, 'Error applying %s:%s' %(attr,key)
    
    # dk value for calculation of effective masses
    self.__dkBulk__ = 1e9
    
  def get_basic_params(self,s):
    ''' Basic parameters of the AlInGaN material system, which depend upon
        composition. These are calculated from Vegard's law with bowing
        parameters. Basic parameters may be overridden using the items in
        self.overrideAttrs. Attributes are as follows: 
          alc0,clc0 : unstrained lattice constants
          Eg0 : unstrained bandgap
          delcr,delso : crystal-hole and split-off band energies
          mepara,meperp : electron masses parallel and perpendicular to c axis
          A1-A6 : valence band A-parameters
          a1-a2 : interband deformation potentials
          D1-D6 : valence band deformation potentials
          C11-C44 : elastic moduli
          d13-d15 : tensor elements
          Psp : spontaneous polarization
          ENd : donor activation energy
          ENa : acceptor activation energy
          Gd : donor degeneracy factor
          Ga : acceptor degeneracy factor
          epsilon : dielectric constant
          muN0 : electron mobility
          muP0 : hole mobility
          B  : bimolecular radiative coefficient
          Cn : e-e-h Auger coefficient
          Cp : e-h-h Auger coefficient
          Ndef  : defect density
          DEdef : defect energy relative to mid-gap
          CCSn  : capture cross-section for electrons by defects
          CCSp  : capture cross-section for holes by defects
    '''
    s.alc0    = ( 3.112*s.x+3.189*(1-s.x-s.y)+3.545*s.y)*1e-10
    s.clc0    = ( 4.982*s.x+5.185*(1-s.x-s.y)+5.703*s.y)*1e-10
    s.Eg0     = ( 6.000*s.x+3.437*(1-s.x-s.y)+0.608*s.y\
                 -0.800*(s.x*(1-s.x-s.y))\
                 -1.400*(s.y*(1-s.x-s.y))-3.400*s.x*s.y)*eV
    s.delcr   = (-0.227*s.x+0.010*(1-s.x-s.y)+0.024*s.y)*eV
    s.delso   = ( 0.036*s.x+0.017*(1-s.x-s.y)+0.005*s.y)*eV
    s.mepara  = ( 0.320*s.x+0.210*(1-s.x-s.y)+0.070*s.y)*m0
    s.meperp  = ( 0.300*s.x+0.200*(1-s.x-s.y)+0.070*s.y)*m0
    s.A1      = (-3.860*s.x-7.210*(1-s.x-s.y)-8.210*s.y)
    s.A2      = (-0.250*s.x-0.440*(1-s.x-s.y)-0.680*s.y)
    s.A3      = ( 3.580*s.x+6.680*(1-s.x-s.y)+7.570*s.y)
    s.A4      = (-1.320*s.x-3.460*(1-s.x-s.y)-5.230*s.y)
    s.A5      = (-1.470*s.x-3.400*(1-s.x-s.y)-5.110*s.y)
    s.A6      = (-1.640*s.x-4.900*(1-s.x-s.y)-5.960*s.y)
    s.a1      = (-3.400*s.x-7.100*(1-s.x-s.y)-4.200*s.y)*eV
    s.a2      = (-11.80*s.x-9.900*(1-s.x-s.y)-4.200*s.y)*eV
    s.D1      = (-2.900*s.x-3.600*(1-s.x-s.y)-3.600*s.y)*eV
    s.D2      = ( 4.900*s.x+1.700*(1-s.x-s.y)+1.700*s.y)*eV
    s.D3      = ( 9.400*s.x+5.200*(1-s.x-s.y)+5.200*s.y)*eV
    s.D4      = (-4.000*s.x-2.700*(1-s.x-s.y)-2.700*s.y)*eV
    s.D5      = (-3.300*s.x-2.800*(1-s.x-s.y)-2.800*s.y)*eV
    s.D6      = (-2.700*s.x-4.300*(1-s.x-s.y)-4.300*s.y)*eV
    s.C11     = ( 396.0*s.x+390.0*(1-s.x-s.y)+225.0*s.y)*1e9
    s.C12     = ( 137.0*s.x+145.0*(1-s.x-s.y)+115.0*s.y)*1e9
    s.C13     = ( 108.0*s.x+106.0*(1-s.x-s.y)+92.00*s.y)*1e9
    s.C33     = ( 373.0*s.x+398.0*(1-s.x-s.y)+224.0*s.y)*1e9
    s.C44     = ( 116.0*s.x+105.0*(1-s.x-s.y)+48.00*s.y)*1e9
    s.d13     = (-2.100*s.x-1.000*(1-s.x-s.y)-3.500*s.y)*1e-12
    s.d33     = ( 5.400*s.x+1.900*(1-s.x-s.y)+7.600*s.y)*1e-12
    s.d15     = ( 3.600*s.x+3.100*(1-s.x-s.y)+5.500*s.y)*1e-12
    s.Psp     = (-0.090*s.x-0.034*(1-s.x-s.y)-0.042*s.y\
                 -0.021*(s.x*(1-s.x-s.y))\
                 +0.037*(s.y*(1-s.x-s.y))+0.070*s.x*s.y)
    s.ENd     = ( 0.086*s.x+0.020*(1-s.x-s.y)+0.020*s.y)*eV
    s.ENa     = ( 0.630*s.x+0.170*(1-s.x-s.y)+0.170*s.y)*eV
    s.Ga      = ( 4.000*s.x+4.000*(1-s.x-s.y)+4.000*s.y)
    s.Gd      = ( 4.000*s.x+4.000*(1-s.x-s.y)+4.000*s.y)
    s.epsilon = ( 8.500*s.x+9.700*(1-s.x-s.y)+13.52*s.y)*epsilon0
    s.muN0    = ( 200.0*s.x+200.0*(1-s.x-s.y)+200.0*s.y)*1e-4
    s.muP0    = ( 10.00*s.x+10.00*(1-s.x-s.y)+10.00*s.y)*1e-4
    s.B       = ( 20.00*s.x+24.00*(1-s.x-s.y)+6.600*s.y)*1e-12*1e-6
    s.Cn      = ( 0.010*s.x+0.100*(1-s.x-s.y)+2.500*s.y)*1e-30*1e-12
    s.Cp      = ( 0.010*s.x+0.100*(1-s.x-s.y)+2.500*s.y)*1e-30*1e-12
    s.DEdef   = ( 0.000*s.x+0.000*(1-s.x-s.y)+0.000*s.y)*eV
    s.CCSn    = ( 1.000*s.x+1.000*(1-s.x-s.y)+1.000*s.y)*1e-18*1e-4
    s.CCSp    = ( 1.000*s.x+1.000*(1-s.x-s.y)+1.000*s.y)*1e-18*1e-4
    for attr, method in self.overrideAttrs.items():
      s[attr] = method(s)

  def get_strain(self,s,sub):
    ''' Calculate the strain components. C-axis orientation is assumed.
    '''
    s.epsxx = (1-s.relaxation)*(sub.alc0-s.alc0)/s.alc0
    s.epsyy = (1-s.relaxation)*(sub.alc0-s.alc0)/s.alc0
    s.epszz = (1-s.relaxation)*(-2*s.C13/s.C33*s.epsxx)
    s.epsxy = (1-s.relaxation)*s.epsxx*0.
    s.epsxz = (1-s.relaxation)*s.epsxx*0.

  def get_polarization(self,s,sub):
    ''' Piezoelectric and total polarization.
    '''
    s.Ppz = -4*s.d13*(s.alc0-sub.alc0)/(s.alc0+sub.alc0)*(s.C11+s.C12-2*s.C13**2/s.C33)
    s.Ptot = (s.Ppz+s.Psp)*s.modelOpts.polarization
  
  def get_C_Hc1x1(s,kx,ky):
    ''' Return the C matrices for the conduction band Hamiltonian for the given
        kx and ky values. Also return the kpSize, degeneracy, and angular
        periodicity in the kx/ky plane.
    
        C1 corresponds to type 1 terms (kz C1 kz)
        C2 corresponds to type 2 terms (kz C2   )
        C3 corresponds to type 3 terms (   C3 kz)
        C4 corresponds to type 4 terms (   C4   )
    '''
    kpSize      = 1
    degen       = 2
    thetaPeriod = pi
    C1 = scipy.complex128(scipy.zeros((kpSize,kpSize,s.grid.rnum)))
    C2 = scipy.complex128(scipy.zeros((kpSize,kpSize,s.grid.rnum)))
    C3 = scipy.complex128(scipy.zeros((kpSize,kpSize,s.grid.rnum)))
    C4 = scipy.complex128(scipy.zeros((kpSize,kpSize,s.grid.rnum)))

    C1[0,0,:] = hbar**2/(2*s.meperp)
    C4[0,0,:] = s.Eref+s.Eg0+s.delcr+s.delso/3+hbar**2/(2*s.mepara)*(kx**2+ky**2)+ \
                (s.a1+s.D1)*s.epszz+(s.a2+s.D2)*(s.epsxx+s.epsyy)
    
    return C1, C2, C3, C4, kpSize, degen, thetaPeriod

  def get_C_Hv6x6(cond,kx,ky,boundaryType=0):
    ''' Return the C matrices for the valence band Hamiltonian for the given
        kx and ky values. Also return the kpSize, degeneracy, and angular
        periodicity in the kx/ky plane.
        
        C1 corresponds to type 1 terms (kz C1 kz)
        C2 corresponds to type 2 terms (kz C2   )
        C3 corresponds to type 3 terms (   C3 kz)
        C4 corresponds to type 4 terms (   C4   )
    '''
    kpSize = 6
    degen       = 1
    thetaPeriod = pi
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

    return C1, C2, C3, C4, kpSize, degen, thetaPeriod
  
  def get_band_params(self,s,sub):
    ''' Calculate parameters related to the band structure.
    '''
    dk = s.modelOpts.dkBulk
    
    # Calculate the band energies at k=0, and assign the bandgap, valence
    # band, and conduction band energies to this Structure. The energies are
    # shifted to yield the conduction band/valence band offset ratio specified
    # in modelOpts for this Structure. 
    E0 = self.bulk_bands_calculator(s,sub,0.0,0.0,0.0)
    offset = -E0[1,:]
    E0 = E0+offset
    s.Eg  = E0[0,:]-E0[1,:]
    s.Ec0 = E0[0 ,:]-(s.Eg-s.Eg[0])*(1-s.modelOpts.cBandOffset)
    s.Ev0 = E0[1:,:]-(s.Eg-s.Eg[0])*(1-s.modelOpts.cBandOffset)
    s.Eref = s.Ev0[0,:]+offset
    
    # Calculate the energies of the conduction band and three valence bands at
    # various points in k-space. Then, calculate the carrier effective masses.
    Exm = self.bulk_bands_calculator(s,sub,-dk,0.0,0.0)+offset
    Exp = self.bulk_bands_calculator(s,sub, dk,0.0,0.0)+offset
    Eym = self.bulk_bands_calculator(s,sub,0.0,-dk,0.0)+offset
    Eyp = self.bulk_bands_calculator(s,sub,0.0, dk,0.0)+offset
    Ezm = self.bulk_bands_calculator(s,sub,0.0,0.0,-dk)+offset
    Ezp = self.bulk_bands_calculator(s,sub,0.0,0.0, dk)+offset
    mx  = hbar**2/((Exm-2*E0+Exp)/dk**2)
    my  = hbar**2/((Eym-2*E0+Eyp)/dk**2)
    mz  = hbar**2/((Ezm-2*E0+Ezp)/dk**2)
    
    # Calculate the conduction and valence band carrier effective masses,
    # density of states effective masses, and the band-edge effective density
    # of states and assign them to the Structure. 
    s.mex =  mx[0,:]
    s.mey =  my[0,:]
    s.mez =  mz[0,:]
    s.mhx = -mx[1:,:]
    s.mhy = -my[1:,:]
    s.mhz = -mz[1:,:]
    s.med = (s.mex*s.mey*s.mez)**(1./3);
    s.mhd = (s.mhx*s.mhy*s.mhz)**(1./3);
    s.Nc  = 1/scipy.sqrt(2)*(s.med*kB*s.modelOpts.T/(pi*hbar**2))**(3./2);
    s.Nv  = 1/scipy.sqrt(2)*(s.mhd*kB*s.modelOpts.T/(pi*hbar**2))**(3./2);

  def bulk_bands_calculator(self,s,sub,kx,ky,kz):
    ''' Calculate the band energies for the specified kx, ky, and kz values.
        The 3x3 Hamiltonian for wurtzite crystals is used for the valence,
        while a 1x1 Hamiltonian is used for the conduction band. The model is
        from the chapter by Vurgaftman and Meyer in the book by Piprek. 
    '''
    E = scipy.zeros((4,len(s.Eg0)))   
    E[0,:] = s.Eg0+s.delcr+s.delso/3+\
                hbar**2/(2*s.mepara)*(kx**2+ky**2)+\
                hbar**2/(2*s.meperp)*(kz**2)+\
                (s.a1+s.D1)*s.epszz+(s.a2+s.D2)*(s.epsxx+s.epsyy)
    L = hbar**2/(2*m0)*(s.A1*kz**2+s.A2*(kx+ky)**2)+\
        s.D1*s.epszz+s.D2*(s.epsxx+s.epsyy)
    T = hbar**2/(2*m0)*(s.A3*kz**2+s.A4*(kx+ky)**2)+\
        s.D3*s.epszz+s.D4*(s.epsxx+s.epsyy)
    F = s.delcr+s.delso/3+L+T
    G = s.delcr-s.delso/3+L+T
    K = hbar**2/(2*m0)*s.A5*(kx+1j*ky)**2+s.D5*(s.epsxx-s.epsyy)
    H = hbar**2/(2*m0)*s.A6*(kx+1j*ky)*kz+s.D6*(s.epsxz)
    d = scipy.sqrt(2)*s.delso/3
    for ii in range(len(s.Eg0)):
      mat = scipy.matrix([[    F[ii],     K[ii],       -1j*H[ii]      ],
                          [    K[ii],     G[ii],       -1j*H[ii]+d[ii]],
                          [-1j*H[ii], -1j*H[ii]+d[ii],     L[ii]      ]])
      w,v = scipy.linalg.eig(mat)
      E[1:,ii] = scipy.flipud(scipy.sort(scipy.real(w)))
    return E