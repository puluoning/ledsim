''' Base module for LEDSIM.
'''
import scipy, scipy.constants, scipy.linalg, scipy.sparse, scipy.sparse.linalg
import copy, pylab, time

pi       = scipy.pi
c        = scipy.constants.c
q        = scipy.constants.elementary_charge
eV       = scipy.constants.electron_volt
m0       = scipy.constants.electron_mass
kB       = scipy.constants.Boltzmann
hbar     = scipy.constants.hbar
epsilon0 = scipy.constants.epsilon_0

class Access():
  ''' Access class provides attribute and index access to the object dictionary.
  '''
  def __setattr__(self,attr,value):
    ''' Set the specified attribute to the given value. Note that __setattr__
        directly accesses the object dictionary.
    '''
    self.__dict__[attr] = value
  
  def __setitem__(self,attr,value):
    ''' Set the specified attribute to the given value. Note that __setitem__
        merely calls __setattr__ instead of accessing the object dictionary.
        For classes inheriting from Access, it is sufficient to overload
        __setattr__ to change the behavior of both methods.
    '''
    self.__setattr__(attr,value)
   
  def __getitem__(self,attr):
    ''' Get the specified attribute by accessing the object dictionary.
    '''
    return self.__dict__[attr]
  
  def attrs(self):
    ''' Return the keys of the object dictionary.
    '''
    return self.__dict__.keys()

class GenericOpts():
  ''' GenericOpts implements standard methods for options objects, such as
      GridOpts and SolverOpts. __setattr__ is overloaded to prevent setting
      of attributes which are not valid for the class.
  '''
  def __setattr__(self,attr,value):
    ''' Set the specified attribute to the given value, checking to make sure
        that the attribute is valid/allowed.
    '''
    if attr not in self.validAttrs:
      raise AttributeError, 'Opts error: %s is not a valid option' %(attr)
    else:
      self.__dict__[attr] = value
  
  def __str__(self):
    ''' Print the attributes of this object 
    '''
    outStr = ''
    attrWidthMax = max([len(attr) for attr in self.validAttrs]) 
    for attr in self.validAttrs:
      outStr += attr.ljust(attrWidthMax)+' : '+str(self.__dict__[attr])+'\n'
    return outStr

class GridOpts(GenericOpts,Access):
  ''' GridOpts object specifies how the grid is assembled for a given structure.
      Note: units are in meters. GridOpts has the following attributes:
        useFixedGrid : Boolean
          True if fixed gridpoint spacing is to be used. Default value is False.
        dz : float
          Gridpoint spacing if fixed grid is used. Ignored if variable spacing
          is specified. Default value is 5e-10 (0.5 nm)
        dzEdge : float
          Gridpoint spacing at edge of layer, if variable grid is used. Default
          value is 2e-10 (0.2 nm)
        dzCenterFraction : float
          Gridpoint spacing at center of layer=dzCenterFraction*layer thickness
          Default value is 0.04
        dzQuantum : float
          Gridpoint spacing thoughout the quantum layers. Default value is 2e-10
  '''
  validAttrs = ['useFixedGrid','dz','dzEdge','dzCenterFraction','dzQuantum']
  
  def __init__(self,**kwargs):
    ''' Construct the GridOpts object. Keyword input arguments can be used to
        override default values.
    '''
    self.useFixedGrid       = False
    self.dz                 = 5e-10
    self.dzEdge             = 2e-10
    self.dzCenterFraction   = 0.040
    self.dzQuantum          = 2e-10
    for attr, value in kwargs.items():
      self.__setattr__(attr,value)

class ModelOpts(GenericOpts,Access):
  ''' ModelOpts controls models used in simulation. ModelOpts has the
      following attributes:
        T : float
          temperature of the simulation
        polarization: float
          factor which scales total calculated polarization. Can be used to
          turn polarization off, i.e. with a value of 0.
        cBandOffset: float
          fraction of bandgap offset occuring in the conduction band
        dkBulk: float
          dk value used in calculating effective masses
        defect : boolean
          determines whether defect-assisted recombination is enabled
        radiative : boolean
          determines whether radiative recombination is enabled
        auger : boolean
          determines whether Auger recombination is enabled
        quantum : boolean
          determines whether quantum models are used for calculating carrier
          densities
  '''
  validAttrs = ['T','dkBulk','cBandOffset','polarization','defect',
                'radiative','auger','quantum']
  
  def __init__(self,**kwargs):
    ''' Construct the ModelOpts object. Keyword input arguments can be used to
        override default values.
    '''
    self.T                    = 300.
    self.polarization         = 1.
    self.cBandOffset          = 0.7
    self.dkBulk               = 1e9
    self.defect               = True
    self.radiative            = True
    self.auger                = True
    self.quantum              = False
    for attr, value in kwargs.items():
      self.__setattr__(attr,value)

class SolverOpts(GenericOpts,Access):
  ''' SolverOpts controls methods used in the solve module. SolverOpts has the
      following attributes:
        dphi : float
          Increment used in numerical evaluation of derivatives with respect to
          electrostatic potential phi and quasi-potentials phiN and phiP. Units
          are volts; default value is 1e-9 V.
        maxAllowedCorrection : float
          Maximum allowed correction per solver iteration. Units are volts;
          default value is 0.1 V.
        convergenceThreshold : float
          Solution is considered converged once the magnitude of the correction
          is smaller than convergenceThreshold. Units are volts; default value
          is 5e-8 V.
        maxitr : int
          Maximum number of solver iterations per solve attempt. Default value
          is 100.
        maxitrOuter : int
          Maximum number of solve attempts for a bias point. Default is 20.
        Jmin : float
          Minimum current for use of current boundary condition. Units are Amps
          per square meter; default value is 1e-2 A/m**2
        dVmax : float
          Maximum voltage step in voltage ramping. Units are volts; default
          value is 0.5 V. 
        verboseLevel : int 
          Determines level of diagnostic output. Default value is 1.
          1 : no output
          2 : 1 + output in bias wrapper
          3 : 2 + output per call to solver
          4 : 3 + output per solver iteration
  ''' 
  validAttrs = ['dphi','maxAllowedCorrection','convergenceThreshold',
                'maxitr','maxitrOuter','Jmin','dVmax','verboseLevel',
                'quantumConvergenceThreshold']
  
  def __init__(self,**kwargs):
    ''' Construct the SolverOpts object. Keyword input arguments can be used to
        override default values.
    '''
    self.dphi                 = 1e-9
    self.maxAllowedCorrection = 0.1
    self.convergenceThreshold = 5e-8
    self.maxitr               = 100
    self.maxitrOuter          = 20
    self.Jmin                 = 1e-2
    self.dVmax                = 0.5
    self.verboseLevel         = 1
    self.quantumConvergenceThreshold = 1e-4
    for attr, value in kwargs.items():
      self.__setattr__(attr,value)

class QuantumOpts(GenericOpts,Access):
  ''' QuantumOpts controls parameters used in quantum aspects simulation.
      QuantumOpts has the following attributes:
        padLength : float
          Distance beyond the quantum layer where wavefunctions are calculated.
          Default value is 10e-9
        blendLength : float
          Length over which quantum carrier density is blended with classical.
          Default value is 3e-9
        deltaE : float
          Determines the energy beyond band extremum where wavefunctions are
          calculated.
        dk : float
          spacing for calculation of carrier subbands (magnitude of k)
        thetaRes : int
          Number of angular points at given k where wavefunctions are calculated.
  '''
  validAttrs = ['deltaE','deltaEk','dk','thetaRes','padLength','blendLength',
                'quantumType','boundaryType']
  
  def __init__(self,**kwargs):
    ''' Construct the QuantumOpts object. Keyword input arguments can be used to
        override default values.
    '''   
    self.deltaE               = 0.10*eV
    self.deltaEk              = 0.25*eV
    self.dk                   = 2e8
    self.thetaRes             = 1
    self.padLength            = 1e-8 
    self.blendLength          = 3e-9
    self.quantumType          = 'parabolic'
    self.boundaryType         = 'Dirichlet'
    for attr, value in kwargs.items():
      self.__setattr__(attr,value)

class Grid(Access):
  ''' Grid object contains the grid information for a structure. The object has
      the following attributes:
        dz : scipy.array
          Vector of region widths (i.e. gridpoint spacing)
        z : scipy.array
          Vector of gridpoint locations
        zr : scipy.array
          Vector of positions corresponding to the center of gridpoints
        znum : int
          Total number of gridpoints
        rnum : int
          Total number of regions (= znum - 1)
  '''
  def __init__(self,layers,gridOpts):
    ''' Initialize the grid using the given layers and grid options.
    '''
    segments = []
    qStart   =  scipy.inf
    qEnd     = -scipy.inf
    for layer in layers:
      if layer.isQuantum:
        d1 = dn = gridOpts.dzQuantum
        segments += [self.get_dz_segment(d1,dn,layer.thickness)]
        qStart = min(qStart,sum([len(seg) for seg in segments[:-1]]))
        qEnd   = max(qEnd,  sum([len(seg) for seg in segments]))
      elif gridOpts.useFixedGrid:
        d1 = dn = gridOpts.dz
        segments += [self.get_dz_segment(d1,dn,layer.thickness)]
      elif layer.thickness*gridOpts.dzCenterFraction > gridOpts.dzEdge:
        d1 = dn = gridOpts.dzEdge
        dc = gridOpts.dzCenterFraction*layer.thickness
        segments += [self.get_dz_segment(d1,dc,layer.thickness/2),
                     self.get_dz_segment(dc,dn,layer.thickness/2)]
      else:
        d1 = dn = gridOpts.dzEdge
        segments += [self.get_dz_segment(d1,dn,layer.thickness)]
    self.dz       = scipy.concatenate(segments)
    self.z        = scipy.concatenate(([0],scipy.cumsum(self.dz)))
    self.zr       = (self.z[:-1]+self.z[1:])/2
    self.znum     = len(self.z)
    self.rnum     = len(self.zr)
    self.gridOpts = gridOpts
    self.qIndex   = scipy.arange(qStart,qEnd+1)   # Wavefunction index
    self.qrIndex  = scipy.arange(qStart,qEnd)     # Quantum region index   

  def get_dz_segment(self,d1,dn,L):
    ''' Calculate the vector of gridpoint spacing given a segment of length L
        and gridpoint spacings d1,dn specified at the two ends of the segment.
    '''
    if L < 2*max(d1,dn):
      raise ValueError, 'Grid error: gridpoint spacing is changing too rapidly'
    if abs(1.-(d1/dn)) < 1e-10:
      N = round(L/d1)  
      rem = L-d1*N
      delvec = scipy.ones(N)*d1
    else:
      isDecreasing = dn < d1
      if isDecreasing:
        dn, d1 = d1, dn
      a = (1-L/d1)/(dn/d1-L/d1)
      N = scipy.log(dn/d1)/scipy.log(a)
      n = scipy.linspace(0,scipy.floor(N),scipy.floor(N))
      delvec = scipy.concatenate((d1*a**n,[dn]))
      rem = L-sum(delvec)
      if isDecreasing:
        delvec = scipy.flipud(delvec)
    remvec = 1-abs(scipy.linspace(-1,1,len(delvec)))
    return delvec+rem*remvec/sum(remvec)
  
  def get_index(self,zmin,zmax):
    ''' Find the starting and stopping indices of regions within the bounds
        specified by zmin and zmax.
    '''
    match = ((zmin < self.zr)*(self.zr < zmax)).tolist()
    startIndex = match.index(True)
    stopIndex  = len(match)-match[::-1].index(True)
    return startIndex,stopIndex

class Structure(Access):
  ''' Structure class gives all the parameters of the structure which are
      bias-independent.
  '''
  def __init__(self,grid,mat,substrate):
    ''' The structure must be created with grid, material, and substrate
        as input arguments.
    '''
    self.grid       = grid
    self.material   = mat
    self.substrate  = substrate
    
  def __getattribute__(self,attr):
    ''' If the structure object already has the requested attribute, return it;
        otherwise, call __getattr__.
    '''
    if attr in self.attrs():
      return self[attr]
    else:
      return self.__getattr__(attr)
    
  def __getattr__(self,attr):
    ''' Refer to the material object for attributes that are not already
        part of the structure. If the material does not provide a means for
        calculating the attribute, raise an AttributeError.
    '''
    if attr in self.material.attrSwitch.keys():
      if self.material.attrSwitch[attr] == self.material.overrideAttrsMethod:
        self.material.attrSwitch[attr](self)
      else:
        self.material.attrSwitch[attr](self,self.substrate)
      if attr in self.attrs():
        return self[attr]
      else:
        raise AttributeError, attr
    else:
      raise AttributeError, attr
    
  def valid_attr_list(self):
    ''' Return the list of valid attributes for the structure.
    '''
    return self.material.attrSwitch.keys()

class Layer(Access):
  ''' Layer object. The Layer object works in conjunction with the build
      method to form a simple way to generate structures. A layer has a
      material type, thickness, and an isQuantum flag, which determines  
      whether wavefunctions are calculated within the layer. In addition,
      the layer can have any properties that the material type may have,
      e.g. composition, dopant concentration, etc.
  '''
  layerAttrs = ['material','thickness','isQuantum']
  
  def __init__(self,material,thickness,**kwargs):
    ''' Initialize the layer object. If isQuantum is not specified, a value of
        False is assumed. Material attributes are checked against the material
        type specification, and default values are used where none is specified.
    '''
    self.material  = material
    self.thickness = thickness
    self.isQuantum = kwargs['isQuantum'] if 'isQuantum' in kwargs.keys() else False
    for attr, spec in material.layerAttrs.items():
      self[attr] = spec['defaultValue']
    for attr, spec in material.subAttrs.items():
      self[attr] = spec['defaultValue']
    for attr, value in kwargs.items():
      if attr in self.attrs():
        self[attr] = value
      else:
        raise AttributeError, '%s is not a valid attribute for %s' %(attr,material)

def build(layers,substrate=None,gridOpts=GridOpts(),modelOpts=ModelOpts(),quantumOpts=QuantumOpts()):
  ''' Build takes a list of layers and creates a structure; the equilibrium
      condition is calculated and returned. Optionally, the substrate can
      be specified, and grid options and model options can be supplied. Also,
      solverOpts used in calculation of equilibrium condition can be provided.
  '''
  def diffuse(a,dz,Ld,f=0.2):
    b = scipy.copy(a)
    if Ld != 0:
      T = 1.
      D = Ld**2/(4*T)
      rnum   = len(dz)
      dtmax  = 0.25/max(D/dz**2)
      # nsteps = scipy.ceil(rnum*f)
      nsteps = scipy.ceil(T/dtmax)
      dt     = T/nsteps;
      ind1   = range(1,rnum)+[rnum-1]
      ind2   = [0]+range(0,rnum-1)
      for ii in scipy.arange(0,nsteps):
        b[1:-1] = (b+dt/(dz*(dz+dz[ind1]))*2*D*(b[ind1]-b)+ \
                     dt/(dz*(dz+dz[ind2]))*2*D*(b[ind2]-b))[1:-1]
    return b
    
  if substrate == None:
    substrate = layers[0]  
  if [layer.material == substrate.material for layer in layers] != [True]*len(layers):
    raise ValueError, 'Build error: all layers must share the same material'
  mat = substrate.material
  sub = Structure(None,mat,None)
  s = Structure(Grid(layers,gridOpts),mat,sub)
  s.modelOpts = modelOpts
  s.quantumOpts = quantumOpts
  for attr in mat.subAttrs.keys():
    s[attr] = scipy.ones(s.grid.rnum)*substrate[attr]
    sub[attr] = substrate[attr]
  for attr in mat.layerAttrs.keys():
    vec = scipy.zeros(s.grid.rnum)
    zmin = 0.
    for layer in layers:
      zmax = zmin+layer.thickness
      startIndex,stopIndex = s.grid.get_index(zmin,zmax)
      vec[startIndex:stopIndex] = layer[attr]
      zmin = zmax
    Ld = mat.layerAttrs[attr]['diffusionLength']
    s[attr] = diffuse(vec,s.grid.dz,Ld)
    sub[attr] = substrate[attr]
  return s