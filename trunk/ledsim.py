''' Base module for LEDSIM.
'''
import scipy, pylab, calc, solve

pi       = scipy.pi
q        = scipy.constants.elementary_charge
eV       = scipy.constants.electron_volt
m0       = scipy.constants.electron_mass
kB       = scipy.constants.Boltzmann
hbar     = scipy.constants.hbar
epsilon0 = scipy.constants.epsilon_0

class GridOpts(calc.GenericOpts,calc.Access):
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
        quantumPadLength : float
          Distance beyond the quantum layer where wavefunctions are calculated.
          Default value is 10e-9
        quantumBlendLength : float
          Length over which quantum carrier density is blended with classical.
          Default value is 3e-9
  '''
  validAttrs = ['useFixedGrid','dz','dzEdge','dzCenterFraction',
                'dzQuantum','quantumPadLength','quantumBlendLength']
  
  def __init__(self,**kwargs):
    ''' Construct the GridOpts object. Keyword input arguments can be used to
        override default values.
    '''
    self.useFixedGrid       = False
    self.dz                 = 5e-10
    self.dzEdge             = 2e-10
    self.dzCenterFraction   = 0.040
    self.dzQuantum          = 2e-10
    self.quantumPadLength   = 1e-8 
    self.quantumBlendLength = 3e-9
    for attr, value in kwargs.items():
      self.__setattr__(attr,value)

class ModelOpts(calc.GenericOpts,calc.Access):
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
        quantumDeltaE : float
          Determines the energy beyond band extremum where wavefunctions are
          calculated.
        dkQuantum : float
          dk value for quantum calculation of carrier density
  '''
  validAttrs = ['T','dkBulk','cBandOffset','polarization','defect',
                'radiative','auger','quantumDeltaE','dkQuantum']
  
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
    self.quantumDeltaE        = 0.1*eV
    self.dkQuantum            = 3e7
    for attr, value in kwargs.items():
      self.__setattr__(self,attr,value)

class Grid(calc.Access):
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
    for layer in layers:
      if layer.isQuantum:
        d1 = dn = gridOpts.dzQuantum
        segments += [self.get_dz_segment(d1,dn,layer.thickness)]
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

class Structure(calc.Access):
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

class Layer(calc.Access):
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

def get_index(zmin,zmax,grid):
  ''' Find the starting and stopping indices of regions within the bounds
      specified by zmin and zmax.
  '''
  match = ((zmin < grid.zr)*(grid.zr < zmax)).tolist()
  startIndex = match.index(True)
  stopIndex  = len(match)-match[::-1].index(True)
  return startIndex,stopIndex

def build(layers,substrate=None,gridOpts=GridOpts(),\
          modelOpts=ModelOpts(),solverOpts=solve.SolverOpts()):
  ''' Build takes a list of layers and creates a structure; the equilibrium
      condition is calculated and returned. Optionally, the substrate can
      be specified, and grid options and model options can be supplied. Also,
      solverOpts used in calculation of equilibrium condition can be provided.
  '''
  if substrate == None:
    substrate = layers[0]  
  if [layer.material == substrate.material for layer in layers] != [True]*len(layers):
    raise ValueError, 'Build error: all layers must share the same material'
  mat = substrate.material
  sub = Structure(None,mat,None)
  s = Structure(Grid(layers,gridOpts),mat,sub)
  s.modelOpts = modelOpts
  for attr in mat.subAttrs.keys():
    s[attr] = scipy.ones(s.grid.rnum)*substrate[attr]
    sub[attr] = substrate[attr]
  for attr in mat.layerAttrs.keys():
    vec = scipy.zeros(s.grid.rnum)
    zmin = 0.
    for layer in layers:
      zmax = zmin+layer.thickness
      startIndex,stopIndex = get_index(zmin,zmax,s.grid)
      vec[startIndex:stopIndex] = layer[attr]
      zmin = zmax
    Ld = mat.layerAttrs[attr]['diffusionLength']
    s[attr] = calc.diffuse(vec,s.grid.dz,Ld)
    sub[attr] = substrate[attr]
  c1 = solve.solve_equilibrium_local(s,solverOpts=solverOpts)
  c2 = solve.solve_equilibrium(c1,solverOpts=solverOpts)
  return c2