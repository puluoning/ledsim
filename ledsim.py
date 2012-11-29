import scipy
import calc
from material import *

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
      GridOpts and SolverOpts. Inherits from Access to provide attribute
      and index access to object attributes. __setattr__ is overloaded to
      prevent setting of attributes which are not valid for the class.
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
          True if a fixed gridpoint spacing is to be used. Default value is False.
        dz : float
          Gridpoint spacing if fixed grid is used. Ignored if variable spacing
          is specified. Default value is 5e-10 (0.5 nm)
        dzEdge : float
          Gridpoint spacing at edge of region, if variable grid is used. Default
          value is 2e-10 (0.2 nm)
        dzCenterFraction : float
          Gridpoint spacing at center of region = dzCenterFraction * region thickness
          Default value is 0.04
        dzQuantum : float
          Gridpoint spacing thoughout the quantum regions. Default value is 2e-10
        quantumPadLength : float
          Distance beyond the quantum region where wavefunctions are calculated.
          Default value is 10e-9
        quantumBlendLength : float
          Length over which quantum carrier densities are blended with classical.
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
          Maximum number of solve attempts for a bias point. Default value is 20.
        Jmin : float
          Minimum current for use of current boundary condition. Units are Amps
          per square meter; default value is 1e-2 A/m**2
        dVmax : float
          Maximum voltage step in voltage ramping. Units are volts; default value
          is 0.5 V. 
        verboseLevel : int 
          Determines level of diagnostic output. Default value is 1.
          1 : no output
          2 : 1 + output in bias wrapper
          3 : 2 + output per call to solver
          4 : 3 + output per solver iteration
  ''' 
  validAttrs = ['dphi','maxAllowedCorrection','convergenceThreshold',
                'maxitr','maxitrOuter','Jmin','dVmax','verboseLevel']
  
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
    for attr, value in kwargs.items():
      self.__setattr__(self,attr,value)

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
    ''' Calculate the vector of gridpoint spacing given a segment of length L and
        gridpoint spacings d1,dn specified at the two ends of the segment.
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

class Layer(Access):
  ''' Layer object. A layer has a material type, a thickness, and an isQuantum
      flag, which determines whether wavefunctions are calculated within the
      layer. In addition, the layer can have any properties that the material
      type may have (e.g. composition, dopant concentration, etc.).
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
    for attr, spec in material.matAttrs.items():
      self[attr] = spec['defaultValue']
    for attr, value in kwargs.items():
      if attr in material.matAttrs.keys() or attr in layerAttrs:
        self[attr] = value
      else:
        raise AttributeError, '%s is not a valid attribute for %s' %(attr,material)

class Structure(Access):
  
  def __init__(self):
    pass
  
  def is_complete(self):
    pass

def build(layers,substrate=None,gridOpts=GridOpts(),buildOpts=None):
  '''
  '''
  def get_index(zmin,zmax,grid):
    match = ((zmin < grid.zr)*(grid.zr < zmax)).tolist()
    startIndex = match.index(True)
    stopIndex  = len(match)-match[::-1].index(True)
    return startIndex,stopIndex
  
  mat = layers[0].material
  if [layer.material == mat for layer in layers] != [True]*len(layers):
    raise ValueError, 'Build error: incompatible materials'
  if substrate == None:
    substrate = layers[0]
    
  s = Structure()
  s.grid = Grid(layers,gridOpts)
  for attr in mat.matAttrs.keys():
    vec = scipy.zeros(s.grid.rnum)
    zmin = 0.
    for layer in layers:
      zmax = zmin+layer.thickness
      startIndex,stopIndex = get_index(zmin,zmax,s.grid)
      vec[startIndex:stopIndex] = layer[attr]
      zmin = zmax
    Ld = mat.matAttrs[attr]['diffusionLength']
    s[attr] = calc.diffuse(vec,s.grid.dz,Ld)
  
  kwargs = dict([(attr,s[attr]) for attr in s.attrs()])
  for attr in mat.calcAttrs:
    s[attr] = mat.calcAttrs[attr](kwargs)

  print kwargs
  
    
  return s

if __name__ == '__main__':  

  layers = [Layer(AlGaInN,thickness=10e-9,x=0.0,y=0.0),
            Layer(AlGaInN,thickness=10e-9,x=0.2,y=0.0),
            Layer(AlGaInN,thickness=10e-9,x=0.0,y=0.0)]

  s = build(layers)