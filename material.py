import ledsim, scipy, scipy.constants
        
q        = scipy.constants.elementary_charge
eV       = scipy.constants.electron_volt
m0       = scipy.constants.electron_mass
kB       = scipy.constants.Boltzmann
hbar     = scipy.constants.hbar
epsilon0 = scipy.constants.epsilon_0
        
class AlGaInN(Wurtzite):
  ''' Class AlGaInN represents the III-Nitride material system. A layer of AlInGaN
      has certain fundamental layer-dependent attributes; these are:
        x  : Al mole fraction
        y  : In mole fraction
        Na : acceptor density
        Nd : donor density
        relaxation : degree of strain relaxation in the layer
        Ndef : point defect (nonradiative center) density in the layer
      In a structure consisting of multiple distinct layers, these attributes do
      not vary on a subatomic scale; instead, they are assumed to change gradually
      with a profile determined via diffusion with a given diffusion length. Both
      the default values and diffusion length for fundamental layer-dependent
      attributes are stored in the dictionary layerAttrs.
      
      In addition to layer-dependent properties, certain layer properties are
      inherited from the substrate. These are:
        Ndis : threading dislocation density in the layer
        T  : temperature
      and are stored in the subAttrs variable.
  '''
  layerAttrs = {'x'           : {'defaultValue':0.      ,'diffusionLength':2e-10},
                'y'           : {'defaultValue':0.      ,'diffusionLength':2e-10},
                'Na'          : {'defaultValue':0.      ,'diffusionLength':2e-9 },
                'Nd'          : {'defaultValue':0.      ,'diffusionLength':2e-9 },
                'relaxation'  : {'defaultValue':0.      ,'diffusionLength':2e-10},
                'Ndef'        : {'defaultValue':1e17*1e6,'diffusionLength':2e-10}}
  subAttrs   = {'Ndis'        : {'defaultValue':5e08*1e4},
                'T'           : {'defaultValue':300     }}
  
  def calc_attrs(self,x,y,Na,Nd,relaxation,Ndef,Ndis,T):
    ''' Fundamental material constants of III-Nitride alloys which depend upon
        the layer compositions/attributes. Input arguments are all layer and
        substrate attributes. The calculated attributes are:
          alc0,clc0 : lattice constants
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
          DEdis : dislocation energy relative to mid-gap
          CCSn  : capture cross-section for electrons by defects
          CCSp  : capture cross-section for holes by defects
    '''
    return {'alc0'    : ( 3.112*x+3.189*(1-x-y)+3.545*y)*1e-10,
            'clc0'    : ( 4.982*x+5.185*(1-x-y)+5.703*y)*1e-10,
            'Eg0'     : ( 6.000*x+3.437*(1-x-y)+0.608*y-\
                          0.800*x*(1-x-y)-1.400*y*(1-x-y)-3.400*x*y)*eV,
            'delcr'   : (-0.227*x+0.010*(1-x-y)+0.024*y)*eV,
            'delso'   : ( 0.036*x+0.017*(1-x-y)+0.005*y)*eV,
            'mepara'  : ( 0.320*x+0.210*(1-x-y)+0.070*y)*m0,
            'meperp'  : ( 0.300*x+0.200*(1-x-y)+0.070*y)*m0,
            'A1'      : (-3.860*x-7.210*(1-x-y)-8.210*y),
            'A2'      : (-0.250*x-0.440*(1-x-y)-0.680*y),
            'A3'      : ( 3.580*x+6.680*(1-x-y)+7.570*y),
            'A4'      : (-1.320*x-3.460*(1-x-y)-5.230*y),
            'A5'      : (-1.470*x-3.400*(1-x-y)-5.110*y),
            'A6'      : (-1.640*x-4.900*(1-x-y)-5.960*y),
            'a1'      : (-3.400*x-7.100*(1-x-y)-4.200*y)*eV,
            'a2'      : (-11.80*x-9.900*(1-x-y)-4.200*y)*eV,
            'D1'      : (-2.900*x-3.600*(1-x-y)-3.600*y)*eV,
            'D2'      : ( 4.900*x+1.700*(1-x-y)+1.700*y)*eV,
            'D3'      : ( 9.400*x+5.200*(1-x-y)+5.200*y)*eV,
            'D4'      : (-4.000*x-2.700*(1-x-y)-2.700*y)*eV,
            'D5'      : (-3.300*x-2.800*(1-x-y)-2.800*y)*eV,
            'D6'      : (-2.700*x-4.300*(1-x-y)-4.300*y)*eV,
            'C11'     : ( 396.0*x+390.0*(1-x-y)+225.0*y)*1e9,
            'C12'     : ( 137.0*x+145.0*(1-x-y)+115.0*y)*1e9,
            'C13'     : ( 108.0*x+106.0*(1-x-y)+92.00*y)*1e9,
            'C33'     : ( 373.0*x+398.0*(1-x-y)+224.0*y)*1e9,
            'C44'     : ( 116.0*x+105.0*(1-x-y)+48.00*y)*1e9,
            'd13'     : (-2.100*x-1.000*(1-x-y)-3.500*y)*1e-12,
            'd33'     : ( 5.400*x+1.900*(1-x-y)+7.600*y)*1e-12,
            'd15'     : ( 3.600*x+3.100*(1-x-y)+5.500*y)*1e-12,
            'Psp'     : (-0.090*x-0.034*(1-x-y)-0.042*y-0.021*x+\
                         (1-x-y)+0.037*y*(1-x-y)+0.070*x*y),
            'ENd'     : ( 0.086*x+0.020*(1-x-y)+0.020*y)*eV,
            'ENa'     : ( 0.630*x+0.170*(1-x-y)+0.170*y)*eV,
            'Ga'      : ( 4.000),
            'Gd'      : ( 2.000),
            'epsilon' : ( 8.500*x+9.700*(1-x-y)+13.52*y)*epsilon0,
            'muN0'    : ( 200.0*x+200.0*(1-x-y)+200.0*y)*1e-4,
            'muP0'    : ( 10.00*x+10.00*(1-x-y)+10.00*y)*1e-4,
            'B'       : ( 20.00*x+24.00*(1-x-y)+6.600*y)*1e-12*1e-6,
            'Cn'      : ( 0.000*x+0.000*(1-x-y)+2.500*y)*1e-30*1e-12,
            'Cp'      : ( 0.000*x+0.000*(1-x-y)+2.500*y)*1e-30*1e-12,
            'Ndef'    : ( 5.000*x+5.000*(1-x-y)+5.000*y)*1e18*1e-6,
            'DEdef'   : ( 0.000*x+0.000*(1-x-y)+0.000*y)*eV,
            'DEdis'   : ( 0.000*x+0.000*(1-x-y)+0.000*y)*eV,
            'CCSn'    : ( 1.000*x+1.000*(1-x-y)+1.000*y)*1e-18*1e-4,
            'CCSp'    : ( 1.000*x+1.000*(1-x-y)+1.000*y)*1e-18*1e-4}

class Wurtzite():
  
  def get_derived_attrs(self,substrate,struct):
    pass