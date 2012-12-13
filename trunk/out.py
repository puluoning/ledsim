''' Plotting methods.
'''
import scipy, pylab

pi       = scipy.pi
q        = scipy.constants.elementary_charge
eV       = scipy.constants.electron_volt
m0       = scipy.constants.electron_mass
kB       = scipy.constants.Boltzmann
hbar     = scipy.constants.hbar
epsilon0 = scipy.constants.epsilon_0

def ledplot(cond,plotType,isUseSmoothing=True,figNum=None):
  ''' Generate plots of type specified by the user given the condition.
      Optionally use smoothing (on by default). Supported plot types are:
        bands: band diagram, including conduction band, valence band, and the
          quasi-Fermi levels for electrons and holes.
        carriers: electron and hole densities, dopant densities and ionized
          dopant densities with a logarithmic y-axis.
        recombination: recombination rates for various mechanisms as a
          function of position with a logarithmic y-axis.
        current: electron and hole current density as a function of position.
  '''
  def interp(a):
    return 0.5*(scipy.hstack((a[0],a))+scipy.hstack((a,a[-1])))
  
  def interp_lr(La,Ra):
    return scipy.hstack((Ra[0],0.5*(La[1:-1]+Ra[1:-1]),La[-1]))
  
  def stretch(a,step):
    return a[scipy.sort(range(len(a))*step)]
  
  def interleave(a,b):
    c = scipy.zeros(2*len(a))
    c[ ::2] = a
    c[1::2] = b
    return c
  
  # Select the figure if one is specified.
  if figNum is not None:
    pylab.figure(figNum)
  
  # Plot the appropriate plotType.
  if plotType == 'bands':
    if isUseSmoothing:
      zp = cond.grid.z*1e9
      EcPlot  = -cond.phi+interp(cond.Ec0)/eV
      EvPlot  = -cond.phi+interp(cond.Ev0[0,:])/eV
      EfnPlot =  cond.phiN-cond.phi
      EfpPlot =  cond.phiP-cond.phi
    else:
      zp = stretch(cond.grid.z,2)[1:-1]*1e9
      EcPlot  = -stretch(cond.phi,2)[1:-1]+stretch(cond.Ec0,2)/eV 
      EvPlot  = -stretch(cond.phi,2)[1:-1]+stretch(cond.Ev0[0,:],2)/eV
      EfnPlot =  stretch(cond.phiN-cond.phi,2)[1:-1]
      EfpPlot =  stretch(cond.phiP-cond.phi,2)[1:-1]
    pylab.plot(zp,EcPlot,'k',zp,EvPlot,'k',zp,EfnPlot,'r',zp,EfpPlot,'b')
    pylab.xlim([0,zp[-1]])
    pylab.ylabel('Energy [eV]')
    pylab.xlabel('Position [nm]')
    pylab.legend(['Ec','Ev','Efn','Efp'])
    
  elif plotType == 'carriers':
    if isUseSmoothing:
      zp = cond.grid.z*1e9
      nPlot = interp_lr(cond.Ln,cond.Rn)*1e-6
      pPlot = interp_lr(cond.Lp,cond.Rp)*1e-6
      NdPlot = interp(cond.Nd)*1e-6
      NaPlot = interp(cond.Na)*1e-6
      NdIonizedPlot = interp_lr(cond.LNdIonized,cond.RNdIonized)*1e-6
      NaIonizedPlot = interp_lr(cond.LNaIonized,cond.RNaIonized)*1e-6
    else:
      zp = stretch(cond.grid.z,2)[1:-1]*1e9
      nPlot = interleave(cond.Ln,cond.Rn)[1:-1]*1e-6
      pPlot = interleave(cond.Lp,cond.Rp)[1:-1]*1e-6
      NdPlot = stretch(cond.Nd,2)*1e-6
      NaPlot = stretch(cond.Na,2)*1e-6
      NdIonizedPlot = interleave(cond.LNdIonized,cond.RNdIonized)[1:-1]*1e-6
      NaIonizedPlot = interleave(cond.LNaIonized,cond.RNaIonized)[1:-1]*1e-6

    pylab.semilogy(zp,nPlot,'r',zp,NdPlot,'r:',zp,NdIonizedPlot,'r--',\
                   zp,pPlot,'b',zp,NaPlot,'b:',zp,NaIonizedPlot,'b--')
    maxVal = max([max(nPlot),max(NdPlot),max(NdIonizedPlot),\
                  max(pPlot),max(NaPlot),max(NaIonizedPlot)])
    pylab.xlim([0,zp[-1]])
    pylab.ylim([1e10,10**scipy.ceil(scipy.log10(maxVal))])
    pylab.ylabel('Density [1/cm3]')
    pylab.xlabel('Position [nm]')
    pylab.legend(['n','Nd','Nd+','p','Na','Na-'])
    
  elif plotType == 'recombination':
    if isUseSmoothing:
      zp = cond.grid.z*1e9
      RdefPlot = interp_lr(cond.LRdef,cond.RRdef)*1e-6
      RradPlot = interp_lr(cond.LRrad,cond.RRrad)*1e-6
      RaugPlot = interp_lr(cond.LRaug,cond.RRaug)*1e-6
      RtotPlot = interp_lr(cond.LRtot,cond.RRtot)*1e-6
    else:
      zp = stretch(cond.grid.z,2)[1:-1]*1e9
      RdefPlot = interleave(cond.LRdef,cond.RRdef)[1:-1]*1e-6
      RradPlot = interleave(cond.LRrad,cond.RRrad)[1:-1]*1e-6
      RaugPlot = interleave(cond.LRaug,cond.RRaug)[1:-1]*1e-6
      RaugPlot = interleave(cond.LRtot,cond.RRtot)[1:-1]*1e-6
            
    pylab.semilogy(zp,RdefPlot,'r',zp,RradPlot,'g',\
                   zp,RaugPlot,'b',zp,RtotPlot,'k')
    maxVal = max(RtotPlot)
    pylab.xlim([0,zp[-1]])
    pylab.ylim([1e10,10**scipy.ceil(scipy.log10(maxVal))])
    pylab.ylabel('Rate [1/cm3s]')
    pylab.xlabel('Position [nm]')
    pylab.legend(['Rdef','Rrad','Raug','Rtot'])
  
  elif plotType == 'currents':
    zp = cond.grid.zr*1e9
    pylab.plot(zp,cond.Jn*1e-4,'r',zp,cond.Jp*1e-4,'b')
    pylab.xlim([0,zp[-1]])
    pylab.ylabel('Current density [A/cm2]')
    pylab.xlabel('Position [nm]')
    pylab.legend(['Jn','Jp'])
  
  else:
    raise ValueError, 'Unknown plotType!'
