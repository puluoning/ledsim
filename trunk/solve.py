'''
'''
from ledsim import *
import calc, dynamic

def out_printer(itr,err,solverOpts):
  '''
  '''
  if solverOpts.verboseLevel >= 5:
    print '    Iteration '+str(itr)+'; Error = '+('+' if err > 0 else '')+str(err)

def solve_equilibrium_local(struct,solverOpts=SolverOpts()):
  ''' Solves for the condition that yields charge neutrality at each gridpoint.
      Note that the input type for this method is an ledsim.Structure rather
      than a dynamic.Condition. Return type is a dynamic.Condition. Options
      for the solver can be specified via solverOpts.
  '''
  # Generate guesses for the electrostatic potential, and electron and hole
  # quasi-Fermi levels yielding charge neutrality. This is done using the 
  # effective donor and acceptor dopant concentrations (taking into account
  # compensation doping) and Boltzmann statistics.
  isQuantum = struct.modelOpts.quantum
  struct.modelOpts.quantum = False
  kBT = kB*struct.modelOpts.T
  effectiveNd = scipy.maximum(scipy.ones(struct.grid.rnum),struct.Nd-struct.Na)
  effectiveNa = scipy.maximum(scipy.ones(struct.grid.rnum),struct.Na-struct.Nd)
  Efmid = (struct.Ec0+kBT*scipy.log(effectiveNd/struct.Nc))/q*(effectiveNd > 1)+ \
          (struct.Ev0[0,:]+kBT*scipy.log(effectiveNa/struct.Nv[0,:]))/q*(effectiveNa > 1)+ \
          (struct.Ec0+struct.Ev0[0,:])/(2*q)*(effectiveNa <= 1)*(effectiveNd <= 1)
  sel   = scipy.concatenate([effectiveNa+effectiveNd,scipy.ones(1)])> \
          scipy.concatenate([scipy.ones(1),effectiveNa+effectiveNd])
  Ef    = scipy.concatenate([[Efmid[0]],Efmid])*(-sel)+ \
          scipy.concatenate([Efmid,[Efmid[-1]]])*sel
  phi   = Ef-Ef[0]
  phiN  = Ef[0]+phi
  phiP  = Ef[0]+phi
  cond  = dynamic.Condition(struct,phi,phiN,phiP)
  
  # Solve for the Fermi-level positions yielding charge neutrality using the
  # Newton-Raphson method.
  if solverOpts.verboseLevel >= 3:
    print '>> running solve.solve_equilibrium_local;'
  itr  = 0
  diverged = False
  converged = False
  while not converged and not diverged and itr < solverOpts.maxitr:
    itr = itr+1
    condPhi     = cond.offset(solverOpts.dphi,solverOpts.dphi,solverOpts.dphi)
    drho2D_dphi = 1/solverOpts.dphi*(condPhi.rho2D-cond.rho2D)
    correction  = -cond.rho2D/drho2D_dphi
    errLoc    = scipy.argmax(abs(correction))
    err       = abs(correction[errLoc])
    signedErr = correction[errLoc]
    if err > solverOpts.maxAllowedCorrection:
      correction = correction/err*solverOpts.maxAllowedCorrection
    cond      = cond.offset(correction,correction,correction)
    converged = err < solverOpts.convergenceThreshold
    diverged  = scipy.sum(scipy.isnan(correction)) > 0
    out_printer(itr,signedErr,solverOpts)
  if converged:
    cond.err = err
    cond.itr = itr
    cond.converged = True
    cond.quantumConverged = False
    struct.modelOpts.quantum = isQuantum
    return cond
  else:
    raise ValueError, 'solve.solve_equilibrium failed to converge!'
    
def solve_poisson_single(initialCond,solverOpts):
  ''' Solves the Poisson equation, i.e. the condition of zero applied bias. The
      input is an initial dynamic.Condition which represents the guess at the
      solution. Solver options may be specified via solverOpts. 
  '''
  def get_jacobian(cond,solverOpts):
    ''' Returns the Jacobian and residue of the Poisson system.
    '''
    condPhi = cond.offset(solverOpts.dphi,solverOpts.dphi,solverOpts.dphi) 
    drho2D_dphi = 1/solverOpts.dphi*(condPhi.rho2D-cond.rho2D)
    res = -cond.epsilon[:-1]/cond.grid.dz[:-1]*(cond.phi[1:-1]-cond.phi[ :-2])+ \
           cond.epsilon[ 1:]/cond.grid.dz[ 1:]*(cond.phi[2:  ]-cond.phi[1:-1])- \
           cond.Ptot[1:]+cond.Ptot[:-1]+cond.rho2D[1:-1]
    nzmax = 3*(cond.grid.znum-2)-2
    sm = calc.SparseMaker(nzmax)
    sm.add_diag(-1,0,-cond.epsilon[1:-1]/cond.grid.dz[1:-1])
    sm.add_diag( 0,0, cond.epsilon[:-1]/cond.grid.dz[:-1]+ \
                      cond.epsilon[1: ]/cond.grid.dz[1: ]- \
                      drho2D_dphi[1:-1])
    sm.add_diag( 1,0,-cond.epsilon[1:-1]/cond.grid.dz[1:-1])
    mat = sm.assemble()
    return mat, res

  cond = initialCond.offset(0,0,0)
  cond.get_carriers_lr()
  itr = 0
  diverged = False
  converged = False
  while not converged and not diverged and itr < solverOpts.maxitr:
    itr = itr+1
    mat,res    = get_jacobian(cond,solverOpts)
    correction = scipy.sparse.linalg.spsolve(mat,res)
    errLoc     = scipy.argmax(abs(correction))
    err        = abs(correction[errLoc])
    signedErr  = correction[errLoc]
    converged  = err < solverOpts.convergenceThreshold
    diverged   = scipy.sum(scipy.isnan(correction)) > 0
    if err > solverOpts.maxAllowedCorrection:
      correction = correction/err*solverOpts.maxAllowedCorrection
    cond = cond.offset(scipy.concatenate((scipy.zeros(1),correction,scipy.zeros(1))),\
                       scipy.concatenate((scipy.zeros(1),correction,scipy.zeros(1))),\
                       scipy.concatenate((scipy.zeros(1),correction,scipy.zeros(1))))
    out_printer(itr,signedErr,solverOpts)
  if converged:
    cond.err = err
    cond.itr = itr
    cond.converged = True
    cond.quantumConverged = False
    return cond
  else:
    raise ValueError, 'Solver failed to converge!'

def solve_poisson_periodic_single(initialCond,solverOpts=SolverOpts(),
                      Ntarget=None,Ptarget=None,Rtarget=None):
  ''' Equilibrium solver which enforces periodic boundary conditions and zero
      total charge within the structure that is simulated. This is appropriate
      for the modeling of quantum wells. Electron sheet density, hole sheet
      density, and total recombination rate targets are supported. Note:
      Ntarget, Ptarget, and Rtarget are in units of 1/m2.
  '''
  def get_jacobian(cond,solverOpts,Ntarget,Ptarget,Rtarget):
    '''
    '''
    condPhi = cond.offset(solverOpts.dphi,solverOpts.dphi,solverOpts.dphi) 
    drho2D_dphi = 1/solverOpts.dphi*(condPhi.rho2D-cond.rho2D)
    res = -cond.epsilon[:-1]/cond.grid.dz[:-1]*(cond.phi[1:-1]-cond.phi[ :-2])+ \
           cond.epsilon[ 1:]/cond.grid.dz[ 1:]*(cond.phi[2:  ]-cond.phi[1:-1])- \
           cond.Ptot[1:]+cond.Ptot[:-1]+cond.rho2D[1:-1]
    nzmax = 7*(cond.grid.znum-2)+2
    sm = calc.SparseMaker(nzmax)
    sm.add_diag(-1,0,-cond.epsilon[1:-1]/cond.grid.dz[1:-1])
    sm.add_diag( 0,0, cond.epsilon[:-1]/cond.grid.dz[:-1]+ \
                      cond.epsilon[1: ]/cond.grid.dz[1: ]- \
                      drho2D_dphi[1:-1])
    sm.add_diag( 1,0,-cond.epsilon[1:-1]/cond.grid.dz[1:-1])
    if Ntarget == None and Ptarget == None and Rtarget == None:
      res = scipy.concatenate((res,scipy.array([cond.Q])))
      condPhiNP     = cond.offset(0,solverOpts.dphi,solverOpts.dphi)
      drho2D_dphiNP = 1/solverOpts.dphi*(-cond.rho2D+condPhiNP.rho2D) 
      dQ_dphiNP     = 1/solverOpts.dphi*(-cond.Q+condPhiNP.Q)
      r = scipy.concatenate((scipy.arange(0,cond.grid.znum-2),
                             scipy.ones(cond.grid.znum-1)*(cond.grid.znum-2)))
      c = scipy.concatenate((scipy.ones(cond.grid.znum-2)*(cond.grid.znum-2),
                             scipy.arange(0,cond.grid.znum-1)))
      v = scipy.concatenate((-drho2D_dphiNP[1:-1],-drho2D_dphi[1:-1],
                             scipy.array([-dQ_dphiNP])))
    else:
      condPhiN      = cond.offset(0,solverOpts.dphi,0)
      condPhiP      = cond.offset(0,0,solverOpts.dphi)
      drho2D_dphiN  = 1/solverOpts.dphi*(-cond.rho2D+condPhiN.rho2D)
      drho2D_dphiP  = 1/solverOpts.dphi*(-cond.rho2D+condPhiP.rho2D)
      dQ_dphiN      = 1/solverOpts.dphi*(-cond.Q+condPhiN.Q)
      dQ_dphiP      = 1/solverOpts.dphi*(-cond.Q+condPhiP.Q)
      r = scipy.concatenate((scipy.arange(0,cond.grid.znum-2),
                             scipy.arange(0,cond.grid.znum-2),
                             scipy.ones(cond.grid.znum)*(cond.grid.znum-2),
                             scipy.ones(cond.grid.znum)*(cond.grid.znum-1)))
      c = scipy.concatenate((scipy.ones(cond.grid.znum-2)*(cond.grid.znum-2),
                             scipy.ones(cond.grid.znum-2)*(cond.grid.znum-1),
                             scipy.arange(0,cond.grid.znum),
                             scipy.arange(0,cond.grid.znum)))
      if Ntarget != None:
        res = scipy.concatenate((res,scipy.array([cond.Q,cond.N-Ntarget])))
        dN_dphiN      = 1/solverOpts.dphi*(-cond.N+condPhiN.N)
        dN_dphiP      = 1/solverOpts.dphi*(-cond.N+condPhiP.N)
        dn2D_dphi     = 1/solverOpts.dphi*(-cond.n2D+condPhi.n2D)
        dn2D_dphiN    = 1/solverOpts.dphi*(-cond.n2D+condPhiN.n2D)
        dn2D_dphiP    = 1/solverOpts.dphi*(-cond.n2D+condPhiP.n2D)
        v = scipy.concatenate((-drho2D_dphiN[1:-1],-drho2D_dphiP[1:-1],
                               -drho2D_dphi[1:-1],scipy.array([-dQ_dphiN,-dQ_dphiP]),
                               -dn2D_dphi[1:-1],scipy.array([-dN_dphiN,-dN_dphiP])))
      elif Ptarget != None:
        res = scipy.concatenate((res,scipy.array([cond.Q,cond.P-Ptarget])))
        dP_dphiN      = 1/solverOpts.dphi*(-cond.P+condPhiN.P)
        dP_dphiP      = 1/solverOpts.dphi*(-cond.P+condPhiP.P)
        dp2D_dphi     = 1/solverOpts.dphi*(-cond.p2D+condPhi.p2D)
        dp2D_dphiN    = 1/solverOpts.dphi*(-cond.p2D+condPhiN.p2D)
        dp2D_dphiP    = 1/solverOpts.dphi*(-cond.p2D+condPhiP.p2D)
        v = scipy.concatenate((-drho2D_dphiN[1:-1],-drho2D_dphiP[1:-1],
                               -drho2D_dphi[1:-1],scipy.array([-dQ_dphiN,-dQ_dphiP]),
                               -dp2D_dphi[1:-1],scipy.array([-dP_dphiN,-dP_dphiP])))
      elif Rtarget != None:
        res = scipy.concatenate((res,scipy.array([cond.Q,cond.Rtot-Rtarget])))
        dRtot_dphiN   = 1/solverOpts.dphi*(-cond.Rtot+condPhiN.Rtot)
        dRtot_dphiP   = 1/solverOpts.dphi*(-cond.Rtot+condPhiP.Rtot)
        dRtot2D_dphi  = 1/solverOpts.dphi*(-cond.Rtot2D+condPhi.Rtot2D)
        dRtot2D_dphiN = 1/solverOpts.dphi*(-cond.Rtot2D+condPhiN.Rtot2D)
        dRtot2D_dphiP = 1/solverOpts.dphi*(-cond.Rtot2D+condPhiP.Rtot2D)
        v = scipy.concatenate((-drho2D_dphiN[1:-1],-drho2D_dphiP[1:-1],
                               -drho2D_dphi[1:-1],scipy.array([-dQ_dphiN,-dQ_dphiP]),
                               -dRtot2D_dphi[1:-1],scipy.array([-dRtot_dphiN,-dRtot_dphiP])))
    sm.add_elem(r,c,v)
    mat = sm.assemble()
    return mat, res
  
  cond = initialCond.offset(0,0,0)
  cond.get_carriers_lr()
  itr = 0
  diverged = False
  converged = False
  while not converged and not diverged and itr < solverOpts.maxitr:
    itr = itr+1
    mat,res    = get_jacobian(cond,solverOpts,Ntarget,Ptarget,Rtarget)
    correction = scipy.sparse.linalg.spsolve(mat,res)
    errLoc     = scipy.argmax(abs(correction))
    err        = abs(correction[errLoc])
    signedErr  = correction[errLoc]
    converged  = err < solverOpts.convergenceThreshold
    diverged   = scipy.sum(scipy.isnan(correction)) > 0
    if err > solverOpts.maxAllowedCorrection:
      correction = correction/err*solverOpts.maxAllowedCorrection
    correctionPhi  = scipy.concatenate((scipy.zeros(1),correction[:cond.grid.znum-2],scipy.zeros(1)))
    if Ntarget == None and Ptarget == None and Rtarget == None:
      correctionPhiN = correctionPhi+correction[cond.grid.znum-2]
      correctionPhiP = correctionPhi+correction[cond.grid.znum-2]
    else:
      correctionPhiN = correctionPhi+correction[cond.grid.znum-2]
      correctionPhiP = correctionPhi+correction[cond.grid.znum-1]
    cond = cond.offset(correctionPhi,correctionPhiN,correctionPhiP)
    out_printer(itr,signedErr,solverOpts)
  if converged:
    cond.err = err
    cond.itr = itr
    cond.converged = True
    cond.quantumConverged = False
    return cond
  else:
    raise ValueError, 'Solver failed to converge!'
    
def solve_nonequilibrium_single(initialCond,solverOpts=SolverOpts(),Jtarget=None):
  ''' Solves the drift-diffusion system, i.e. the Poisson equation together with
      continuity equations for electrons and holes. A target current density may
      be specified using Jtarget (units are A/m2). If no current is specified, 
      a voltage boundary condition is used, with voltage matching that of the
      provided initial condition (initialCond). Solver options may be specified
      using solverOpts.
  '''
  def get_jacobian(cond,solverOpts,Jtarget):
    ''' Return the Jacobian for the drift-diffusion system.
    ''' 
    # Calculate the normalization factors for the Poisson equation and the 
    # continuity equations for electrons and holes. Also calculate the 
    # difference in phi, Efn/q, and Efp/q, as well as the residues for each of
    # the three equations.
    nfp         = 2/(cond.epsilon[:-1]/cond.grid.dz[:-1]+cond.epsilon[ 1:]/cond.grid.dz[ 1:])
    nfN         = 2/(cond.cn[:-1]/cond.grid.dz[:-1]+cond.cn[ 1:]/cond.grid.dz[ 1:])
    nfP         = 2/(cond.cp[:-1]/cond.grid.dz[:-1]+cond.cp[ 1:]/cond.grid.dz[ 1:])
    phiDiff     = cond.phi[1:]-cond.phi[:-1]
    EfnDiff     = cond.phiN[1:]-cond.phiN[:-1]-phiDiff
    EfpDiff     = cond.phiP[1:]-cond.phiP[:-1]-phiDiff
    residuePhi  = (-cond.epsilon[:-1]/cond.grid.dz[:-1]*phiDiff[:-1]+ \
                    cond.epsilon[ 1:]/cond.grid.dz[ 1:]*phiDiff[1: ]- \
                    cond.Ptot[1:]+cond.Ptot[:-1]+cond.rho2D[1:-1])*nfp
    residuePhiN = ( cond.cn[:-1]/cond.grid.dz[:-1]*EfnDiff[:-1]- \
                    cond.cn[1: ]/cond.grid.dz[1: ]*EfnDiff[1: ]+cond.Rtot2D[1:-1])*nfN
    residuePhiP = ( cond.cp[:-1]/cond.grid.dz[:-1]*EfpDiff[:-1]- \
                    cond.cp[1: ]/cond.grid.dz[1: ]*EfpDiff[1: ]-cond.Rtot2D[1:-1])*nfP
    
    # Generate the offset conditions used in calculating derivatives of
    # various terms with respect to phi, phiN, and phiP.      
    dphi0 = solverOpts.dphi*calc.alternate(scipy.ones(cond.grid.znum),scipy.zeros(cond.grid.znum))
    dphi1 = solverOpts.dphi*calc.alternate(scipy.zeros(cond.grid.znum),scipy.ones(cond.grid.znum))
    condPhi0  = cond.offset(dphi0,0,0)
    condPhi1  = cond.offset(dphi1,0,0)
    condPhiN0 = cond.offset(0,dphi0,0)
    condPhiN1 = cond.offset(0,dphi1,0)
    condPhiP0 = cond.offset(0,0,dphi0)
    condPhiP1 = cond.offset(0,0,dphi1)
    
    # Now calculate the derivatives of cn and cp with respect to phi, phiN, and
    # phiP to the left and right of each region, as well as the derivative of
    # 2D sheet charge density and recombination within each integration box 
    # also with respect to phi, phiN, and phiP.
    dcn_dLphi     = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhi0.cn,condPhi1.cn))
    dcn_dRphi     = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhi1.cn,condPhi0.cn))
    dcn_dLphiN    = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhiN0.cn,condPhiN1.cn))
    dcn_dRphiN    = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhiN1.cn,condPhiN0.cn))
    dcn_dLphiP    = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhiP0.cn,condPhiP1.cn))
    dcn_dRphiP    = EfnDiff/solverOpts.dphi/cond.grid.dz*(-cond.cn+calc.alternate(condPhiP1.cn,condPhiP0.cn))
    dcp_dLphi     = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhi0.cp,condPhi1.cp))
    dcp_dRphi     = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhi1.cp,condPhi0.cp))
    dcp_dLphiN    = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhiN0.cp,condPhiN1.cp))
    dcp_dRphiN    = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhiN1.cp,condPhiN0.cp))
    dcp_dLphiP    = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhiP0.cp,condPhiP1.cp))
    dcp_dRphiP    = EfpDiff/solverOpts.dphi/cond.grid.dz*(-cond.cp+calc.alternate(condPhiP1.cp,condPhiP0.cp))
    drho2D_dphi   = 1/solverOpts.dphi*(-cond.rho2D+calc.alternate(condPhi0.rho2D,condPhi1.rho2D))
    drho2D_dphiN  = 1/solverOpts.dphi*(-cond.rho2D+calc.alternate(condPhiN0.rho2D,condPhiN1.rho2D))
    drho2D_dphiP  = 1/solverOpts.dphi*(-cond.rho2D+calc.alternate(condPhiP0.rho2D,condPhiP1.rho2D))
    dRtot2D_dphi  = 1/solverOpts.dphi*(-cond.Rtot2D+calc.alternate(condPhi0.Rtot2D,condPhi1.Rtot2D))
    dRtot2D_dphiN = 1/solverOpts.dphi*(-cond.Rtot2D+calc.alternate(condPhiN0.Rtot2D,condPhiN1.Rtot2D))
    dRtot2D_dphiP = 1/solverOpts.dphi*(-cond.Rtot2D+calc.alternate(condPhiP0.Rtot2D,condPhiP1.Rtot2D))
    
    # Create the container for the Jacobian matrix.
    nzmax = cond.grid.znum*3*9
    snum  = cond.grid.znum-2
    sm    = calc.SparseMaker(nzmax)
    
    # Add the matrix elements for the Poisson equation, the continuity equation
    # for electrons, and the continuity equation for holes.
    sm.add_diag( 0*snum-1,0*snum  ,nfp[1: ]*(-cond.epsilon[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 0*snum  ,0*snum  ,nfp     *( cond.epsilon[:-1 ]/cond.grid.dz[:-1 ]+ \
                                              cond.epsilon[1:  ]/cond.grid.dz[1:  ]-drho2D_dphi[1:-1]))
    sm.add_diag( 0*snum+1,0*snum  ,nfp[:-1]*(-cond.epsilon[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 1*snum  ,0*snum  ,nfp     *(-drho2D_dphiN[1:-1]))
    sm.add_diag( 2*snum  ,0*snum  ,nfp     *(-drho2D_dphiP[1:-1]))
    sm.add_diag(-1*snum-1,0*snum  ,nfN[1: ]*(-dcn_dLphi[1:-1]-cond.cn[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag(-1*snum  ,0*snum  ,nfN     *( dcn_dLphi[1:  ]-dcn_dRphi[:-1]+ \
                                              cond.cn[:-1]/cond.grid.dz[:-1]+ \
                                              cond.cn[1: ]/cond.grid.dz[1: ]-dRtot2D_dphi[1:-1]))
    sm.add_diag(-1*snum+1,0*snum+1,nfN[:-1]*( dcn_dRphi[1:-1]-cond.cn[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 0*snum-1,1*snum  ,nfN[1: ]*(-dcn_dLphiN[1:-1]+cond.cn[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 0*snum  ,1*snum  ,nfN     *( dcn_dLphiN[1:  ]-dcn_dRphiN[:-1]- \
                                              cond.cn[:-1]/cond.grid.dz[:-1]- \
                                              cond.cn[1: ]/cond.grid.dz[1: ]-dRtot2D_dphiN[1:-1]))
    sm.add_diag( 0*snum+1,1*snum  ,nfN[:-1]*( dcn_dRphiN[1:-1]+cond.cn[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 1*snum-1,1*snum+1,nfN[1: ]*(-dcn_dLphiP[1:-1]))
    sm.add_diag( 1*snum  ,1*snum  ,nfN     *( dcn_dLphiP[1:  ]-dcn_dRphiP[:-1]-dRtot2D_dphiP[1:-1]))
    sm.add_diag( 1*snum+1,1*snum  ,nfN[:-1]*( dcn_dRphiP[1:-1]))
    sm.add_diag(-2*snum-1,0*snum  ,nfP[1: ]*(-dcp_dLphi[1:-1]-cond.cp[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag(-2*snum  ,0*snum  ,nfP     *( dcp_dLphi[1:  ]-dcp_dRphi[:-1]+ \
                                              cond.cp[:-1]/cond.grid.dz[:-1]+ \
                                              cond.cp[1: ]/cond.grid.dz[1: ]+dRtot2D_dphi[1:-1]))
    sm.add_diag(-2*snum+1,0*snum+1,nfP[:-1]*( dcp_dRphi[1:-1]-cond.cp[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag(-1*snum-1,1*snum  ,nfP[1: ]*(-dcp_dLphiN[1:-1]))
    sm.add_diag(-1*snum  ,1*snum  ,nfP     *( dcp_dLphiN[1:  ]-dcp_dRphiN[:-1]+dRtot2D_dphiN[1:-1]))
    sm.add_diag(-1*snum+1,1*snum+1,nfP[:-1]*( dcp_dRphiN[1:-1]))
    sm.add_diag( 0*snum-1,2*snum  ,nfP[1: ]*(-dcp_dLphiP[1:-1]+cond.cp[1:-1]/cond.grid.dz[1:-1]))
    sm.add_diag( 0*snum  ,2*snum  ,nfP     *( dcp_dLphiP[1:  ]-dcp_dRphiP[:-1]- \
                                              cond.cp[:-1]/cond.grid.dz[:-1]- \
                                              cond.cp[1: ]/cond.grid.dz[1: ]+dRtot2D_dphiP[1:-1]))
    sm.add_diag( 0*snum+1,2*snum  ,nfP[:-1]*( dcp_dRphiP[1:-1]+cond.cp[1:-1]/cond.grid.dz[1:-1]))
    
    # Add matrix elements for current boundary condition.
    if Jtarget != None:
      nfC = q/Jtarget
      residueCBC = scipy.array([(-Jtarget+cond.J)/q*nfC])
      row = scipy.array([1*snum-1,2*snum-1,3*snum-1,3*snum  ,3*snum  ,3*snum  ,3*snum]) 
      col = scipy.array([3*snum  ,3*snum  ,3*snum  ,1*snum-1,2*snum-1,3*snum-1,3*snum])
      val = scipy.array([nfp[-1]*(-cond.epsilon[-1]/cond.grid.dz[-1]),
                         nfN[-1]*( dcn_dRphi[-1]-cond.cn[-1]/cond.grid.dz[-1]),
                         nfP[-1]*( dcp_dRphi[-1]-cond.cp[-1]/cond.grid.dz[-1]),
                         nfC*(dcn_dLphi[-1] +dcp_dLphi[-1]+cond.cn[-1]/cond.grid.dz[-1]+ \
                                                           cond.cp[-1]/cond.grid.dz[-1]),
                         nfC*(dcn_dLphiN[-1]+dcp_dLphiN[-1]-cond.cn[-1]/cond.grid.dz[-1]),
                         nfC*(dcn_dLphiP[-1]+dcp_dLphiP[-1]-cond.cp[-1]/cond.grid.dz[-1]),
                         nfC*(dcn_dRphi[-1] +dcp_dRphi[-1]-cond.cn[-1]/cond.grid.dz[-1]- \
                                                           cond.cp[-1]/cond.grid.dz[-1])])
      sm.add_elem(row,col,val)
      res = scipy.concatenate((residuePhi,residuePhiN,residuePhiP,residueCBC))
    else:
      res = scipy.concatenate((residuePhi,residuePhiN,residuePhiP))
    mat = sm.assemble()
    return mat,res
  
  if solverOpts.verboseLevel >= 3:
    if Jtarget != None:
      print '>> running solve.solve_nonequilibrium_single; Jtarget='+str(Jtarget)+'A/m2'
    else:
      print '>> running solve.solve_nonequilibrium_single; Vtarget='+str(initialCond.V)
  cond = initialCond.offset(0,0,0)
  cond.get_carriers_lr()
  itr = 0
  snum = cond.grid.znum-2
  diverged = False
  converged = False
  while not converged and not diverged and itr < solverOpts.maxitr:
    itr = itr+1
    mat,res    = get_jacobian(cond,solverOpts,Jtarget)
    correction = scipy.sparse.linalg.spsolve(mat,res)
    errLoc     = scipy.argmax(abs(correction))
    err        = abs(correction[errLoc])
    signedErr  = correction[errLoc]
    converged  = err < solverOpts.convergenceThreshold
    diverged   = scipy.sum(scipy.isnan(correction)) > 0
    if err > solverOpts.maxAllowedCorrection:
      correction = correction/err*solverOpts.maxAllowedCorrection
    correctionPhi  = scipy.concatenate((scipy.zeros(1),correction[0*snum:1*snum],scipy.zeros(1)))
    correctionPhiN = scipy.concatenate((scipy.zeros(1),correction[1*snum:2*snum],scipy.zeros(1)))
    correctionPhiP = scipy.concatenate((scipy.zeros(1),correction[2*snum:3*snum],scipy.zeros(1)))
    if Jtarget != None:
      correctionPhi[-1] = correction[-1]
    cond = cond.offset(correctionPhi,correctionPhiN,correctionPhiP)
    out_printer(itr,signedErr,solverOpts)
  if converged:
    cond.err = err
    cond.itr = itr
    cond.converged = True
    cond.quantumConverged = False
    return cond
  else:
    raise ValueError, 'Solver failed to converge!'
 
def solve_poisson(initialCond,solverOpts=SolverOpts()):
  ''' Solve the basic Poisson equation. For the classical case, this simply 
      calls the method solve_poisson_periodic_single, while for the quantum 
      case the outer loop of the solution is handled.
  '''
  if solverOpts.verboseLevel >= 3:
    print '>> running solve.solve_poisson;'
  if not initialCond.modelOpts.quantum:
    return solve_poisson_single(initialCond,solverOpts)
  else:
    if not initialCond.quantumConverged:
      if solverOpts.verboseLevel >= 4:
        print ' > solving for classical initial condition;'
      initialCond.modelOpts.quantum = False
      initialCond = solve_poisson_periodic_single(initialCond,solverOpts)
      initialCond.modelOpts.quantum = True
    itr = 0
    converged = False
    solverOpts.maxitrOuter = 3
    while not converged and itr < solverOpts.maxitrOuter:
      itr += 1
      cond = solve_poisson_single(initialCond,solverOpts)
      error = max(max(abs(cond.EcWavefunctions.phiOrig-cond.phi)),max(abs(cond.EvWavefunctions.phiOrig-cond.phi)))
      if solverOpts.verboseLevel >= 4:
        print ' > Outer iteration %i; Error ='%itr,error
      converged = error < solverOpts.quantumConvergenceThreshold
      if not converged:
        initialCond = cond.offset(0,0,0,keepWavefunctions=False)
    if converged:
      cond.quantumConverged = True
      return cond
    else:
      raise ValueError, 'Solver failed to converge!'
  
def solve_poisson_periodic(initialCond,solverOpts=SolverOpts(),Ntarget=None,Ptarget=None,Rtarget=None):
  ''' Solve the Poisson equation with periodic boundary conditions and
      specified targets for either electron or hole sheet density, or sheet
      recombination rate. For the classical case, this simply calls the method
      solve_poisson_periodic_single, while for the quantum case the outer loop
      of the solution is handled.
  '''
  if solverOpts.verboseLevel >= 3:
    print '>> running solve.solve_poisson_periodic;'
  if not initialCond.modelOpts.quantum:
    return solve_poisson_periodic_single(initialCond,solverOpts,Ntarget,Ptarget,Rtarget)
  else:
    if not initialCond.quantumConverged:
      if solverOpts.verboseLevel >= 4:
        print ' > solving for classical initial condition;'
      initialCond.modelOpts.quantum = False
      initialCond = solve_poisson_periodic_single(initialCond,solverOpts,Ntarget,Ptarget,Rtarget)
      initialCond.modelOpts.quantum = True
    itr = 0
    converged = False
    solverOpts.maxitrOuter = 3
    while not converged and itr < solverOpts.maxitrOuter:
      itr += 1
      cond = solve_poisson_periodic_single(initialCond,solverOpts,Ntarget,Ptarget,Rtarget)
      error = max(max(abs(cond.EcWavefunctions.phiOrig-cond.phi)),max(abs(cond.EvWavefunctions.phiOrig-cond.phi)))
      if solverOpts.verboseLevel >= 4:
        print ' > Outer iteration %i; Error ='%itr,error
      converged = error < solverOpts.quantumConvergenceThreshold
      if not converged:
        initialCond = cond.offset(0,0,0,keepWavefunctions=False)
    if converged:
      cond.quantumConverged = True
      return cond
    else:
      raise ValueError, 'Solver failed to converge!'
  
def solve_nonequilibrium(initialCond,solverOpts=SolverOpts(),Jtarget=None):
  '''
  '''
  return solve_nonequilibrium_single(initialCond,solverOpts,Jtarget)
 
def bias(initialCond,solverOpts=SolverOpts(),Vtarget=None,Jtarget=None):
  ''' Solve for the structure at the specified voltage or current.
  '''  
  def solve_prep(cond,Vtarget,solverOpts):
    Vincr = Vtarget-cond.V
    dphi  = scipy.concatenate((scipy.zeros(cond.grid.znum-1),scipy.ones(1)))*Vincr
    return dynamic.Condition(cond.struct,cond.phi+dphi,cond.phiN,cond.phiP) 
  
  def voltage_ramp(cond,solverOpts=None,Vtarget=None,Jtarget=None):
    dV = solverOpts.dVmax
    itr = 0
    done = False
    while itr < solverOpts.maxitrOuter and not done:
      itr = itr+1
      if Vtarget != None:
        dV = min(abs(Vtarget-cond.V),dV)
        done = dV == abs(Vtarget-cond.V)
        guess = solve_prep(cond,cond.V+dV*scipy.sign(Vtarget-cond.V),solverOpts)
      else:
        guess = solve_prep(cond,cond.V+dV,solverOpts)
      try:
        cond = solve_nonequilibrium(guess,solverOpts=solverOpts)
        dV = min(1.1*dV,solverOpts.dVmax)
        if Jtarget != None:
          done = cond.J > Jtarget
      except:
        dV = 0.5*dV
        done = False
    if itr == solverOpts.maxitrOuter:
      raise ValueError, 'Maximum number of outer iterations exceeded!'
    else:
      return cond

  if isinstance(initialCond,Structure):
    cond = solve_equilibrium(solve_equilibrium_local(initialCond,solverOpts),solverOpts)
  else:
    cond = initialCond.offset(0,0,0)
  if Jtarget != None:
    if solverOpts.verboseLevel >= 2:
      print '>> Solving for bias condition: J='+str(Jtarget)+'A/m2'
    if cond.J < solverOpts.Jmin:
      cond = voltage_ramp(cond,solverOpts=solverOpts,Jtarget=Jtarget)  
      cond = solve_nonequilibrium(cond,solverOpts=solverOpts,Jtarget=Jtarget)
    else:
      cond = solve_nonequilibrium(cond,solverOpts=solverOpts,Jtarget=Jtarget)
  elif Vtarget != None:
    if solverOpts.verboseLevel >= 2:
      print '>> Solving for bias condition: Vf='+str(Vtarget)+'V'
    cond = voltage_ramp(cond,solverOpts=solverOpts,Vtarget=Vtarget)  
  return cond