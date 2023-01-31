#
# Trapezoidal transcription for dynamical systems
#

import numpy as np 
import scipy.sparse as scis
import scipy.sparse.linalg as scisl 

#
# USAGE: import trapztr ...
# 
# DEMOS: python3 trapztr.py --help
#        python3 trapztr.py --nnodes 500 --hess-optimize 
#        python3 trapztr.py --nnodes 500 --hess-optimize --umin -1.0 --umax 1.0
#        python3 trapztr.py --nnodes 500 --hess-optimize --umin -4.0 --umax 3.0 --adjust-optimum
#

def coo_data_from_block(B, rowofs = 0, colofs = 0, deleteZeros = False):
  nbr = B.shape[0]
  nbc = B.shape[1]
  bir = np.tile(np.arange(rowofs, rowofs + nbr).reshape((nbr, 1)), (1, nbc))
  bic = np.tile(np.arange(colofs, colofs + nbc), (nbr, 1))
  if deleteZeros:
    isNonZero = B.flatten() != 0.0
    return B.flatten()[isNonZero], bir.flatten()[isNonZero], bic.flatten()[isNonZero]
  else:
    return B.flatten(), bir.flatten(), bic.flatten()

# Return function value, total gradient, and total block sparse Hessian.
# If gradf = None, only the function value is returned. 
# The hessian is only assembled if gradf and hessf are both not None.
def totalObjective(Z, f, gradf, hessf, timePoints, sigmaVector, weightVector, zref, pinchCost):
  assert callable(f)
  nz = len(sigmaVector)
  nk = len(timePoints)  # nk=N+1 time points, defines N=nk-1 time intervals
  assert len(weightVector) == nz 
  assert len(Z) == nk * nz
  N = nk - 1

  activeStates = sigmaVector > 0.0
  assert activeStates.size == nz
  nv = np.sum(activeStates)
  hasActiveStates = nv > 0

  equalityStates = sigmaVector == 0.0
  nveq = np.sum(equalityStates)
  hasEqualityStates = nveq > 0

  fZ = 0.0
  calcGrad = not gradf is None 
  gradfZ = np.zeros((nk * nz, )) if calcGrad else None 
  hessfZ = None
  calcHess = calcGrad and (not hessf is None)

  if calcHess:
    coo_data = []
    coo_rows = []
    coo_cols = []

  for k in range(N):
    if not hasActiveStates: break # skip entire section
    hk = timePoints[k + 1] - timePoints[k]
    assert hk > 0.0
    idx0 = np.arange(k * nz, k * nz + nz)
    idx1 = np.arange((k + 1) * nz, (k + 1) * nz + nz)
    zk0 = Z[idx0]
    zk1 = Z[idx1]
    fk0 = f(zk0)
    fk1 = f(zk1)
    hhk = hk / 2.0
    wk = (zk1 - zk0) - hhk * (fk0 + fk1)
    sqrt_hk = np.sqrt(hk)
    ek = wk[activeStates] / (sqrt_hk * sigmaVector[activeStates])
    fZ += 0.5 * np.sum(ek ** 2)

    # d(sum(e^2))/dz0 and d(e^2)/dz1 are needed for each k
    # d(sum(e^2)) = d(e'*e) = d(e')*e + e'*d(e) = 2 * e' * d(e)
    # d(e) = d(w) / (sqrth * sigma)
    # d(w) = d[z1 - z0 - (h/2) * (f(z0) + f(z1))]
    # dwdz0 = -1 * eye - (h/2) * dfz0 
    # dwdz1 = eye - (h/2) * dfz1

    if calcGrad:
      dfk0 = gradf(zk0)
      dfk1 = gradf(zk1)
      v = (ek / (sqrt_hk * sigmaVector[activeStates])).reshape((nv, 1))
      dfdz0 = np.sum((-np.eye(nz) - hhk * dfk0)[activeStates, :] * np.tile(v, (1, nz)), axis = 0)
      dfdz1 = np.sum((+np.eye(nz) - hhk * dfk1)[activeStates, :] * np.tile(v, (1, nz)), axis = 0)
      gradfZ[idx0] = gradfZ[idx0] + dfdz0
      gradfZ[idx1] = gradfZ[idx1] + dfdz1

    if calcHess:
      hfk0 = hessf(zk0)
      hfk1 = hessf(zk1)
      assert hfk0.shape == (nz, nz, nz)
      assert hfk1.shape == (nz, nz, nz)

      v0 = (1.0 / (sqrt_hk * sigmaVector[activeStates])).reshape((nv, 1))
      v1 = (-np.eye(nz) - hhk * dfk0)[activeStates, :]
      v2 = (+np.eye(nz) - hhk * dfk1)[activeStates, :]
      V = np.hstack([v1, v2]) * np.tile(v0, (1, 2 * nz))
      Hk = np.dot(V.T, V)

      for j in range(nz):
        if activeStates[j]: # and (np.any(hfk0[j, :, :] != 0) or np.any(hfk1[j, :, :] != 0)):
          Hkj0 = -1.0 * hhk * hfk0[j, :, :]
          Hkj1 = -1.0 * hhk * hfk1[j, :, :]
          skj = wk[j] / (hk * (sigmaVector[j] ** 2))
          Hk += skj * np.vstack([np.hstack([Hkj0, np.zeros((nz, nz))]), 
                                 np.hstack([np.zeros((nz, nz)), Hkj1])])

      blkdata, blkrow, blkcol = coo_data_from_block(Hk, rowofs = k * nz, colofs = k * nz)
      coo_data += blkdata.tolist()
      coo_rows += blkrow.tolist()
      coo_cols += blkcol.tolist()

  activeWeights = weightVector > 0.0
  assert activeWeights.size == nz
  hasAnyWeight = np.sum(activeWeights) > 0

  # Handle the trapezoidal integral cost term (i.e. control cost and/or state regularization etc..)
  if hasAnyWeight:
    if zref is None:
      zref = np.zeros((nk, nz))

    assert zref.shape[0] == nk and zref.shape[1] == nz
    for k in range(N):
      hk = timePoints[k + 1] - timePoints[k]
      idx0 = np.arange(k * nz, k * nz + nz)
      idx1 = np.arange((k + 1) * nz, (k + 1) * nz + nz)
      zk0 = Z[idx0]
      zk1 = Z[idx1]
      cost0 = 0.5 * np.sum( weightVector[activeWeights] * ((zk0 - zref[k, :])[activeWeights]) ** 2)
      cost1 = 0.5 * np.sum( weightVector[activeWeights] * ((zk1 - zref[k + 1, :])[activeWeights]) ** 2)
      fZ += hk * (cost0 + cost1) / 2.0

      if calcGrad:
        gk0 = weightVector[activeWeights] * (zk0 - zref[k, :])[activeWeights]
        gradfZ[idx0[activeWeights]] += hk * gk0 / 2.0

        gk1 = weightVector[activeWeights] * (zk1 - zref[k + 1, :])[activeWeights]
        gradfZ[idx1[activeWeights]] += hk * gk1 / 2.0 

      if calcHess:
        Hk1 = np.zeros((nz, ))
        Hk1[activeWeights] = weightVector[activeWeights] * hk / 2.0
        Hk2 = np.zeros((nz, ))
        Hk2[activeWeights] = weightVector[activeWeights] * hk / 2.0
        Hk = np.diag(np.hstack([Hk1, Hk2]))

        blkdata, blkrow, blkcol = coo_data_from_block(Hk, rowofs = k * nz, colofs = k * nz, deleteZeros = True)
        coo_data += blkdata.tolist()
        coo_rows += blkrow.tolist()
        coo_cols += blkcol.tolist()

  # Handle the unordered node-specific "observational" cost terms; 
  # ignore non-quadratic cost terms (i.e. type != 2)
  if pinchCost is not None:
    assert isinstance(pinchCost, dict)
    for K in pinchCost.keys():
      pcK = pinchCost[K]
      tK = pcK['time']
      if tK < timePoints[0] or tK > timePoints[-1]:
        continue
      if pcK['type'] != 2: 
        continue 
      bK = pcK['bias']
      sK = pcK['sigma']
      assert bK.size == nz 
      assert sK.size == nz
      activeK = sK > 0.0
      naK = np.sum(activeK)
      if naK == 0:
        continue
      k1 = np.where(timePoints >= tK)[0][0]
      if k1 == 0:
        k0 = 0
        k1 = 1
      else:
        k0 = k1 - 1
      tk0 = timePoints[k0]
      tk1 = timePoints[k1]
      xik = (tK - tk0) / (tk1 - tk0)
      assert xik >= 0.0 and xik <= 1.0 
      #
      # xi is in [0, 1]
      # z(t) = z0 * (1 - xi) + z1 * (xi)
      # cost_k = sum([ (z(t) - b(t)) / s(t)]^2)
      # 
      idx0 = np.arange(k0 * nz, k0 * nz + nz)
      idx1 = np.arange(k1 * nz, k1 * nz + nz)
      zk0 = Z[idx0]
      zk1 = Z[idx1]
      zxi = (1.0 - xik) * zk0 + xik * zk1 

      errK = (zxi[activeK] - bK[activeK]) / sK[activeK]
      costK = 0.5 * np.sum(errK ** 2)

      fZ += costK

      if calcGrad: 
        gradfZ[idx0[activeK]] += (1.0 - xik) * errK / sK[activeK]
        gradfZ[idx1[activeK]] += (xik) * errK / sK[activeK]

      if calcHess:
        # Set Mhat = diag(1/sigma)*[(1-xi)*I, xi*I], only picking out the active rows
        # Then errK = Mhat * [z0;z1] - bK/sigma, and the gradient is 2*Mhat'*errK; and the hessian block is 2*Mhat'*Mhat
        Mhat = np.hstack([(1.0 - xik) * np.eye(nz), xik * np.eye(nz)])[activeK, :] * np.tile((1 / sK[activeK]).reshape((naK, 1)), (1, 2 * nz))
        Hk = np.dot(Mhat.T, Mhat)

        blkdata, blkrow, blkcol = coo_data_from_block(Hk, rowofs = k0 * nz, colofs = k0 * nz, deleteZeros = True)
        coo_data += blkdata.tolist() 
        coo_rows += blkrow.tolist()
        coo_cols += blkcol.tolist()


  if calcHess:
    hessfZ = scis.coo_array((coo_data, (coo_rows, coo_cols)), shape = (nk * nz, nk * nz))

  return fZ, gradfZ, None if not calcHess else hessfZ.tocsc() 
  

def make_bounds_tuples(zmin, zmax, nt):
  assert len(zmin) == len(zmax)
  assert np.all(zmin <= zmax)
  nz = len(zmin)
  B = []
  for k in range(nt):
    for i in range(nz):
      B.append((zmin[i], zmax[i]))
  return B 

# Construct sparse matrix E and rhs f, such that E * z <= f contains all bounds
# zmin <= z <= zmax for all nt node vectors; use COO format for construction and
# return E in CSC format. Each block has the form: I * zk <= zmax & -I * zk <= -zmin

def make_bounds_E_f(zmin, zmax, nt):
  # Generate full matrix then delete all rows not in use afterwards; easy to find by elements in f
  nz = zmin.size
  assert nz == zmax.size 
  assert np.all(zmin <= zmax)
  E_data = []
  E_rows = []
  E_cols = []
  for k in range(nt): 
    blkdata, blkrow, blkcol = coo_data_from_block(np.vstack([np.eye(nz), -1.0 * np.eye(nz)]), rowofs = k * 2 * nz, colofs = k * nz, deleteZeros = True)
    E_data += blkdata.tolist() 
    E_rows += blkrow.tolist()
    E_cols += blkcol.tolist()

  E = scis.coo_array((E_data, (E_rows, E_cols)), shape = (nz * nt * 2, nz * nt)).tocsc()
  f = np.tile(np.hstack([zmax, -1 * zmin]), (nt, ))

  assert f.size == E.shape[0]
  idxKeep = np.isfinite(f)

  return E[idxKeep, :], f[idxKeep]

# Actually make this return the constraint error value wk[not activeStates], and its Jacobian at a specific Z.
# If the system is linear then the Jacobian is a constant (and sparse) matrix; but it is nonlinear (and sparse) in general.
# Expect "nt - 1" time-intervals, and Z.size == nt * nz;
def eval_equality_constraint(Z, f, gradf, timePoints, sigmaVector):
  assert callable(f)
  calcJacobian = not gradf is None
  if calcJacobian:
    C_data = []
    C_rows = []
    C_cols = []
    assert callable(gradf)

  nz = sigmaVector.size 
  nt = timePoints.size 
  equalityStates = sigmaVector == 0.0
  nveq = np.sum(equalityStates)
  hasEqualityStates = nveq > 0
  if not hasEqualityStates:
    return None, None 
  neqs = (nt - 1) * nveq 
  nx = nz * nt # Jacobian has size (neqs, nx) and error vector has size (neqs, )
  assert len(Z.shape) == 1 and Z.size == nx
  hx = np.tile(np.nan, (neqs, ))
 
  r = 0
  for k in range(nt - 1):
    hk = timePoints[k + 1] - timePoints[k]
    hhk = hk / 2.0
    idx0 = np.arange(k * nz, k * nz + nz)
    idx1 = np.arange((k + 1) * nz, (k + 1) * nz + nz)
    zk0 = Z[idx0]
    zk1 = Z[idx1]
    fk0 = f(zk0)
    fk1 = f(zk1)
    wk = zk1 - zk0 - hhk * (fk0 + fk1)
    hx[np.arange(r, r + nveq)] = wk[equalityStates]

    if calcJacobian:
      gk0 = gradf(zk0)
      gk1 = gradf(zk1)
      Jk0 = (-1.0 * np.eye(nz) - hhk * gk0)[equalityStates, :]
      Jk1 = (+1.0 * np.eye(nz) - hhk * gk1)[equalityStates, :]
      Bk = np.hstack([Jk0, Jk1])
      assert Bk.shape[0] == nveq and Bk.shape[1] == 2 * nz  
      # Insert matrix block Bk = [Jk0, Jk1] at rows [r, r + neqv) and cols [idx0, idx1]
      blkdata, blkrow, blkcol = coo_data_from_block(Bk, rowofs = r, colofs = k * nz, deleteZeros = True)
      C_data += blkdata.tolist() 
      C_rows += blkrow.tolist()
      C_cols += blkcol.tolist()

    r += nveq

  assert r == neqs
  assert np.all(np.isfinite(hx))
  return hx, scis.coo_array((C_data, (C_rows, C_cols)), shape = (neqs, nx)).tocsc() if calcJacobian else None  

# Generate equality constraints (with Jacobian) from the ptCost items that contain sigma == 0 elements
# These terms are independent of the dynamics function f.
def eval_equality_constraint_pt(Z, timePoints, pinchCost, calcJacobian = True):
  nt = timePoints.size 
  nz = Z.size // nt 
  assert nt * nz == Z.size 
  if pinchCost is None:
    return None, None
  assert isinstance(pinchCost, dict)
  hdata = []
  cdata = []
  crows = []
  ccols = []
  r = 0
  for K in pinchCost.keys():
    pcK = pinchCost[K]
    tK = pcK['time']
    if tK < timePoints[0] or tK > timePoints[-1]:
      continue
    if pcK['type'] != 2: 
      continue 
    bK = pcK['bias']
    sK = pcK['sigma']
    assert bK.size == nz 
    assert sK.size == nz
    equalityK = (sK == 0.0)
    neqv = np.sum(equalityK)
    if neqv == 0:
      continue 
    k1 = np.where(timePoints >= tK)[0][0]
    if k1 == 0:
      k0 = 0
      k1 = 1
    else:
      k0 = k1 - 1
    tk0 = timePoints[k0]
    tk1 = timePoints[k1]
    xik = (tK - tk0) / (tk1 - tk0)
    assert xik >= 0.0 and xik <= 1.0 
    #
    # xi is in [0, 1]
    # z(t) = z0 * (1 - xi) + z1 * (xi)
    # cost_k = sum([ (z(t) - b(t)) / s(t)]^2)
    #
    # generate the equality constraint: z(t) - b(t) = 0
    # assuming piecewise linear interpolation within Z vector
    # and also generate the Jacobian matrix (always a constant matrix here)
    # 
    idx0 = np.arange(k0 * nz, k0 * nz + nz)
    idx1 = np.arange(k1 * nz, k1 * nz + nz)

    zk0 = Z[idx0]
    zk1 = Z[idx1]
    zxi = (1.0 - xik) * zk0 + xik * zk1 

    hxk = zxi[equalityK] - bK[equalityK]  # this is the error term
    hdata += hxk.tolist()
 
    #errK = (zxi[activeK] - bK[activeK]) / sK[activeK]
    #costK = np.sum(errK ** 2)
    #fZ += costK
    #gradfZ[idx0[activeK]] += (1.0 - xik) * 2.0 * errK / sK[activeK]
    #gradfZ[idx1[activeK]] += (xik) * 2.0 * errK / sK[activeK]

    if calcJacobian:
      Ck = np.hstack([(1.0 - xik) * np.eye(nz), xik * np.eye(nz)])[equalityK, :]
      assert Ck.shape[0] == neqv and Ck.shape[1] == 2 * nz
      blkdata, blkrow, blkcol = coo_data_from_block(Ck, rowofs = r, colofs = k0 * nz, deleteZeros = True)
      cdata += blkdata.tolist() 
      crows += blkrow.tolist()
      ccols += blkcol.tolist()

    r += neqv

  assert r == len(hdata)
  return np.array(hdata), scis.coo_array((cdata, (crows, ccols)), shape = (r, nt * nz)).tocsc() if calcJacobian else None

def sqrt_of_diag_of_inv_hess(H):
  assert scis.issparse(H)
  nh = H.shape[0]
  assert nh == H.shape[1]
  lus = scisl.splu(H)
  dinvh = np.zeros((nh, ))
  for i in range(nh):
    ei = np.zeros((nh, ))
    ei[i] = 1.0
    invhi = lus.solve(ei)
    dinvh[i] = invhi[i]
  return np.sqrt(dinvh)

#
# BASIC TEST PROGRAM 
#

if __name__ == '__main__':

  import argparse
  import matplotlib.pyplot as plt 
  import ipmlineq 

  def write_trace(t, y, ylabel, filename):
    plt.plot(t, y)
    plt.xlabel('time')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.title('num. pts. = {}'.format(len(t)))
    plt.savefig(filename)
    plt.close()

  parser = argparse.ArgumentParser()
  parser.add_argument('--nnodes', type = int, default = 125, help = 'number of transcription points (trapezoidal method)')
  parser.add_argument('--tstop', type = float, default = 10.0, help = 'stop time T: t = 0..T')
  parser.add_argument('--hess-optimize', action = 'store_true', help = 'run optimizer (using Hessian-based programming)')
  parser.add_argument('--adjust-optimum', action = 'store_true', help = 'adjust solution to have error-free dynamics')
  parser.add_argument('--umin', type = float, default = 1.0, help = 'min control level')
  parser.add_argument('--umax', type = float, default = -1.0, help = 'max control level')
  parser.add_argument('--weightu', type = float, default = 1.0, help = 'weight for controls integral')
  parser.add_argument('--sigmaw', type = float, default = 1.0e-1, help = 'sigma for dynamics stochastic term')
  parser.add_argument('--sigmap', type = float, default = 0.5e-2, help = 'sigma for point cost terms')

  args = parser.parse_args()

  # dot(z) = F(z)
  # stage vector: z = [x1, x2, x3, u]
  # dot(z) = [x2, x3, u, 0]

  def numpy_F(z):
    return np.array([z[1], z[2], z[3], 0.0])

  def numpy_gradF(z):
    return np.array([[0., 1., 0., 0.],
                     [0., 0., 1., 0.],
                     [0., 0., 0., 1.],
                     [0., 0., 0., 0.]])

  def numpy_hessF(z):
    return np.zeros((4, 4, 4))

  F = numpy_F
  gradF = numpy_gradF
  hessF = numpy_hessF

  tpts = np.linspace(0.0, args.tstop, args.nnodes)
  nz = 4

  sgma = np.ones(nz) * args.sigmaw 
  sgma[3] = 0.0       # mark as not part of "dynamics state"

  wgth = np.zeros(nz)
  wgth[3] = args.weightu        # the 4th stage variable (control input) has an active integral weight
  Z = np.random.randn(len(tpts) * nz)

  enableControlBounds = (args.umin <= args.umax)
  Eineq = None 
  fineq = None 

  if enableControlBounds and args.hess_optimize:
    zmin = np.array([-np.inf, -np.inf, -np.inf, args.umin])
    zmax = np.array([ np.inf,  np.inf,  np.inf, args.umax])
    Eineq, fineq = make_bounds_E_f(zmin, zmax, len(tpts))
    print(Eineq.shape)
    print(fineq.shape) 

  ptCost = {}
  ptCost['initial'] = { 'time' : tpts[0], 
                        'type' : int(2),
                        'bias' : np.zeros((4, )), 
                        'sigma' : args.sigmap * np.array([1.0, 1.0, 1.0, 1.0]) }
  ptCost['final']   = { 'time' : tpts[-1],
                        'type' : int(2), 
                        'bias' : np.zeros((4, )), 
                        'sigma' : args.sigmap * np.array([1.0, 1.0, 1.0, 1.0]) }
  ptCost['waypoint1']  = { 'time' : 0.234567 * args.tstop,
                           'type' : int(2), 
                           'bias' : np.array([0.0, 1.0, 0.0, np.nan]), 
                           'sigma' : args.sigmap * np.array([1.0, 1.0, 1.0, -1.0]) }
  ptCost['waypoint2']  = { 'time' : 0.754321 * args.tstop,
                           'type' : int(2), 
                           'bias' : np.array([1.0, 0.0, 0.0, np.nan]), 
                           'sigma' : args.sigmap * np.array([1.0, 1.0, 1.0, -1.0]) }

  ptCostEq = {}
  for K in ptCost.keys():
    ptCostEq[K] = ptCost[K].copy()

  ptCostEq['initial']['sigma'] = np.array([0.0, 0.0, 0.0, 0.0])
  ptCostEq['final']['sigma'] = np.array([0.0, 0.0, 0.0, 0.0])
  ptCostEq['waypoint1']['sigma'] = np.array([0.0, 0.0, 0.0, -1.0])
  ptCostEq['waypoint2']['sigma'] = np.array([0.0, 0.0, 0.0, -1.0])

  if args.hess_optimize:
    def localfgh(a):
      a, ga, ha = totalObjective(a, F, gradF, hessF, tpts, sgma, wgth, None, ptCost)
      return a, ga, ha

    if enableControlBounds:
      assert Eineq.shape[0] == fineq.size 
      print('has {} linear inequalities'.format(fineq.size))
      xinit = (-1.0 + 2.0 * np.random.rand(len(tpts) * nz)) * 0.10
      result_opt = ipmlineq.solveSparse(localfgh, Eineq, fineq, x0 = xinit, verbose = True)

    else:
      print('running unconstrained method') 
      xinit = (-1.0 + 2.0 * np.random.rand(len(tpts) * nz)) * 0.10
      result_opt = ipmlineq.solveDenseOrSparseUnc(localfgh, x0 = xinit, verbose = True)

    if result_opt['converged']:
      print('solved after {} iterations with f* = {}'.format(result_opt['iters'], result_opt['fx']))

      Zopt = result_opt['x'].reshape((len(tpts), nz))
      for k in range(nz):
        write_trace(tpts, Zopt[:, k], 'z{}'.format(k + 1), 'hess-z{}.png'.format(k + 1))

      print('calculating inverse Hessian at optimum..')
      _, _, Hess_at_Zopt = localfgh(Zopt.flatten())
      sigma_at_zopt = sqrt_of_diag_of_inv_hess(Hess_at_Zopt)
      print(np.mean(sigma_at_zopt.reshape((len(tpts), 4)), axis = 0))

  if args.hess_optimize and result_opt['converged'] and args.adjust_optimum:
    print('attempting to adjust solution to have exact dynamics...')
    import dampnewt

    sgmax = np.array([0.0, 0.0, 0.0, -1.0]) # zero means exact; negative always means ignore

    def adjust_eq(a):
      hx1, Cx1 = eval_equality_constraint_pt(a, tpts, ptCostEq)
      hx2, Cx2 = eval_equality_constraint(a, F, gradF, tpts, sgmax)
      return np.hstack([hx1, hx2]), scis.vstack([Cx1, Cx2])

    def adjust_ineq(a):
      return Eineq.dot(a) - fineq, Eineq

    def adjust_obj(a):
      a, ga, ha = totalObjective(a, F, gradF, hessF, tpts, sgmax, wgth, None, ptCostEq)
      return a, ga, ha

    repEqIneq = dampnewt.solveEq(adjust_obj, 
                                 adjust_eq, 
                                 FG = adjust_ineq if enableControlBounds else None, 
                                 x0 = result_opt['x'], 
                                 verbosity = 1, 
                                 eta = 1.0,
                                 useEye = False, 
                                 epstop = 1.0e-8,
                                 lmbda0 = 1.0e-6)  # a very small lmbda0 seems to work here!

    if repEqIneq['converged']:
      Zx = repEqIneq['x'].reshape((len(tpts), nz))
      for k in range(nz):
        write_trace(tpts, Zx[:, k], 'z{}'.format(k + 1), 'adju-z{}.png'.format(k + 1))
    else:
      print('did not converge.')

  print('all done.')
