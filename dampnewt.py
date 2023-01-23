#
# Nonlinear programming (SQP type):
#
# minimize F(x), optionally subject to FH(x) = 0 and/or FG(x) <= 0
#
# -- F is a function returning tuple {value, gradient, Hessian} for objective @ x.
# -- FH is a function returning tuple {residual, gradient} for equality constraint @ x.
# -- FG is a function returning tuple {residual, gradient} for inequality constraint @ x.
# -- always require an initial guess x = x0
#
# Example: for a linear inequality constraint E*x <= f, FG(x) would return the tuple (E*x-f, E).
#
# Hessian-based optimization with adaptive trust region.
# Newton damping scheme similar to Levenberg-Marquart.
#
# These solvers are implemented:
#
# 1. solve(.) SQP (if needed); optional linearized inequality constraint FG
# 2. solveEq(.) SQP with mandatory linearized equality constraints FH, and optional linearized inequality constraint FG
#
# (TBD) 3. solveMu(.) attempts to integrate barrier update with damped Newton stepping 
# 

import numpy as np 
import scipy.sparse as scis 
import ipmlineq

def default_ratio_opts():
  return { 'ratio-dec'  : 0.25,
           'factor-dec' : 0.50,
           'ratio-inc'  : 0.25,
           'factor-inc' : 2.0 }

def default_inner_opts():
  return { 'epstop'  : -1.0, # 0 or less will force update based on outer epstop
           'fstype'  : int(1),
           'kmax'    : int(50),
           'verbose' : False,
           'eta'     : 0.95 }

def solve(F, 
          x0 = None,
          FG = None, 
          qpopts = None,
          lmbda0 = 25.0, 
          kmax = 100, 
          epstop = 1.0e-6, 
          eta = 1.0, 
          verbosity = 1,
          useEye = False,
          ropts = None):

  assert callable(F)
  assert not x0 is None

  x = np.copy(x0)
  assert len(x.shape) == 1
  nx = x.size 
  assert np.all(np.isfinite(x))

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert nx == g.size  
  useSparse = scis.issparse(H)
  assert H.shape[0] == nx and H.shape[1] == nx
  assert isinstance(epstop, float)
  assert isinstance(eta, float) and eta > 0.0

  if ropts is None:
    ropts = default_ratio_opts()
  assert isinstance(ropts, dict)

  hasLinearIneqs = not FG is None 
  if hasLinearIneqs:
    assert callable(FG) 
    gx, Ex = FG(x) 
    assert isinstance(gx, np.ndarray)
    niq = gx.size 
    assert Ex.shape[0] == niq and Ex.shape[1] == nx
    if np.any(gx >= 0):
      print('warning: initial x is not (strictly) interior')
    if qpopts is None:
      qpopts = default_inner_opts()
      if qpopts['epstop'] <= 0:
        qpopts['epstop'] = epstop / 10.0
    assert isinstance(qpopts, dict)

  diagH = H.diagonal() if useSparse else np.diag(H)
  normginf = np.max(np.abs(g))
  thrL = (1.0 + normginf) * epstop

  hasSwitchedToSQP = False 
  lmbda = lmbda0
  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      diagH = H.diagonal() if useSparse else np.diag(H)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop

      if hasLinearIneqs:
        gx, Ex = FG(x)

      actual_improvement = valx - prev_valx 
      predct_improvement = valx - pred_valx 
      ratio_improv = predct_improvement / actual_improvement

      rel_val_change = actual_improvement / (1.0 + np.abs(valx))
      prev_rel_dx = np.max(np.abs(dx)) / (1.0 + np.max(np.abs(x)))

      if verbosity > 1:
        print('ratio pred/actual = {:.6f}, rel step = {:.8e}'.format(ratio_improv, prev_rel_dx))

      if ratio_improv > ropts['ratio-dec']:
        lmbda *= ropts['factor-dec']
      elif ratio_improv < ropts['ratio-inc']:
        lmbda *= ropts['factor-inc']

    if verbosity > 0:
      if k == 0: dx = np.zeros((nx, ))
      print('k={:04} @ lmbda = {:.8e} : |g| = {:.8e}, |dx| = {:.8e}, f(x) = {:.8e}'.format(k, lmbda, np.max(np.abs(g)), np.max(np.abs(dx)), valx))

    isConverged = np.max(np.abs(g)) < thrL if not hasLinearIneqs else k > 0  
    if k > 0:
      isConverged = isConverged and (rel_val_change < epstop and prev_rel_dx < epstop)

    if isConverged:
      break

    if k == kmax:
      break

    if useEye:
      regTerm = lmbda * scis.eye(nx) if useSparse else lmbda * np.eye(nx)
    else:
      regTerm = scis.diags(lmbda * diagH) if useSparse else np.diag(lmbda * diagH)

    if not hasLinearIneqs or (hasLinearIneqs and not hasSwitchedToSQP):
      if useSparse:
        dx = scis.linalg.spsolve(H + regTerm, -1.0 * g)
      else:
        dx = np.linalg.solve(H + regTerm, -1.0 * g)

    if hasLinearIneqs and not hasSwitchedToSQP:
      # Explicit check whether we would still be interior with the "trial" non-constrained update (faster if possible)
      gx_, _ = FG(x + eta * dx)
      if np.any(gx_ > 0.0):
        hasSwitchedToSQP = True 
        if verbosity > 0:
          print('switching to SQP mode')

    if hasLinearIneqs and hasSwitchedToSQP:
      # The linearization gx + Ex * dx <= 0, is sent to local QP as: Ex * z <= -1.0 * gx
      # Solve local damped QP for step "z = dx" explicitly respecting the linearized inequalitites
      def Flocal(z):
        Hz = H + regTerm
        vz = np.sum(g * z) + 0.5 * np.sum(z * Hz.dot(z)) # valx offset dropped
        gz = g + Hz.dot(z) 
        return vz, gz, Hz 

      if useSparse:
        localResult = ipmlineq.solveSparse(Flocal, 
                                           Ex, -1.0 * gx, 
                                           x0 = None, 
                                           fstype = qpopts['fstype'], 
                                           kmax = qpopts['kmax'], 
                                           epstop = qpopts['epstop'],
                                           eta = qpopts['eta'], 
                                           verbose = qpopts['verbose'])
      else:
        localResult = ipmlineq.solveDense(Flocal, 
                                          Ex, -1.0 * gx, 
                                          x0 = None, 
                                          fstype = qpopts['fstype'], 
                                          kmax = qpopts['kmax'], 
                                          epstop = qpopts['epstop'],
                                          eta = qpopts['eta'], 
                                          verbose = qpopts['verbose'])

      if not localResult['converged']:
        if verbosity > 0:
          print('local QP failed')
        break 
      else:
        if verbosity > 1:
          print('local QP solved with {} iterations'.format(localResult['iters']))

      dx = localResult['x']

    assert isinstance(dx, np.ndarray) and len(dx.shape) == 1 and dx.size == nx

    # Local quadratic approx is: valx + g'*dx + 0.5*dx'*H*dx
    prev_valx = valx 
    pred_valx = valx + eta * np.sum(g * dx) + 0.5 * eta * eta * np.sum(dx * (H + regTerm).dot(dx))
    x += eta * dx 

    if verbosity > 2:
      print('gradness = {:.6f}'.format(np.sum(g * dx) / np.sqrt( np.sum(g ** 2) * np.sum(dx ** 2) )))

    k += 1

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : useSparse,
           'niqs' : niq if hasLinearIneqs else 0,
           'usedsqp' : hasSwitchedToSQP,
           'x' : x,
           'fx' : valx,
           'infg' : np.max(np.abs(g)) }

def solveEq(F,
            FH, 
            FG = None,
            x0 = None, 
            qpopts = None,
            lmbda0 = 25.0, 
            kmax = 100, 
            epstop = 1.0e-6, 
            eta = 1.0, 
            verbosity = 1,
            useEye = False,
            ropts = None):

  assert callable(F)
  assert not x0 is None

  x = np.copy(x0)
  assert len(x.shape) == 1
  nx = x.size 
  assert np.all(np.isfinite(x))

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert nx == g.size  
  isSparseH = scis.issparse(H)
  assert H.shape[0] == nx and H.shape[1] == nx
  assert isinstance(epstop, float)
  assert isinstance(eta, float) and eta > 0.0

  if ropts is None:
    ropts = default_ratio_opts()
  assert isinstance(ropts, dict)

  diagH = H.diagonal() if isSparseH else np.diag(H)

  assert callable(FH)
  hx, Cx = FH(x)

  assert isinstance(hx, np.ndarray)
  neq = hx.size 
  assert Cx.shape[0] == neq and Cx.shape[1] == nx 

  hasIneqs = not FG is None 
  if hasIneqs:
    assert callable(FG) 
    gx, Ex = FG(x) 
    assert isinstance(gx, np.ndarray)
    niq = gx.size 
    assert Ex.shape[0] == niq and Ex.shape[1] == nx 

  # Taylor approximation of F: valx + g'*dx + 0.5*dx'*H*dx
  # Linearized constraint: hx + Cx * dx = 0
  # Optionally also: gx + Ex * dx <= 0

  if qpopts is None:
    qpopts = default_inner_opts()
    if qpopts['epstop'] <= 0:
      qpopts['epstop'] = epstop / 10.0
  assert isinstance(qpopts, dict)

  assert isinstance(lmbda0, float) and lmbda0 > 0
  lmbda = lmbda0 
  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      diagH = H.diagonal() if isSparseH else np.diag(H)
      hx, Cx = FH(x)
      if hasIneqs:
        gx, Ex = FG(x) 

      actual_improvement = valx - prev_valx 
      predct_improvement = valx - pred_valx 
      ratio_improv = predct_improvement / actual_improvement

      rel_val_change = actual_improvement / (1.0 + np.abs(valx))
      prev_rel_dx = np.max(np.abs(dx)) / (1.0 + np.max(np.abs(x)))

      if verbosity > 1:
        print('ratio pred/actual = {:.6f}, rel step = {:.8e}'.format(ratio_improv, prev_rel_dx))

      if ratio_improv > ropts['ratio-dec']:
        lmbda *= ropts['factor-dec']
      elif ratio_improv < ropts['ratio-inc']:
        lmbda *= ropts['factor-inc']

    if verbosity > 0:
      if k == 0: dx = np.zeros((nx, )) # not calculated until k > 0
      if hasIneqs:
        print('k={:04} @ lmbda = {:.8e} : |hx| = {:.8e}, max(gx) = {:.8e}, |dx| = {:.8e}, f(x) = {:.8e}'.format(k, lmbda, np.max(np.abs(hx)), np.max(gx), np.max(np.abs(dx)), valx))
      else:
        print('k={:04} @ lmbda = {:.8e} : |hx| = {:.8e}, |dx| = {:.8e}, f(x) = {:.8e}'.format(k, lmbda, np.max(np.abs(hx)), np.max(np.abs(dx)), valx))

    isConverged = rel_val_change < epstop and prev_rel_dx < epstop if k > 0 else False
    if isConverged:
      break

    if k == kmax:
      break

    if useEye:
      regTerm = lmbda * scis.eye(nx) if isSparseH else lmbda * np.eye(nx)
    else:
      regTerm = scis.diags(lmbda * diagH) if isSparseH else np.diag(lmbda * diagH)

    # Solve local damped QP for step "z = dx" explicitly respecting the linear inequalitites
    def Flocal(z):
      Hz = H + regTerm
      vz = np.sum(g * z) + 0.5 * np.sum(z * Hz.dot(z)) # valx offset dropped
      gz = g + Hz.dot(z) 
      return vz, gz, Hz 

    if not hasIneqs:
      localResult = ipmlineq.solveDenseOrSparseOnlyEq(Flocal, 
                                                      Cx, -1.0 * hx, 
                                                      x0 = None, y0 = None,
                                                      #fstype = qpopts['fstype'], 
                                                      kmax = qpopts['kmax'], 
                                                      epstop = qpopts['epstop'],
                                                      eta = qpopts['eta'], 
                                                      verbose = qpopts['verbose'])

    if hasIneqs:
      localResult = ipmlineq.solveDenseOrSparseEq(Flocal, 
                                                  Ex, -1.0 * gx, 
                                                  Cx, -1.0 * hx, 
                                                  x0 = None, y0 = None, 
                                                  #fstype = qpopts['fstype'], 
                                                  kmax = qpopts['kmax'], 
                                                  epstop = qpopts['epstop'],
                                                  eta = qpopts['eta'], 
                                                  verbose = qpopts['verbose'])

    if not localResult['converged']:
      if verbosity > 0:
        print('local QP failed')
        break 
    else:
      if verbosity > 1:
        print('local QP solved with {} iterations'.format(localResult['iters']))

    dx = localResult['x']
    assert isinstance(dx, np.ndarray) and len(dx.shape) == 1 and dx.size == nx

    prev_valx = valx 
    pred_valx = valx + eta * np.sum(g * dx) + 0.5 * eta * eta * np.sum(dx * (H + regTerm).dot(dx))
    x += eta * dx

    k += 1

  return { 'iters' : k, 
           'converged' : isConverged,
           'neqs' : neq,
           'niqs' : niq if hasIneqs else 0,
           'x' : x,
           'fx' : valx,
           'infg' : np.max(gx) if hasIneqs else np.nan,
           'infh' : np.max(np.abs(hx)) } 

'''
def solveMu(F, 
            x0 = None, 
            E = None, 
            f = None, 
            t0 = 1.0,
            mu = 10.0,
            lmbda0 = 25.0, 
            kmax = 100, 
            epstop = 1.0e-6, 
            eta = 1.0, 
            verbosity = 1,
            useEye = False):

  assert callable(F)
  assert not x0 is None

  x = np.copy(x0)
  assert len(x.shape) == 1
  nx = x.size 
  assert np.all(np.isfinite(x))

  valx, g, H = F(x)

  assert isinstance(valx, float)
  assert isinstance(g, np.ndarray)
  assert nx == g.size  
  useSparse = scis.issparse(H)
  assert H.shape[0] == nx and H.shape[1] == nx
  assert isinstance(epstop, float)
  assert isinstance(eta, float) and eta > 0.0

  assert not E is None
  assert not f is None 
  assert isinstance(f, np.ndarray)

  m = E.shape[0]
  assert f.size == m 
  assert E.shape[1] == nx 

  if useSparse and not scis.issparse(E):
    print('warning: hessian is sparse but E matrix is not sparse')

  s = f - E.dot(x)
  if np.any(s <= 0):
    print('warning: initial x is not (strictly) interior')

  # TBD: this algorithm updates the barrier parameter every T-th iteration or each time the trust region is NOT shrunk...
  assert False 

  k = 0
  while True:

    if k == kmax:
      break 

    k += 1

  return {}
'''

if __name__ == '__main__':

  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dim', type = int, default = 100, help = 'dimension of test problem')
  parser.add_argument('--eps', type = float, default = 1.0e-6, help = 'dimension of test problem')
  args = parser.parse_args()
  assert args.dim >= 1

  n = args.dim 
  M = np.random.randn(2 * n, n) / np.sqrt(float(n))
  H = M.T.dot(M)
  h = np.random.randn(n)
  
  def objfunc(x):
    f = np.sum(h * x) + 0.5 * np.sum(x * H.dot(x))
    g = h + H.dot(x) 
    return f, g, H 

  xinit = np.random.randn(n)

  print('--- UNCONSTRAINED SOLVER ---')
  rep = solve(objfunc, x0 = xinit, verbosity = 1, epstop = args.eps, useEye = False, lmbda0 = 1.0)

  if rep['converged']:
    xtrue = np.linalg.solve(H, -1.0 * h)
    print(np.max(np.abs(rep['x'] - xtrue)) / (1.0 + np.max(np.abs(xtrue))))
    print('unbounded x[{}]={}'.format(n // 2, rep['x'][n // 2]))
  else:
    print('did not converge!')

  # Constrained problem using the damped Newton method (SQP activated as needed)
  E = np.zeros((2, n))
  E[0, n // 2] = -1.0
  E[1, n // 2] = 1.0
  f = np.array([1.0, 1.0])

  def ineqFunc(x):
    return E.dot(x) - f, E

  xinit[n // 2] = -1.0 + np.random.rand() * 2.0

  print('--- INEQ. CONSTRAINED SQP SOLVER ---')
  repIneq = solve(objfunc, x0 = xinit, verbosity = 1, epstop = args.eps, useEye = False, lmbda0 = 1.0, FG = ineqFunc)

  if not repIneq['converged']:
    print('did not converge!')

  # Check above solution against pure QP solver (the problem is exactly QP anyway)
  repTrue = ipmlineq.solveDense(objfunc, E, f, x0 = None, fstype = 1, kmax = 50, epstop = args.eps, eta = 0.95, verbose = False)
  if repTrue['converged'] and repIneq['converged']:
    xtrue_bnd = repTrue['x']
    print(np.max(np.abs(repIneq['x'] - xtrue_bnd)) / (1.0 + np.max(np.abs(xtrue_bnd))))
    print('bounded x[{}]={}'.format(n // 2, repIneq['x'][n // 2]))

  # Define equality constraint "sum(z) - 1 = 0", and solve with solveEq(.)
  def eqFunc(z):
    return np.array([np.sum(z) - 1.0]), np.ones((1, z.size))

  print('--- EQ. CONSTRAINED SQP SOLVER ---')
  repEq = solveEq(objfunc, eqFunc, x0 = xinit, verbosity = 1, epstop = args.eps, useEye = False, lmbda0 = 1.0)
  if not repEq['converged']:
    print('did not converge!')
  else:
    print('sum(x) = {}'.format(np.sum(repEq['x'])))

  repEqTrue = ipmlineq.solveDenseOrSparseOnlyEq(objfunc, np.ones((1, n)), np.ones((1, )), verbose = False, epstop = args.eps)
  if repEqTrue['converged'] and repEq['converged']:
    xtrue_eq = repEqTrue['x']
    print(np.max(np.abs(repEq['x'] - xtrue_eq)) / (1.0 + np.max(np.abs(xtrue_eq))))

  # Run a final problem with BOTH eqfunc and ineqfunc
  print('--- EQ. + INEQ. CONSTRAINED SQP SOLVER ---')
  repEqIneq = solveEq(objfunc, eqFunc, FG = ineqFunc, x0 = xinit, verbosity = 1, epstop = args.eps, useEye = False, lmbda0 = 1.0)
  if not repEqIneq['converged']:
    print('did not converge!')
  else:
    print('sum(x) = {}'.format(np.sum(repEqIneq['x'])))
    print('bounded x[{}]={}'.format(n // 2, repEqIneq['x'][n // 2]))

  repEqIneqTrue = ipmlineq.solveDenseOrSparseEq(objfunc, E, f, np.ones((1, n)), np.ones((1, )), verbose = False, epstop = args.eps)
  if repEqIneqTrue['converged'] and repEqIneq['converged']:
    xtrue_eqineq = repEqIneqTrue['x']
    print(np.max(np.abs(repEqIneq['x'] - xtrue_eqineq)) / (1.0 + np.max(np.abs(xtrue_eqineq))))
  
  print('tests done.')
