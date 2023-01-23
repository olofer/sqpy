#
# Basic standalone interior point methods for (convex) nonlinear programming.
# Requires NUMPY and SCIPY. Dense and sparse matrix versions.
# Minimize convex F(x), s.t. linear inequality and/or equality constraints.
#
#   min F(x), s.t. E * x <= f
#   min F(x), s.t. E * x <= f, and C * x = d
#   min F(x), s.t. C * x = d 
#   min F(x)  
#
# F(x) must return the triple: {value, gradient, hessian}.
# All problem versions use variants of Newton iterations.
#
# Run basic tests ("--check" requires CVXOPT as reference) like so:
#
#   python3 ipmlineq.py --check
#   python3 ipmlineq.py --check --eps 1.0e-8
#   python3 ipmlineq.py --check --eps 1.0e-8 --fstype 0 --dim 100
#   python3 ipmlineq.py --check --eps 1.0e-8 --fstype 1 --sparse
#   python3 ipmlineq.py --check --eps 1.0e-8 --fstype 1 --sparse --neq 2
#
# https://docs.scipy.org/doc/scipy/reference/sparse.html
# https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
#

import numpy as np 
import scipy.linalg as scil

import scipy.sparse as scis
import scipy.sparse.linalg as scisl

'''
The following subprograms handle this particular system of equations (Type 0):

[[ H  E' 0 ]    [[ dx ]    [[ r1 ]
 [ E  0  I ]  *  [ dz ]  =  [ r2 ]
 [ 0  S  L ]]    [ ds ]]    [ r3 ]]

with S = diag(s), L = diag(l)

Alternative form (Type 1):

[[ H  E' ]  * [[ dx ]  = [[ r1 ]
 [ E -S/L]]    [ dz ]]    [ r2 - r3/l ]]

with ds = r2 - E * dx

Most compact form (Type 2):

[H + E' * (L/S) * E] * dx = r1 + E' * ((l * r2 - r3) / s)

with ds = r2 - E * dx, and dz = (r3 - l * ds) / s

Type 0 and 1 are solved using general LU factorization.
Type 2 is solved with Cholesky factorization.

---

For the case with equality constraint (C * x = d) only "Type 1" is implemented:

[[ H  C' E'  ]    [[ dx ]    [[ r1 ]
 [ C  0  0   ]  *  [ dy ]  =  [ r2 ]
 [ E  0 -S/L ]]    [ dz ]]    [ r3 - r4 / l ]]

And ds = (r4 - s * dz) / l 

---

For the case with only equality constraints, the following equations is solved.

[[ H  C']  * [[ dx ]  = [[ r1 ]
 [ C  0 ]]    [ dy ]]    [ r2 ]]

And for the unconstrained case this becomes: H * dx = r1

'''

# BEGIN DENSE CODE

# Largest non-symmetric system
def factorizeDenseType0(H, E, l, s):
  nz = E.shape[0]
  nx = E.shape[1]
  R1 = np.hstack((H, E.T, np.zeros((nx, nz))))
  R2 = np.hstack((E, np.zeros((nz, nz)), np.eye(nz)))
  R3 = np.hstack((np.zeros((nz, nx)), np.diag(s), np.diag(l)))
  lu, piv = scil.lu_factor(np.vstack((R1, R2, R3)))
  return lu, piv

# Does not use: E, l, s
def solveDenseType0(lu_tuple, E, l, s, r1, r2, r3):
  X = scil.lu_solve(lu_tuple, np.hstack((r1, r2, r3)))
  nx = r1.size
  nz = r2.size
  return X[:nx], X[nx:(nx + nz)], X[(nx + nz):]

# Smaller symmetric but not pos. def system
def factorizeDenseType1(H, E, l, s):
  M = np.vstack((np.hstack((H, E.T)), np.hstack((E, np.diag(-1.0 * s / l)))))
  lu, piv = scil.lu_factor(M)
  return lu, piv

# Does not use: s
def solveDenseType1(lu_tuple, E, l, s, r1, r2, r3):
  b = np.hstack((r1, r2 - r3 / l))
  X = scil.lu_solve(lu_tuple, b)
  nx = r1.size 
  dx = X[:nx]
  dz = X[nx:]
  ds = r2 - np.dot(E, dx)
  return dx, dz, ds

# Smallest (and usually) pos. def system (but less accurate)
def factorizeDenseType2(H, E, l, s):
  tmp1 = np.dot(np.diag(np.sqrt(l / s)), E) 
  L, low = scil.cho_factor(H + np.dot(tmp1.T, tmp1), lower = True)
  return L, low

def solveDenseType2(cho_tuple, E, l, s, r1, r2, r3):
  r_bar = np.dot(E.T, (-r3 + l * r2) / s)
  dx = scil.cho_solve(cho_tuple, r1 + r_bar)
  ds = r2 - np.dot(E, dx)
  dz = (r3 - l * ds) / s 
  return dx, dz, ds

#
# Basic interior-point solver for general nonlinear objective with _linear_ inequalities
#
# minimize F(x) s.t. E * x <= f
#
# F(x) is a function that returns a tuple (Fx, g, H) where Fx is the scalar value F(x)
# and g = g(x) is the gradient of F w.r.t. to x and H = H(x) is the Hessian at x.
#

def solveDense(F, E, f, x0 = None, fstype = 1, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False):
  assert callable(F) 
  assert isinstance(fstype, int) and fstype >= 0 and fstype <= 2
  factorizeDense = [factorizeDenseType0, factorizeDenseType1, factorizeDenseType2][fstype]
  solveDense = [solveDenseType0, solveDenseType1, solveDenseType2][fstype]
  nz = E.shape[0]
  nx = E.shape[1]
  assert f.size == nz 
  if x0 is None:
    x = np.zeros((nx, ))
  else:
    x = np.copy(x0)
  assert x.size == nx

  valx, g, H = F(x)

  assert isinstance(valx, float)
  assert g.size == nx 
  assert H.shape[0] == nx and H.shape[1] == nx 

  Ex_m_f = np.dot(E, x) - f 
  if verbose and not np.all(Ex_m_f < 0.0):
    print('WARNING: initial x is not strictly interior')

  if isinstance(epstop, float):
    epstop = np.tile(epstop, (3, ))

  assert epstop.size == 3

  normginf = np.max(np.abs(g))
  normfinf = np.max(np.abs(f))
  normEinf = np.max(np.abs(E))
  normHinf = np.max(np.abs(H))
  etainf = np.max([normginf, normfinf, normEinf, normHinf]) 

  thrL = (1.0 + normginf) * epstop[0]
  thrs = (1.0 + normfinf) * epstop[1]
  thrmu = epstop[2]

  z = np.ones((nz, )) * np.sqrt(etainf)
  s = np.ones((nz, )) * np.sqrt(etainf)
  e = np.ones((nz, ))

  k = 0
  while True:
    if k > 0: # no need to do this twice during initialization..
      valx, g, H = F(x)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop[0]

    rL = g + np.dot(E.T, z)                # If F is a quadratic form then g = H * x + h
    rs = s + np.dot(E, x) - f              # rs = s + E * x - f
    rsz = s * z                            # rsz = s.*z
    mu = np.sum(rsz) / nz                  # mu = sum(z.*s) / nz

    if verbose:
      #print([np.max(np.abs(rL)), np.max(np.abs(rs)), mu])
      print('mu(k={}) = {}'.format(k, mu))

    isConverged = np.max(np.abs(rL)) < thrL and np.max(np.abs(rs)) < thrs and mu < thrmu 
    if isConverged:
      break

    if k == kmax:
      break

    decomp = factorizeDense(H, E, z, s)

    dx_, dz_, ds_ = solveDense(decomp, E, z, s, -rL, -rs, -rsz)

    akp = 1 / np.max([1.0, np.max(-dz_ / z), np.max(-ds_ / s)])
    sigma = (np.sum((z + akp * dz_) * (s + akp * ds_)) / (nz * mu)) ** 3 
    rsz += dz_ * ds_ - sigma * mu * e 

    dx, dz, ds = solveDense(decomp, E, z, s, -rL, -rs, -rsz)

    akpc = 1.0
    idxz = dz < 0
    if np.sum(idxz) > 0:
      akpc = np.min([akpc, 1 / np.max(-dz[idxz] / z[idxz])])

    idxs = ds < 0
    if np.sum(idxs) > 0:
      akpc = np.min([akpc, 1 / np.max(-ds[idxs] / s[idxs])])

    akpc *= eta 
    x += akpc * dx
    z += akpc * dz
    s += akpc * ds

    assert np.all(s > 0)
    assert np.all(z > 0)

    k += 1 # did one update; go to next check and (possibly) next update

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : False,
           'x' : x,
           'fx' : valx,
           'mu' : mu,
           'res1' : np.max(np.abs(rL)),
           'res2' : np.max(np.abs(rs)),
           'z' : z,
           's' : s }

# BEGIN SPARSE CODE

# NOTE: SCIPY.LINALG.SPLU(.) issues efficiency warnings if the format is not CSC, so force it 
def factorizeSparseType0(H, E, l, s):
  nx = E.shape[1]
  nz = E.shape[0]
  R1 = scis.hstack([H, E.T, scis.csc_matrix((nx, nz))])
  R2 = scis.hstack([E, scis.csc_matrix((nz, nz)), scis.eye(nz)])
  R3 = scis.hstack([scis.csc_matrix((nz, nx)), scis.diags(s), scis.diags(l)])
  return scisl.splu(scis.vstack([R1, R2, R3], format = 'csc')) 

# Not using E, l, s
def solveSparseType0(lusolver, E, l, s, r1, r2, r3):
  b = np.hstack((r1, r2, r3))
  X = lusolver.solve(b)
  assert isinstance(X, np.ndarray)
  assert len(X.shape) == 1
  nx = r1.size 
  nz = r2.size 
  return X[:nx], X[nx:(nx + nz)], X[(nx + nz):]

def factorizeSparseType1(H, E, l, s):
  M = scis.vstack([scis.hstack([H, E.T]), scis.hstack([E, scis.diags(-1.0 * s / l)])], format = 'csc')
  assert scis.issparse(M) 
  return scisl.splu(M)

# Not using: s
def solveSparseType1(lusolver, E, l, s, r1, r2, r3):
  b = np.hstack((r1, r2 - r3 / l))
  X = lusolver.solve(b)
  assert isinstance(X, np.ndarray)
  assert len(X.shape) == 1
  nx = r1.size 
  dx = X[:nx]
  dz = X[nx:]
  ds = r2 - E.dot(dx)
  return dx, dz, ds

def factorizeSparseType2(H, E, l, s):
  tmp1 = scis.diags(np.sqrt(l / s)).dot(E)
  M = H + (tmp1.T).dot(tmp1) 
  return scisl.splu(M.tocsc())

def solveSparseType2(lusolver, E, l, s, r1, r2, r3):
  r_bar = E.T.dot((-r3 + l * r2) / s)
  dx = lusolver.solve(r1 + r_bar)
  assert isinstance(dx, np.ndarray)
  assert len(dx.shape) == 1
  ds = r2 - E.dot(dx)
  dz = (r3 - l * ds) / s 
  return dx, dz, ds

# The sparse solver is identical to the dense solver "in spirit", but care must be taken to avoid using
# NUMPY on the sparse matrices incorrectly; so keep the sparse solver fully separate from the dense solver.
# The solver is agnostic as to which subtype of sparse SCIPY matrices are provided (for E and H).

def solveSparse(F, E, f, x0 = None, fstype = 1, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False):
  assert callable(F) 
  assert isinstance(fstype, int) and fstype >= 0 and fstype <= 2
  factorizeSparse = [factorizeSparseType0, factorizeSparseType1, factorizeSparseType2][fstype]
  solveSparse = [solveSparseType0, solveSparseType1, solveSparseType2][fstype]

  assert scis.issparse(E)

  nz = E.shape[0]
  nx = E.shape[1]
  assert f.size == nz 

  if x0 is None:
    x = np.zeros((nx, ))
  else:
    x = np.copy(x0)
  assert x.size == nx

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert g.size == nx 
  assert scis.issparse(H)
  assert H.shape[0] == nx and H.shape[1] == nx 

  Ex_m_f = E.dot(x) - f 
  if verbose and not np.all(Ex_m_f < 0.0):
    print('WARNING: initial x is not strictly interior')

  if isinstance(epstop, float):
    epstop = np.tile(epstop, (3, ))

  assert epstop.size == 3

  normginf = np.max(np.abs(g))
  normfinf = np.max(np.abs(f))
  normEinf = np.max(np.abs(E.data)) 
  normHinf = np.max(np.abs(H.data))
  etainf = np.max([normginf, normfinf, normEinf, normHinf]) 

  thrL = (1.0 + normginf) * epstop[0]
  thrs = (1.0 + normfinf) * epstop[1]
  thrmu = epstop[2]

  z = np.ones((nz, )) * np.sqrt(etainf)
  s = np.ones((nz, )) * np.sqrt(etainf)
  e = np.ones((nz, ))

  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop[0]

    rL = g + E.T.dot(z)
    rs = s + E.dot(x) - f
    rsz = s * z
    mu = np.sum(rsz) / nz

    if verbose:
      print('mu(k={}) = {}'.format(k, mu))

    isConverged = np.max(np.abs(rL)) < thrL and np.max(np.abs(rs)) < thrs and mu < thrmu 
    if isConverged:
      break

    if k == kmax:
      break

    decomp = factorizeSparse(H, E, z, s)

    dx_, dz_, ds_ = solveSparse(decomp, E, z, s, -rL, -rs, -rsz)

    akp = 1 / np.max([1.0, np.max(-dz_ / z), np.max(-ds_ / s)])
    sigma = (np.sum((z + akp * dz_) * (s + akp * ds_)) / (nz * mu)) ** 3 
    rsz += dz_ * ds_ - sigma * mu * e 

    dx, dz, ds = solveSparse(decomp, E, z, s, -rL, -rs, -rsz)

    akpc = 1.0
    idxz = dz < 0
    if np.sum(idxz) > 0:
      akpc = np.min([akpc, 1 / np.max(-dz[idxz] / z[idxz])])

    idxs = ds < 0
    if np.sum(idxs) > 0:
      akpc = np.min([akpc, 1 / np.max(-ds[idxs] / s[idxs])])

    akpc *= eta 
    x += akpc * dx
    z += akpc * dz
    s += akpc * ds

    assert np.all(s > 0)
    assert np.all(z > 0)
    k += 1 

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : True,
           'x' : x,
           'fx' : valx,
           'mu' : mu,
           'res1' : np.max(np.abs(rL)),
           'res2' : np.max(np.abs(rs)),
           'z' : z,
           's' : s }

# BEGIN EQ. + INEQ. SOLVER CODE

def factorizeEqDense1(H, E, C, l, s):
  nz = E.shape[0]
  ny = C.shape[0]
  R1 = np.hstack([H, C.T, E.T])
  R2 = np.hstack([C, np.zeros((ny, ny)), np.zeros((ny, nz))])
  R3 = np.hstack([E, np.zeros((nz, ny)), np.diag(-1.0 * s / l)])
  M = np.vstack([R1, R2, R3])
  lu, piv = scil.lu_factor(M)
  return lu, piv

def solveEqDense1(lu_tuple, l, s, r1, r2, r3, r4):
  b = np.hstack((r1, r2, r3 - r4 / l))
  X = scil.lu_solve(lu_tuple, b)
  nx = r1.size 
  ny = r2.size
  nz = r3.size
  dx = X[:nx]
  dy = X[nx:(nx + ny)]
  dz = X[(nx + ny):]
  ds = (r4 - s * dz) / l
  return dx, dy, dz, ds

def factorizeEqSparse1(H, E, C, l, s):
  nz = E.shape[0]
  ny = C.shape[0]
  R1 = scis.hstack([H, C.T, E.T])
  R2 = scis.hstack([C, scis.csc_matrix((ny, ny)), scis.csc_matrix((ny, nz))])
  R3 = scis.hstack([E, scis.csc_matrix((nz, ny)), scis.diags(-1.0 * s / l)])
  return scisl.splu(scis.vstack([R1, R2, R3], format = 'csc'))

def solveEqSparse1(lusolver, l, s, r1, r2, r3, r4):
  b = np.hstack((r1, r2, r3 - r4 / l))
  X = lusolver.solve(b)
  assert isinstance(X, np.ndarray)
  assert len(X.shape) == 1
  nx = r1.size 
  ny = r2.size
  nz = r3.size
  dx = X[:nx]
  dy = X[nx:(nx + ny)]
  dz = X[(nx + ny):]
  ds = (r4 - s * dz) / l
  return dx, dy, dz, ds

#
# min F(x), s.t. E * x <= f, C * x = d
#
# Both E,f and C,d need to be specified 
# Uses sparse methods if any of {E, C, H} is sparse, otherwise uses dense methods.
#
def solveDenseOrSparseEq(F, E, f, C, d, x0 = None, y0 = None, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False):
  assert callable(F)
  assert isinstance(f, np.ndarray)
  assert isinstance(d, np.ndarray)

  isSparseE = scis.issparse(E)
  isSparseC = scis.issparse(C)

  nx = E.shape[1]
  nz = E.shape[0]
  ny = C.shape[0]

  assert C.shape[1] == nx
  assert f.size == nz 
  assert d.size == ny 

  if x0 is None:
    x = np.zeros((nx, ))
  else:
    x = np.copy(x0)
  assert x.size == nx

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert g.size == nx 
  isSparseH = scis.issparse(H)

  assert H.shape[0] == nx and H.shape[1] == nx

  Ex_m_f = E.dot(x) - f 
  if verbose and not np.all(Ex_m_f < 0.0):
    print('WARNING: initial x is not strictly interior')

  if isinstance(epstop, float):
    epstop = np.tile(epstop, (4, ))

  assert epstop.size == 4

  normginf = np.max(np.abs(g))
  normdinf = np.max(np.abs(d))
  normfinf = np.max(np.abs(f))
  normCinf = np.max(np.abs(C.data)) if isSparseC else np.max(np.abs(C))
  normEinf = np.max(np.abs(E.data)) if isSparseE else np.max(np.abs(E))
  normHinf = np.max(np.abs(H.data)) if isSparseH else np.max(np.abs(H))
  etainf = np.max([normginf, normdinf, normfinf, normEinf, normHinf, normCinf]) 

  thrE = (1.0 + normdinf) * epstop[0]
  thrL = (1.0 + normginf) * epstop[1]
  thrs = (1.0 + normfinf) * epstop[2]
  thrmu = epstop[3]

  if y0 is None:
    y = np.zeros((ny, ))
  else:
    y = np.copy(y0)
  assert y.size == ny 

  z = np.ones((nz, )) * np.sqrt(etainf)
  s = np.ones((nz, )) * np.sqrt(etainf)
  e = np.ones((nz, ))

  useSparseEq = isSparseE or isSparseH or isSparseC
  factorizeEq = factorizeEqSparse1 if useSparseEq else factorizeEqDense1
  solveEq = solveEqSparse1 if useSparseEq else solveEqDense1

  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop[1]

    rL = g + C.T.dot(y) + E.T.dot(z)
    rE = C.dot(x) - d 
    rs = s + E.dot(x) - f
    rsz = s * z
    mu = np.sum(rsz) / nz

    if verbose:
      print('mu(k={}) = {}'.format(k, mu))

    isConverged = np.max(np.abs(rL)) < thrL and np.max(np.abs(rs)) < thrs and mu < thrmu and np.max(np.abs(rE)) < thrE 
    if isConverged:
      break

    if k == kmax:
      break

    decomp = factorizeEq(H, E, C, z, s)
    dx_a, dy_a, dz_a, ds_a = solveEq(decomp, z, s, -rL, -rE, -rs, -rsz)

    alpha_a = 1.0
    idxz = dz_a < 0
    if np.sum(idxz) > 0:
      alpha_a = np.min([alpha_a, np.min(-1.0 * z[idxz] / dz_a[idxz])])

    idxs = ds_a < 0
    if np.sum(idxs) > 0:
      alpha_a = np.min([alpha_a, np.min(-1.0 * s[idxs] / ds_a[idxs])])

    mu_a = np.sum((z + alpha_a * dz_a) * (s + alpha_a * ds_a)) / nz
    sigma = (mu_a / mu) ** 3

    rsz += ds_a * dz_a - sigma * mu * e 

    dx, dy, dz, ds = solveEq(decomp, z, s, -rL, -rE, -rs, -rsz)

    alpha = 1.0
    idxz = dz < 0
    if np.sum(idxz) > 0:
      alpha = np.min([alpha, np.min(-1.0 * z[idxz] / dz[idxz])])

    idxs = ds < 0
    if np.sum(idxs) > 0:
      alpha = np.min([alpha, np.min(-1.0 * s[idxs] / ds[idxs])])

    ea = eta * alpha 
    x += ea * dx
    y += ea * dy
    z += ea * dz
    s += ea * ds 

    k += 1

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : useSparseEq,
           'x' : x,
           'fx' : valx,
           'mu' : mu,
           'res1' : np.max(np.abs(rL)),
           'res2' : np.max(np.abs(rs)),
           'res3' : np.max(np.abs(rE)),
           'y' : y,
           'z' : z,
           's' : s }

#
# min F(x), s.t. C * x = d ; basic Newton method to locate stationary point x (incl Lagrange multiplier y).
#
def solveDenseOrSparseOnlyEq(F, C, d, x0 = None, y0 = None, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False, forcePos = False):
  assert callable(F)
  assert isinstance(d, np.ndarray)

  isSparseC = scis.issparse(C)

  nx = C.shape[1]
  ny = C.shape[0]
 
  assert d.size == ny 

  if x0 is None:
    x = np.zeros((nx, ))
  else:
    x = np.copy(x0)
  assert x.size == nx

  if forcePos:
    assert np.all(x > 0)

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert g.size == nx 
  isSparseH = scis.issparse(H)

  assert H.shape[0] == nx and H.shape[1] == nx

  useSparse = isSparseC or isSparseH

  if y0 is None:
    y = np.zeros((ny, ))
  else:
    y = np.copy(y0)
  assert y.size == ny

  if isinstance(epstop, float):
    epstop = np.tile(epstop, (2, ))
  assert epstop.size == 2

  normginf = np.max(np.abs(g))
  normdinf = np.max(np.abs(d))
  thrE = (1.0 + normdinf) * epstop[0]
  thrL = (1.0 + normginf) * epstop[1]

  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop[1]

    rL = g + C.T.dot(y) 
    rE = C.dot(x) - d 

    if verbose:
      print('k={} : rL = {}, rE = {}'.format(k, np.max(np.abs(rL)), np.max(np.abs(rE))))

    isConverged = np.max(np.abs(rL)) < thrL and np.max(np.abs(rE)) < thrE 
    if isConverged:
      break

    if k == kmax:
      break

    brhs = np.hstack([-rL, -rE])

    if useSparse:
      M = scis.vstack([scis.hstack([H, C.T]), scis.hstack([C, scis.csc_matrix((ny, ny))])], format = 'csc')
      dxy = scis.linalg.spsolve(M, brhs)
    else:
      M = np.vstack([np.hstack([H, C.T]), np.hstack([C, np.zeros((ny, ny))])])
      dxy = np.linalg.solve(M, brhs)

    assert isinstance(dxy, np.ndarray) and len(dxy.shape) == 1 and dxy.size == nx + ny 

    dx = dxy[:nx]
    dy = dxy[nx:]

    if forcePos:
      assert np.all(x > 0.0)
      while np.any(x + eta * dx <= 0.0):
        dx *= 0.5
        dy *= 0.5

    x += eta * dx 
    y += eta * dy

    k += 1

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : useSparse,
           'x' : x,
           'fx' : valx,
           'res1' : np.max(np.abs(rL)),
           'res2' : np.max(np.abs(rE)),
           'y' : y }

def solveDenseOrSparseUnc(F, x0 = None, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False, forcePos = False):
  assert callable(F)
  assert not x0 is None

  x = np.copy(x0)
  assert len(x.shape) == 1
  nx = x.size 
  assert np.all(np.isfinite(x))

  if forcePos:
    assert np.all(x > 0)

  valx, g, H = F(x)

  assert isinstance(valx, float)   # type checking first time calling F
  assert isinstance(g, np.ndarray)
  assert nx == g.size  
  useSparse = scis.issparse(H)
  assert H.shape[0] == nx and H.shape[1] == nx

  assert isinstance(epstop, float)

  normginf = np.max(np.abs(g))
  thrL = (1.0 + normginf) * epstop

  k = 0
  while True:
    if k > 0: 
      valx, g, H = F(x)
      normginf = np.max(np.abs(g))
      thrL = (1.0 + normginf) * epstop

    rL = g 

    if verbose:
      print('k={} : rL = {}'.format(k, np.max(np.abs(rL))))

    isConverged = np.max(np.abs(rL)) < thrL  
    if isConverged:
      break

    if k == kmax:
      break

    if useSparse:
      dx = scis.linalg.spsolve(H, -rL)
    else:
      dx = np.linalg.solve(H, -rL)

    assert isinstance(dx, np.ndarray) and len(dx.shape) == 1 and dx.size == nx

    if forcePos:
      assert np.all(x > 0.0)
      while np.any(x + eta * dx <= 0.0):
        dx *= 0.5

    x += eta * dx 

    k += 1

  return { 'iters' : k, 
           'converged' : isConverged,
           'sparse' : useSparse,
           'x' : x,
           'fx' : valx,
           'res1' : np.max(np.abs(rL)) }

#
# Automatic simple common interface
#
def solve(F, E = None, f = None, C = None, d = None, x0 = None, fstype = 1, kmax = 50, epstop = 1.0e-6, eta = 0.95, verbose = False):
  if not E is None and not C is None:
    assert not d is None
    assert not f is None 

    if verbose and fstype != 1:
      print('NOTICE: only fstype = 1 supported for equality constrained solver')

    return solveDenseOrSparseEq(F, E, f, C, d, x0 = x0, y0 = None, kmax = kmax, epstop = epstop, eta = eta, verbose = verbose)

  if not E is None and C is None:
    assert not f is None

    if scis.issparse(E):
      return solveSparse(F, E, f, x0 = x0, fstype = fstype, kmax = kmax, epstop = epstop, eta = eta, verbose = verbose)
    else:
      return solveDense(F, E, f, x0 = x0, fstype = fstype, kmax = kmax, epstop = epstop, eta = eta, verbose = verbose)

  if E is None and not C is None:
    assert not d is None 
    return solveDenseOrSparseOnlyEq(F, C, d, x0 = x0, y0 = None, kmax = kmax, epstop = epstop, eta = eta, verbose = verbose)

  assert E is None and C is None 
  return solveDenseOrSparseUnc(F, x0 = x0, kmax = kmax, epstop = epstop, eta = eta, verbose = verbose)

# BEGIN TEST/MAIN CODE

if __name__ == '__main__':
 
  import argparse 
  parser = argparse.ArgumentParser()
  parser.add_argument('--dim', type = int, default = 50, help = 'dimension of test problem')
  parser.add_argument('--fstype', type = int, default = 1, help = 'inner equation solver type : {0, 1, 2}')
  parser.add_argument('--eps', type = float, default = 1.0e-6, help = 'attempted accuracy requirement for solver')
  parser.add_argument('--check', action = 'store_true')
  parser.add_argument('--sparse', action = 'store_true')
  parser.add_argument('--neq', type = int, default = 0, help = 'number of random equality constraints to add')
  args = parser.parse_args()

  assert args.dim > 0

  # Create an objective function (A*x - b) ** 2
  # And impose box constraints |x| <= 1 elementwise

  print('random dense quadratic test program with box-constraints (dim. = {})'.format(args.dim))

  A = np.random.randn(args.dim, args.dim) 
  b = np.random.randn(args.dim) 

  E = np.vstack((np.eye(args.dim), -1.0 * np.eye(args.dim))) # E*x <= fmax and E*x >= fmin
  xmax = np.ones((args.dim, ))
  xmin = -1.0 * np.ones((args.dim, ))
  f = np.hstack((xmax, -xmin))
  assert f.size == E.shape[0]
  assert E.shape[1] == args.dim
  assert E.shape[0] == 2 * args.dim 

  def obj(x):
    err = np.dot(A, x) - b
    valu = 0.5 * np.sum(err ** 2)
    grad = np.dot(A.T, err)
    hess = np.dot(A.T, A)
    return valu, grad, hess

  xsol = np.linalg.solve(A, b)
  fsol, gsol, hsol = obj(xsol) # unconstrained solution value ...

  result_unc = solveDenseOrSparseUnc(obj, x0 = np.zeros((args.dim, )), epstop = args.eps, eta = 0.999)
  assert result_unc['converged']
  print('solved unconstrained after {} iterations: max. |diff| = {}'.format(result_unc['iters'], np.max(np.abs(result_unc['x'] - xsol))))

  result = solveDense(obj, E, f, epstop = args.eps, fstype = args.fstype)
  sol_x = result['x']
  ofs_term = 0.5 * np.sum(b * b)

  assert result['converged']
  print('solved after {} iterations with f* = {}'.format(result['iters'], result['fx']))

  if args.sparse:
    def obj_sparse(x):
      v, g, h = obj(x)
      return v, g, scis.csr_array(h)
    
    result_sparse = solveSparse(obj_sparse, scis.csr_array(E), f, epstop = args.eps, fstype = args.fstype)
    assert result_sparse['converged']
    print('solved after {} iterations with f* = {} (dense-as-sparse)'.format(result_sparse['iters'], result_sparse['fx']))

    sol_x_sparse = result_sparse['x']
    print('dense-as-sparse: largest element absolute difference = {}'.format(np.max(np.abs(sol_x - sol_x_sparse))))

  if args.check:

    print('checking against CVXOPT')

    # https://cvxopt.org/userguide/coneprog.html#quadratic-programming

    import cvxopt

    cvxopt.solvers.options['reltol'] = args.eps
    cvxopt.solvers.options['show_progress'] = False

    H = cvxopt.matrix(np.dot(A.T, A))
    h = cvxopt.matrix(-1.0 * np.dot(A.T, b))
    ref_solve = cvxopt.solvers.qp(H, h, cvxopt.matrix(E), cvxopt.matrix(f))

    ref_x = np.array(ref_solve['x']).flatten()

    err_x = sol_x - ref_x
    print('largest absolute element difference cmp. to CVXOPT = {}'.format(np.max(np.abs(err_x))))

    err_fx = result['fx'] - (ref_solve['primal objective'] + ofs_term)
    print('absolute objective difference cmp. to CVXOPT = {}'.format(np.abs(err_fx)))

  # Whenever --neq > 0 is provided; add neq random equality constraints to the random test problem and solve again (with different solvers)
  if args.neq > 0:
    C = np.random.randn(args.neq, args.dim)
    d = np.random.randn(args.neq)
    eqresult = solveDenseOrSparseEq(obj, E, f, C, d, epstop = args.eps)
    assert eqresult['converged']
    assert not eqresult['sparse']
    print('solved after {} iterations with f* = {} (with {} eq. contraints)'.format(eqresult['iters'], eqresult['fx'], args.neq))

  if args.neq > 0 and args.sparse:
    eqresult_sparse = solveDenseOrSparseEq(obj_sparse, scis.csr_array(E), f, scis.csr_array(C), d, epstop = args.eps)
    assert eqresult_sparse['converged']
    assert eqresult_sparse['sparse']
    print('solved after {} iterations with f* = {} (with {} eq. contraints "dense-as-sparse")'.format(eqresult_sparse['iters'], eqresult_sparse['fx'], args.neq))

  if args.neq > 0 and args.check:
    eq_ref_solve = cvxopt.solvers.qp(H, h, cvxopt.matrix(E), cvxopt.matrix(f), A=cvxopt.matrix(C), b=cvxopt.matrix(d))
    eq_ref_x = np.array(eq_ref_solve['x']).flatten()

    eq_err_x = eqresult['x'] - eq_ref_x
    print('largest absolute element difference cmp. to CVXOPT = {} (with {} eq. constraints)'.format(np.max(np.abs(eq_err_x)), args.neq))

    eq_err_fx = eqresult['fx'] - (eq_ref_solve['primal objective'] + ofs_term)
    print('absolute objective difference cmp. to CVXOPT = {} (with {} eq. constraints)'.format(np.abs(err_fx), args.neq))

  if args.neq > 0:
    ny = args.neq
    nx = args.dim 
    eqonly_solxy = np.linalg.solve(np.vstack([np.hstack([np.dot(A.T, A), C.T]), np.hstack([C, np.zeros((ny, ny))])]), np.hstack([np.dot(A.T, b), d]))
    result_eqonly = solveDenseOrSparseOnlyEq(obj, C, d, epstop = args.eps, eta = 0.999)
    assert result_eqonly['converged']
    print('solved eq. only constrained problem after {} iterations: max. |diff| = {}'.format(result_eqonly['iters'], np.max(np.abs(result_eqonly['x'] - eqonly_solxy[:nx]))))

  print('test program done.') 
