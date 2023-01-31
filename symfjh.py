#
# Auto-generate basic NUMPY functions for Jacobian, Hessian
# given SYMPY list of expressions F and a SYMPY list of symbols X.
#
# Creates a string for the source code which can be used either to
# dump to a file or just exec(.) to define "here and now".
# Optionally autoformats the generated source file
#   (requires: https://pypi.org/project/black/)
#
# DEMO: python3 symfjh.py
# USAGE: import symfjh + (see demo code)
#

import numpy as np
import sympy 
import os

# Take a list of sympy symbols X and a list of sympy expressions F
# Generate the symbolic Jacobian matrix
def symJacobian(F, X):
  J = []

  for f in F:
    dfrow = []
    for x in X:
      dfrow.append(sympy.diff(f, x))

    assert len(dfrow) == len(X)
    J.append(dfrow)

  assert len(J) == len(F)
  return J

# Take a single sympy expression f and generate its Hessian w.r.t. list of symbols X
def symHessian(f, X):
  H = []
  for a in X:
    A = []
    for b in X:
      A.append(sympy.diff(sympy.diff(f, a), b))

    assert len(A) == len(X)
    H.append(A)

  assert len(H) == len(X)
  return H

def symHessianCube(F, X):
  return [symHessian(e, X) for e in F]

def create_def_string(funcname, F, X, P = None):
  S = repr(F)

  for k in range(len(X)):
    xk = repr(X[k])
    S = S.replace(xk, 'x[{}]'.format(k))

  if not P is None:
    for k in range(len(P)):
      pk = repr(P[k])
      S = S.replace(pk, 'p[{}]'.format(k))

  for npname in ['sin', 'cos', 'exp', 'log', 'sqrt']:
    S = S.replace(npname, 'np.{}'.format(npname))

  argstr = '(x, p)' if not P is None else '(x)'
  return 'def {}{}: return np.array({}, dtype=float)'.format(funcname, argstr, S)

def evalf_recursive(item, subsdict):
  if isinstance(item, list):
    return [evalf_recursive(x, subsdict) for x in item]
  else:
    # Assume this is a symbolic expression object
    return float(item.evalf(subs = subsdict))

def write_python_file(L, 
                      pyfname, 
                      X = None, 
                      P = None, 
                      npimport = True, 
                      autoformat = True):

  assert isinstance(L, list)
  
  with open(pyfname, 'w') as f:
    if not X is None:
      f.write('# X = {}\n'.format(repr(X)))
    if not P is None:
      f.write('# P = {}\n'.format(repr(P)))
    if npimport:
      f.write('import numpy as np\n')
    for l in L:
      f.write(l + '\n')

  rval = os.system('black --quiet {}'.format(pyfname)) if autoformat else 0  
  return rval 

if __name__ == '__main__':
 
  import argparse 

  parser = argparse.ArgumentParser()
  parser.add_argument('--no-genfile', action = 'store_true')
  parser.add_argument('--no-black', action = 'store_true')
  parser.add_argument('--genfile', type = str, default = 'generated_demo_code')
  parser.add_argument('--ncheck', type = int, default = 250)
  parser.add_argument('--atol', type = float, default = 1.0e-12)
  parser.add_argument('--rtol', type = float, default = 1.0e-12)

  args = parser.parse_args()

  # Demonstration / example 

  x1, x2, x3, p1, p2, p3 = sympy.symbols('x1,x2,x3,p1,p2,p3')

  f1 = x1 + x2 + x3
  f2 = x1 ** 2 + x2 ** 2 + x3 ** 2
  f3 = sympy.cos(x1) + sympy.sin(x2) + sympy.cos(2.0 * x3)

  F = [f1, f2, f3] # list of element expressions for vector values function F = F(X [,P])
  X = [x1, x2, x3]
  P = [p1, p2, p3]
  print(F)

  J = symJacobian(F, X)
  print(J)

  H = symHessianCube(F, X)

  SF = create_def_string('myFunc', F, X)
  print(SF)
  exec(SF)

  SJ = create_def_string('myGrad', J, X)
  print(SJ)
  exec(SJ)

  SH = create_def_string('myHess', H, X)
  print(SH)
  exec(SH)

  SFxp = create_def_string('myFuncXP', [f1 + p1, f2 + p2, f3 + p3], X, P = P)
  SJxp = create_def_string('myGradXP', symJacobian([f1 + p1, f2 + p2, f3 + p3], X), X, P = P)

  ntests = args.ncheck
  print(f'testing {ntests}x random data numeric equivalence (sympy.evalf vs. generated numpy code)...')
  for _ in range(ntests):
    xcheck = np.random.randn(3)
    H_gen = myHess(xcheck)
    xsubs = {'x1' : xcheck[0], 'x2' : xcheck[1], 'x3' : xcheck[2]}
    H_sym = np.array(evalf_recursive(H, xsubs))
    assert np.allclose(H_gen, H_sym, atol = args.atol, rtol = args.rtol)

  if not args.no_genfile:
    assert len(args.genfile) > 0

    ok = write_python_file([SF, SJ, SH, SFxp, SJxp], 
                           '{}.py'.format(args.genfile), 
                           X = X, 
                           P = P,
                           autoformat = not args.no_black)
    assert ok == 0

    exec('import {} as gdc'.format(args.genfile))

    print(f'testing {ntests}x random data numeric equivalence (imported gen. sourcefile)...')
    for _ in range(ntests):
      xcheck = np.random.randn(3)
      assert np.allclose(myFunc(xcheck), gdc.myFunc(xcheck), atol = args.atol, rtol = args.rtol)
      assert np.allclose(myGrad(xcheck), gdc.myGrad(xcheck), atol = args.atol, rtol = args.rtol)
      assert np.allclose(myHess(xcheck), gdc.myHess(xcheck), atol = args.atol, rtol = args.rtol)

  print('done.')
