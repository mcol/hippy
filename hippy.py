#!/usr/bin/python
#
# hippy.py
#
# Hippy Interior Point methods in Python.
#
# Copyright (c) 2007, 2008, 2012 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

import sys
from numpy import append, dot
from scipy.sparse import csc_matrix, spdiags
from scipy.sparse.linalg import factorized, use_solver
from mps import Mps
from scale import Scale
from sparsevector import Sparsevector

def stepsize(v, dv):
    '''Compute the feasible stepsize from v along the direction dv.'''
    ratios = -v / dv
    try:
        alpha = min(1.0, min(ratios[ratios > 0.0]))
    except ValueError:
        alpha = 1.0
    return alpha

class normalequations:

    def __init__(self, A, x, s, z, w):
        '''Constructor.'''
        self.A = A
        self.x = x
        self.z = z
        self.d = s / x
        self.f = w / z
        self.D = 1.0 / (self.d + self.f.todense())
        n = len(self.D)
        M = self.A * spdiags(self.D, 0, n, n) * self.A.T
        try:
            self.factsolve = factorized(M)
        except RuntimeError:
            # UMFPACK_ERROR_out_of_memory
            print "Switching type of LU factorization"
            use_solver(useUmfpack = False)
            self.factsolve = factorized(M)

    def solve(self, xib, xic, xim, xiu, xiz):
        '''Solve the normal equations system for the given right-hand side.'''
        t = xim / self.x
        v = xiz / self.z - xiu * self.f
        r = xic - t + v.todense()
        rhs = self.A * (self.D * r) + xib
        dy = self.factsolve(rhs)
        dx = self.D * (self.A.T * dy - r)
        ds = t - dx * self.d
        xx = dx[xiu.idx]
        dz = xiu - xx
        dw = v + self.f * xx
        return dx, dy, ds, dz, dw

class hippy:

    def __init__(self, file):
        '''Constructor.'''
        self.optol = 1e-8
        self.iter  = 0
        self.maxiters = 30
        self.mpsfile = file
        self.read()
        self.status = None

    def direction(self):
        '''Build the Newton system and compute the search direction.'''
        NE = normalequations(self.A, self.x, self.s, self.z, self.w)

        dx, dy, ds, dz, dw = self.newton(NE, self.x, self.s, self.z, self.w)
        dx, dy, ds, dz, dw = self.mehrotra(NE, dx, dy, ds, dz, dw)
        self.alphap = 0.9995 * min(stepsize(self.x, dx),
                                   stepsize(self.z.data(), dz.data()))
        self.alphad = 0.9995 * min(stepsize(self.s, ds),
                                   stepsize(self.w.data(), dw.data()))

        self.makestep(dx, dy, ds, dz, dw)
        self.xi()

    def newton(self, NE, x, s, z, w, mu = 0.0):
        '''Compute the affine-scaling direction.'''
        v = -x * s + mu
        t = -z * w + mu
        return NE.solve(self.xib, self.xic, v, self.xiu, t)

    def sigmamu(self, dx, ds, dz, dw):
        '''Compute the target barrier parameter for Mehrotra's corrector.'''
        alphap = min(stepsize(self.x, dx), stepsize(self.z.data(), dz.data()))
        alphad = min(stepsize(self.s, ds), stepsize(self.w.data(), dw.data()))
        x = self.x + alphap * dx
        s = self.s + alphad * ds
        z = self.z + alphap * dz
        w = self.w + alphad * dw
        g = self.average(x, s, z, w)
        return g**3 / self.mu**2

    def mehrotra(self, NE, dx, dy, ds, dz, dw):
        '''Compute Mehrotra's corrector.'''
        m = self.sigmamu(dx, ds, dz, dw)
        v = -dx * ds + m
        t = -dz * dw + m
        zb = [0.0] * self.m
        zc = [0.0] * self.n
        zu = Sparsevector(self.n, [0.0] * len(t), t.idx)
        mx, my, ms, mz, mw = NE.solve(zb, zc, v, zu, t)
        return dx + mx, dy + my, ds + ms, dz + mz, dw + mw

    def read(self):
        '''Read the MPS file.'''
        try:
            print "Reading file", self.mpsfile + "."
            mpsdata = Mps(self.mpsfile)
        except IOError as e:
            print e
            return sys.exit(1)
        except (IndexError, NotImplementedError, ValueError) as e:
            print e
            print "Parsing of the MPS file interrupted."
            return sys.exit(1)

        self.A, self.b, self.c, bounds = mpsdata.getdata()
        self.rowNames, self.rowTypes = mpsdata.rowNames, mpsdata.rowTypes
        self.m = len(self.b)
        self.n = len(self.c)
        uppVal, uppIdx = bounds[0:2]
        lowVal, lowIdx = bounds[2:4]
        self.u = Sparsevector(self.n, uppVal, uppIdx)
        self.l = Sparsevector(self.n, lowVal, lowIdx)

    def preprocess(self):
        '''Preprocess the problem before solving it.'''
        self.shiftbounds()
        self.addslacks()
        self.scale()

    def postprocess(self):
        '''Postprocess the problem after having solved it.'''
        self.unscale()
        self.removeslacks()
        self.removeshift()

    def shiftbounds(self):
        '''Shift the lower bounded variables.'''
        l = self.l.todense()

        # shift the objective
        self.objshift = dot(self.c, l)

        # shift the right-hand side
        self.b -= self.A * l

        # shift the upper bounds
        for i in range(len(self.u)):
            self.u[i] -= l[self.u.idx[i]]

    def removeshift(self):
        '''Remove the shift from the lower bounded variables.'''
        self.x    += self.l.todense()
        self.pobj += self.objshift
        self.dobj += self.objshift

    def scale(self):
        '''Scale the problem data.'''
        self.scaling = Scale(self.A, self.b, self.c, self.u)

    def unscale(self):
        '''Restore the original scaling of the data and the solution.'''
        try:
            rowfactor = self.scaling.rowfactor
            colfactor = self.scaling.colfactor
        except AttributeError:
            return

        self.c /= colfactor
        self.x *= colfactor
        self.s /= colfactor
        self.b /= rowfactor
        self.y *= rowfactor
        for i in range(len(self.u)):
            factor = colfactor[self.u.idx[i]]
            self.u[i] *= factor
            self.z[i] /= factor
            self.w[i] /= factor

    def addslacks(self):
        '''Add slack variables to inequality constraints.'''
        self.nslacks, nnnz = 0, self.A.nnz
        data, rows, ptrs = [], [], []
        for key in self.rowTypes.iterkeys():
            if self.rowTypes[key] is 'E':
                continue
            if self.rowTypes[key] is 'L':
                data.append(1.0)
            elif self.rowTypes[key] is 'G':
                data.append(-1.0)

            rows.append(self.rowNames[key])
            ptrs.append(nnnz)
            self.nslacks += 1
            nnnz += 1

        # add the last element
        ptrs.append(nnnz)

        if self.nslacks == 0:
            return

        # extend the matrix with the slacks
        data = append(self.A.data, data)
        rows = append(self.A.indices, rows)
        ptrs = append(self.A.indptr[:-1], ptrs)
        self.A = csc_matrix((data, rows, ptrs))
        self.c = append(self.c, [0] * self.nslacks)
        self.u.dim = self.l.dim = self.n = len(self.c)

        # report the number of slacks added
        print "Added %d slacks variables." % self.nslacks

    def removeslacks(self):
       '''Remove the slack variables.'''
       if self.nslacks == 0:
           return
       self.n -= self.nslacks
       self.c = self.c[:self.n]
       self.x = self.x[:self.n]
       self.s = self.s[:self.n]
       self.u.dim = self.l.dim = self.n

    def init(self):
        '''Compute the initial iterate according to Mehrotra's heuristic.'''
        self.iter = 0
        A = self.A
        u = self.u.todense()

        # compute an initial point
        # D = diag(1.0 if no upper bound; 0.5 if upper bound)
        # ADA^Tv = b - Adu   x = DA^Tv + Du   z = u - x
        # ADA^Ty = ADc       s = Dc - DA^Ty   w = -s
        d = [1.0] * self.n
        for i in self.u.idx:
            d[i] = 0.5
        F = A * spdiags(d, 0, self.n, self.n)
        factsolve = factorized(F * A.T)
        v = factsolve(self.b - F * u)
        x = F.T * v + d * u
        y = factsolve(F * self.c)
        s = d * self.c - F.T * y
        z = self.u - x[self.u.idx]
        w = Sparsevector(self.n, -s[self.u.idx], self.u.idx)

        # compute the shift
        # dp = max{-1.5 * min {x_i}, -1.5 * min {z_i}, 0.1}
        # dd = max{-1.5 * min {s_i}, -1.5 * min {w_i}, 0.1}
        dp = max(-1.5 * min(x), 0.1)
        dd = max(-1.5 * min(s), 0.1)
        if len(self.u) > 0:
            dp, dd = max(dp, -1.5 * min(z.val)), max(dd, -1.5 * min(w.val))

        # update the shift
        # this is a simplified version which doesn't take into account the
        # contribution of z and w. note that the shift is not uniform for
        # problems without upper bounds.
        xs = (x + dp) * (s + dd)
        if len(self.u) > 0:
            xs = sum(xs)
        dp += 0.5 * xs / (sum(x + dp) + sum(z + dp))
        dd += 0.5 * xs / (sum(s + dd) + sum(w + dd))

        # shift the point
        self.x = x + dp
        self.y = y
        self.s = s + dd
        if len(self.u) == 0:
            dp, dd = dp[z.idx], dd[w.idx]
        self.z = z + dp
        self.w = w + dd

        self.xi()

    def initpoint(self, point):
        '''Provide the initial iterate.'''
        self.iter = 0
        try:
            (self.x, self.y, self.s) = point
            self.xi()
        except ValueError:
            print "The vectors given to initpoint() have the wrong dimensions."
            raise

    def average(self, x, s, z, w):
        '''Compute the average complementarity gap.'''
        gap = dot(x, s) + dot(z.data(), w.data())
        return gap / (self.n + len(z))

    def xi(self):
        '''Compute duality gap, complementarity gap and infeasibilities.'''
        self.pobj = dot(self.c, self.x)
        self.dobj = dot(self.b, self.y) - dot(self.u.data(), self.w.data())
        self.gap = self.pobj - self.dobj
        self.mu  = self.average(self.x, self.s, self.z, self.w)
        self.xib = self.b - self.A * self.x
        self.xic = self.c - self.A.T * self.y - self.s + self.w.todense()
        self.xiu = self.u - self.z - self.x[self.u.idx]
        self.erb = max(abs(self.xib))
        self.erc = max(abs(self.xic))

    def reportiter(self):
        '''Print a line with some information on the iteration.'''
        if (self.iter == 1):
            print
            print "Iter  alphap     alphad       xib\t xic\t    mu\t       gap"

        erb, erc = self.erb, self.erc
        alphap, alphad = self.alphap, self.alphad
        print "%3d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e" % \
              (self.iter, alphap, alphad, erb, erc, self.mu, self.gap)

    def makestep(self, dx, dy, ds, dz, dw):
        '''Move along the given direction.'''
        (alphap, alphad) = (self.alphap, self.alphad)
        self.x += alphap * dx
        self.y += alphad * dy
        self.s += alphad * ds
        self.z += alphap * dz
        self.w += alphad * dw

    def info(self):
        '''Report statistics on the solution.'''
        print
        if self.status is 'optimal':
            print "Problem solved in %d iterations." % self.iter
            print "Objective: ", self.pobj
        elif self.status is 'suboptimal':
            print "Suboptimal solution found in %d iterations." % self.iter
            print "Objective: ", self.pobj
        elif self.status is 'infeasible':
            print "The problem is infeasible."
        elif self.status is 'maxiters':
            print "Maximum number of iterations reached."
        elif self.status is 'failed':
            print "The solution of the problem failed."
        else:
            print "Unknown status."

    def solve(self):
        '''Solve the problem, from the MPS file to the optimal solution.'''
        self.preprocess()
        self.init()
        self.solver()
        self.postprocess()
        self.info()
        return self.status

    def solver(self):
        '''Call the solver.'''
        while (self.mu > self.optol or abs(self.gap) > self.optol or \
               self.erb > self.optol or self.erc > self.optol) and \
               self.iter < self.maxiters:

            oldgap = self.gap
            self.iter += 1
            self.direction()
            self.reportiter()

            if abs(self.gap) > 2 * abs(oldgap) and self.iter > 3:
                if self.mu < self.optol:
                    self.status = 'suboptimal'
                else:
                    self.status = 'failed'
                return

        if self.iter >= self.maxiters:
            self.status = 'maxiters'
        else:
            self.status = 'optimal'

    def getsolution(self):
        '''Retrieve the solution vectors.'''
        return (self.x, self.y, self.s, self.z, self.w)

    def printiter(self):
        '''Print the primal-dual iterate.'''
        print "x:", self.x
        print "y:", self.y
        print "s:", self.s
        print "z:", self.z
        print "w:", self.w

def usage():
    '''Report the usage message.'''
    print "Usage: hippy.py <problem.mps>"

def main(argv = None):

    # change argv if we are not running from the interactive prompt
    if argv is None:
        argv = sys.argv[1:]

    if argv == []:
        usage()
        return 1

    mpsfile = argv[0]
    problem = hippy(mpsfile)
    problem.solve()

if __name__ == "__main__":
    sys.exit(main())
