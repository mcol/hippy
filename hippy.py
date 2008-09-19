#!/usr/bin/python
#
# hippy.py
#
# Hippy Interior Point methods in Python.
#
# Copyright (c) 2007, 2008 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

import sys
from numpy import diag, dot
from scipy import linsolve
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

    def __init__(self, A, x, s):
        '''Constructor.'''
        self.A = A
        self.x = x
        self.d = s / x
        self.D = 1.0 / (self.d)
        self.M = self.A * diag(self.D) * self.A.T

    def solve(self, xib, xic, xim):
        '''Solve the normal equations system for the given right-hand side.'''
        t = xim / self.x
        r = xic - t
        rhs = self.A * (self.D * r) + xib
        dy = linsolve.spsolve(self.M, rhs)
        dx = self.D * (self.A.T * dy - r)
        ds = t - dx * self.d
        return dx, dy, ds

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
        NE = normalequations(self.A, self.x, self.s)

        (dx, dy, ds) = self.newton(NE, self.x, self.s)
        (dx, dy, ds) = self.mehrotra(NE, dx, dy, ds)
        self.alphap = 0.9995 * stepsize(self.x, dx)
        self.alphad = 0.9995 * stepsize(self.s, ds)

        self.makestep(dx, dy, ds)
        self.xi()

    def newton(self, NE, x, s, mu = 0.0):
        '''Compute the affine-scaling direction.'''
        v = -x * s + mu
        return NE.solve(self.xib, self.xic, v)

    def sigmamu(self, dx, ds):
        '''Compute the target barrier parameter for Mehrotra's corrector.'''
        alphap = stepsize(self.x, dx)
        alphad = stepsize(self.s, ds)
        x = self.x + alphap * dx
        s = self.s + alphad * ds
        g = self.average(x, s)
        return g**3 / self.mu**2

    def mehrotra(self, NE, dx, dy, ds):
        '''Compute Mehrotra's corrector.'''
        v = -dx * ds + self.sigmamu(dx, ds)
        zb = [0.0] * len(self.xib)
        zc = [0.0] * len(self.xic)
        mx, my, ms = NE.solve(zb, zc, v)
        return dx + mx, dy + my, ds + ms

    def read(self):
        '''Read the MPS file.'''
        mpsdata = Mps(self.mpsfile)

        try:
            mpsdata.readMps()
        except (IOError, IndexError):
            return sys.exit(1)

        self.A, self.b, self.c, bndVal, bndIdx = mpsdata.getdata()
        self.n = len(self.c)
        self.u = Sparsevector(self.n, bndVal, bndIdx)

    def scale(self):
        '''Scale the problem data.'''
        Scale(self.A, self.b, self.c, self.u)

    def init(self):
        '''Compute the initial iterate according to Mehrotra's heuristic.'''
        self.iter = 0
        A = self.A
        u = self.u.todense()

        # compute an initial point
        # D = diag(1.0 if no upper bound; 0.5 if upper bound)
        # ADA^Tv = b - ADu   x = DA^Tv + Du
        # ADA^Ty = ADc       s = Dc - DA^Ty
        d = [1.0] * self.n
        for i in self.u.idx:
            d[i] = 0.5
        F = A * diag(d)
        M = F * A.T
        v = linsolve.spsolve(M, self.b - F * u)
        x = F.T * v + d * u
        y = linsolve.spsolve(M, F * self.c)
        s = d * self.c - F.T * y

        # shift the point
        # dp = max(-1.5 * min { x_i }, 0.1)
        # dd = max(-1.5 * min { s_i }, 0.1
        # xs = (x + dp)^T (s + dd) / x^Ts
        # dp = dp + 0.5 * xs / sum(x + dp)
        # dd = dd + 0.5 * xs / sum(s + dd)
        # x = x + dp,  s = s + dd
        dp = max(-1.5 * min(x), 0.1)
        dd = max(-1.5 * min(s), 0.1)
        xs = (x + dp).T * (s + dd)
        dp += 0.5 * xs / sum(x + dp)
        dd += 0.5 * xs / sum(s + dd)

        self.x = x + dp
        self.y = y
        self.s = s + dd
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

    def average(self, x, s):
        '''Compute the average complementarity gap.'''
        gap = dot(x, s)
        return gap / self.n

    def xi(self):
        '''Compute duality gap, complementarity gap and infeasibilities.'''
        self.pobj = dot(self.c, self.x)
        self.dobj = dot(self.b, self.y)
        self.gap = self.pobj - self.dobj
        self.mu  = self.average(self.x, self.s)
        self.xib = self.b - self.A * self.x
        self.xic = self.c - self.A.T * self.y - self.s
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

    def makestep(self, dx, dy, ds):
        '''Move along the given direction.'''
        (alphap, alphad) = (self.alphap, self.alphad)
        self.x += alphap * dx
        self.y += alphad * dy
        self.s += alphad * ds

    def info(self):
        '''Report statistics on the solution.'''
        print
        if self.status is 'optimal':
            print "Problem solved in %d iterations." % self.iter
            print "Objective: ", self.pobj
        elif self.status is 'infeasible':
            print "The problem is infeasible."
        elif self.status is 'maxiters':
            print "Maximum number of iterations reached."
        elif self.status is 'interrupted':
            print "The solution of the problem failed."
        else:
            print "Unknown status."

    def solve(self):
        '''Solve the problem, from the MPS file to the optimal solution.'''
        self.scale()
        self.init()
        self.solver()
        self.info()

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
                self.status = 'interrupted'
                return

        if self.iter >= self.maxiters:
            self.status = 'maxiters'
        else:
            self.status = 'optimal'

    def getsolution(self):
        '''Retrieve the solution vectors.'''
        return (self.x, self.y, self.s)

    def printiter(self):
        '''Print the primal-dual iterate.'''
        print "x:", self.x
        print "y:", self.y
        print "s:", self.s

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
