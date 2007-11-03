#!/usr/bin/python
#
# hippy.py
#
# Hippy Interior Point methods in Python.
#
# Copyright (c) 2007 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

import sys
from numpy import array, asmatrix, diagflat, dot, linalg, multiply, ravel, \
     zeros_like
from scipy import linsolve
from mps import Mps
from scale import Scale

def stepsize(v, dv):
    '''Compute the feasible stepsize from v along the direction dv.'''
    alpha = 1.0
    for i in range(len(v)):
        if dv[i] < 0:
            ratio = - v[i] / dv[i]
            if ratio < alpha:
                alpha = ratio.item()
    return alpha

class normalequations:

    def __init__(self, A, X, S):
        '''Constructor.'''
        self.A = A
        self.X = X
        self.D = X * S.I
        self.M = self.A * self.D * self.A.T

    def setrhs(self, xib, xic, xim):
        '''Set the right-hand side for which the system must be solved.'''
        self.xib = xib
        self.xic = xic
        self.xim = xim

    def solve(self):
        '''Solve the normal equations system.'''
        r = self.X.I * self.xim
        t = self.xic - r
        rhs = self.A * (self.D * t) + self.xib
        dy = asmatrix(linsolve.spsolve(self.M, ravel(rhs))).T
        dx = self.D * (self.A.T * dy - t)
        ds = r - self.D.I * dx
        return dx, dy, ds

class hippy:

    def __init__(self, file):
        '''Constructor.'''
        self.optol = 1e-8
        self.iter  = 0
        self.maxiters = 20
        self.mpsfile = file
        self.status = None

    def direction(self):
        '''Build the Newton system and compute the search direction.'''
        X = diagflat(self.x)
        S = diagflat(self.s)
        NE = normalequations(self.A, X, S)

        (dx, dy, ds) = self.newton(NE, self.x, self.s)
        (dx, dy, ds) = self.mehrotra(NE, dx, dy, ds)
        alphap = 0.9995 * stepsize(self.x, dx)
        alphad = 0.9995 * stepsize(self.s, ds)

        self.makestep(dx, dy, ds, alphap, alphad)

    def newton(self, NE, x, s, mu = 0.0):
        '''Compute the affine-scaling direction.'''
        v = -multiply(x, s) + mu
        NE.setrhs(self.xib, self.xic, v)
        return NE.solve()

    def sigmamu(self, dx, ds):
        '''Compute the target barrier parameter for Mehrotra\'s corrector.'''
        alphap = stepsize(self.x, dx)
        alphad = stepsize(self.s, ds)
        x = self.x + alphap * dx
        s = self.s + alphad * ds
        g = (x.T * s).item() / (self.n * self.mu)
        return g**3 * self.mu

    def mehrotra(self, NE, dx, dy, ds):
        '''Compute Mehrotra\'s corrector.'''
        v = -multiply(dx, ds) + self.sigmamu(dx, ds)
        NE.setrhs(zeros_like(self.xib), zeros_like(self.xic), v)
        mx, my, ms = NE.solve()
        return dx + mx, dy + my, ds + ms

    def read(self):
        '''Read the MPS file.'''
        mpsdata = Mps(self.mpsfile)

        try:
            mpsdata.readMps()
        except (IOError, IndexError):
            return sys.exit(1)

        self.A, self.b, self.c = mpsdata.getdata()
        self.n = len(self.c)

    def scale(self):
        '''Scale the problem data.'''
        Scale(self.A, self.b, self.c)

    def init(self):
        '''Compute the initial iterate according to Mehrotra\'s heuristic.'''
        A = self.A

        # Mehrotra's way (following comments in OOPS)
        # AA^Tv = b    x = A^Tv
        # AA^Ty = Ac   s = c - A^Ty
        M = A * A.T
        v = asmatrix(linsolve.spsolve(M, ravel(self.b))).T
        x = A.T * v
        y = asmatrix(linsolve.spsolve(M, ravel(A * self.c))).T
        s = self.c - A.T * y

        # shift the point
        # dp = max(-1.5 * min { x_i }, 0)
        # dd = max(-1.5 * min { s_i }, 0)
        # xs = (x + dp)^T (s + dd) / x^Ts
        # dp = dp + 0.5 * xs / e^Tx, dd = dd + 0.5 * xs / e^Ts
        # x = x + dp,  s = s + dd
        dp = max(-1.5 * min(x).item(), 0)
        dd = max(-1.5 * min(s).item(), 0)
        xs = (x + dp).T * (s + dd)

        self.x = x + dp + 0.5 * xs / sum(x)
        self.y = y
        self.s = s + dd + 0.5 * xs / sum(s)
        self.gap = self.c.T * self.x - self.b.T * self.y
        self.xi()

    def xi(self):
        '''Compute the average complementarity gap and the infeasibilities.'''
        self.mu  = (self.x.T * self.s / self.n).item()
        self.xib = self.b - self.A * self.x
        self.xic = self.c - self.A.T * self.y - self.s

    def reportiter(self, alphap, alphad):
        '''Print a line with some information on the iteration.'''
        erb = linalg.norm(self.xib)
        erc = linalg.norm(self.xic)
        print "%3d %10.3e %10.3e %10.3e %10.3e %10.3e %10.3e" % \
              (self.iter, alphap, alphad, erb, erc, self.mu, self.gap)

    def makestep(self, dx, dy, ds, alphap, alphad):
        '''Move along the given direction and report the progress.'''
        self.x += alphap * dx
        self.y += alphad * dy
        self.s += alphad * ds
        self.xi()
        self.reportiter(alphap, alphad)

    def info(self):
        '''Report statistics on the solution.'''
        print
        if self.status is 'optimal':
            print "Problem solved in %d iterations." % self.iter
            print "Objective: ", (self.c.T * self.x).item()
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
        self.read()
        self.scale()
        self.init()

        print "Iter  alphap     alphad       xib\t xic\t    mu\t       gap"
        while self.mu > self.optol and self.iter < self.maxiters:

            self.iter += 1
            self.direction()

            gap = self.c.T * self.x - self.b.T * self.y
            if gap > 2 * self.gap and self.iter > 3:
                self.status = 'interrupted'
                return
            else:
                self.gap = gap

        if self.iter >= self.maxiters:
            self.status = 'maxiters'
        else:
            self.status = 'optimal'

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
    problem.info()

if __name__ == "__main__":
    sys.exit(main())
