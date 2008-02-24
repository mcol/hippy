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
from numpy import diag, dot, linalg, zeros_like
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
                alpha = ratio
    return alpha

class normalequations:

    def __init__(self, A, x, s):
        '''Constructor.'''
        self.A = A
        self.x = x
        self.d = x / s
        self.M = self.A * diag(self.d) * self.A.T

    def setrhs(self, xib, xic, xim):
        '''Set the right-hand side for which the system must be solved.'''
        self.xib = xib
        self.xic = xic
        self.xim = xim

    def solve(self):
        '''Solve the normal equations system.'''
        r = self.xim / self.x
        t = self.xic - r
        rhs = self.A * (t * self.d) + self.xib
        dy = linsolve.spsolve(self.M, rhs)
        dx = self.d * (self.A.T * dy - t)
        ds = r - dx / self.d
        return dx, dy, ds

class hippy:

    def __init__(self, file):
        '''Constructor.'''
        self.optol = 1e-8
        self.iter  = 0
        self.maxiters = 20
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
        NE.setrhs(self.xib, self.xic, v)
        return NE.solve()

    def sigmamu(self, dx, ds):
        '''Compute the target barrier parameter for Mehrotra's corrector.'''
        alphap = stepsize(self.x, dx)
        alphad = stepsize(self.s, ds)
        x = self.x + alphap * dx
        s = self.s + alphad * ds
        g = dot(x, s) / (self.n * self.mu)
        return g**3 * self.mu

    def mehrotra(self, NE, dx, dy, ds):
        '''Compute Mehrotra's corrector.'''
        v = -dx * ds + self.sigmamu(dx, ds)
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
        '''Compute the initial iterate according to Mehrotra's heuristic.'''
        self.iter = 0
        A = self.A

        # Mehrotra's way (following comments in OOPS)
        # AA^Tv = b    x = A^Tv
        # AA^Ty = Ac   s = c - A^Ty
        M = A * A.T
        v = linsolve.spsolve(M, self.b)
        x = A.T * v
        y = linsolve.spsolve(M, A * self.c)
        s = self.c - A.T * y

        # shift the point
        # dp = max(-1.5 * min { x_i }, 0.1)
        # dd = max(-1.5 * min { s_i }, 0.1
        # xs = (x + dp)^T (s + dd) / x^Ts
        # dp = dp + 0.5 * xs / e^Tx, dd = dd + 0.5 * xs / e^Ts
        # x = x + dp,  s = s + dd
        dp = max(-1.5 * min(x), 0.1)
        dd = max(-1.5 * min(s), 0.1)
        xs = (x + dp).T * (s + dd)

        self.x = x + dp + 0.5 * xs / max(sum(x), 0.1)
        self.y = y
        self.s = s + dd + 0.5 * xs / max(sum(s), 0.1)
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

    def xi(self):
        '''Compute duality gap, complementarity gap and infeasibilities.'''
        self.mu  = dot(self.x, self.s) / self.n
        self.gap = dot(self.c, self.x) - dot(self.b, self.y)
        self.xib = self.b - self.A * self.x
        self.xic = self.c - self.A.T * self.y - self.s
        self.erb = linalg.norm(self.xib)
        self.erc = linalg.norm(self.xic)

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
            print "Objective: ", dot(self.c, self.x)
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

            if self.gap > 2 * oldgap and self.iter > 3:
                self.status = 'interrupted'
                return

        if self.iter >= self.maxiters:
            self.status = 'maxiters'
        else:
            self.status = 'optimal'

    def getsolution(self):
        '''Retrieve the solution vectors.'''
        return (self.x, self.y, self.s)

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
