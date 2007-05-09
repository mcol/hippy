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
import numpy
from numpy import array, dot, zeros, linalg
from mps import Mps

def stepsize(v, dv):
    '''stepsize(v, dv):
    Compute the feasible stepsize from v along the direction dv.'''
    alpha = 1.0
    for i in range(len(v)):
        if dv[i] < 0:
            ratio = - v[i] / dv[i]
            if ratio < alpha:
                alpha = ratio.item()
    return alpha

class hippy:

    def __init__(self, file):
        '''__init()__:
        Constructor.'''
        self.sigma = 0.1
        self.optol = 1e-8
        self.iter  = 0
        self.maxiters = 20
        self.mpsfile = file
        self.status = None

    def newton(self, mu):
        '''newton(mu):
        Build the Newton system and compute the search direction.'''

        (dx, dy, ds) = self.normaleqns(self.x, self.y, self.s, mu)
        alphap = 0.9995 * stepsize(self.x, dx)
        alphad = 0.9995 * stepsize(self.s, ds)
        return (dx, dy, ds, alphap, alphad)

    def normaleqns(self, x, y, s, mu):
        '''normaleqns(x, y, s, mu):
        Find the search direction by solving the normal equations system.'''

        S = numpy.diagflat(s)
        X = numpy.diagflat(x)
        D = X * S.I
        AD = self.A * D
        M = AD * self.A.T

        e = numpy.asmatrix(numpy.ones(self.n)).T
        r = -s + numpy.multiply(mu, X.I * e)
        rhs = AD * (self.xic - r) + self.xib
        dy = linalg.solve(M, rhs)
        dx = D * (self.A.T * dy - self.xic + r)
        ds = r - S * dx / x
        return (dx, dy, ds)

    def read(self):
        '''read():
        Read the problem data.'''
        mpsdata = Mps(self.mpsfile)
        mpsdata.readMps()
        self.A, self.b, self.c = mpsdata.getdata()
        self.A = self.A.todense()
        self.n = len(self.c)

    def init(self):
        '''init():
        Provide the initial iterate.'''
        A = self.A

        # Mehrotra's way (following comments in OOPS)
        # AA^Tv = b    x = A^Tv
        # AA^Ty = Ac   s = c - A^Ty
        M = A * A.T
        v = linalg.solve(M, self.b)
        x = A.T * v
        y = linalg.solve(M, A * self.c)
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
        self.obj = self.c.T * self.x

    def xi(self):
        '''xi():
        Compute the value of mu, xib and xic.'''
        self.mu  = (self.x.T * self.s / self.n).item()
        self.xib = self.b - self.A * self.x
        self.xic = self.c - self.A.T * self.y - self.s

    def reportiter(self, alphap, alphad):
        '''reportiter(alphap, alphad):
        Print a line with some information on the iteration.'''
        erb = linalg.norm(self.xib)
        erc = linalg.norm(self.xic)
        print "%3d %10.3e %10.3e %10.3e %10.3e %10.3e" % \
              (self.iter, alphap, alphad, erb, erc, self.mu)

    def info(self):
        '''info():
        Report statistics on the solution.'''
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
        self.read()
        self.init()
        self.xi()

        print "Iter   alphap     alphad      xib        xic         mu"
        while self.mu > self.optol and self.iter < self.maxiters:
            self.iter += 1
            muhat = min(self.mu*self.mu, self.sigma*self.mu)
            (dx, dy, ds, alphap, alphad) = self.newton(muhat)

            self.x += alphap * dx
            self.y += alphad * dy
            self.s += alphad * ds
            self.xi()
            self.reportiter(alphap, alphad)
            obj = self.c.T * self.x
            if obj > 2 * self.obj:
                self.status = 'interrupted'
                break
            else:
                self.obj = obj

        if self.iter >= self.maxiters:
            self.status = 'maxiters'
        else:
            self.status = 'optimal'
        self.info()

def main(argv = None):

    # change argv if we are not running from the interactive prompt
    if argv is None:
        argv = sys.argv[1:]

    mpsfile = argv[0]
    problem = hippy(mpsfile)
    problem.solve()

if __name__ == "__main__":
    sys.exit(main())
