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
from numpy import array, zeros, linalg

def stepsize(v, dv):
    '''stepsize(v, dv):
    Compute the feasible stepsize from v along the direction dv.'''
    alpha = 1.0
    for i in range(len(v)):
        if dv[i] < 0:
            ratio = - v[i] / dv[i]
            if ratio < alpha:
                alpha = ratio
    return alpha

class hippy:

    def __init__(self):
        self.sigma = 0.1
        self.optol = 1e-8
        self.iter  = 0
        self.maxiters = 20

    def newton(self, mu):
        '''newton(self, mu):
        Build the Newton system and compute the search direction.'''

        return self.normaleqns(self.x, self.y, self.s, mu)

    def normaleqns(self, x, y, s, mu):
        '''normaleqns(self, x, y, s, mu):
        Find the search direction by solving the normal equations system.'''

        D2 = numpy.diag(x/s)
        AD2 = numpy.dot(self.A, D2)
        M = numpy.dot(AD2, self.At)

        r = -s + mu/x
        rhs = numpy.dot(AD2, self.xic - r) + self.xib
        dy = linalg.solve(M, rhs)
        dx = numpy.dot(D2, numpy.dot(self.At, dy) - self.xic + r)
        ds = r - s * dx / x
        return (dx, dy, ds)

    def read(self):
        self.A = array([[0, 1, 1]])
        self.b = array([2])
        self.c = array([1, 8, 0])
        self.n = len(self.c)
        self.At = self.A.transpose()

    def init(self):
        '''init():
        Provide the initial iterate.'''
        #    x = array([8, 1.95, 0.05])
        #    y = array([-0.1])
        #    s = array([1, 8.1, 0.1])
        A, At = self.A, self.At

        # Mehrotra's way (following comments in OOPS)
        # AA^Tv = b    x = A^Tv
        # AA^Ty = Ac   s = c - A^Ty
        M = numpy.dot(A, At)
        v = linalg.solve(M, self.b)
        x = numpy.dot(At, v)
        y = linalg.solve(M, numpy.dot(A, self.c))
        s = self.c - numpy.dot(At, y)

        # shift the point
        # dp = -1.5 * min { x_i },  dd = -1.5 * min { s_i }
        # xs = (x + dp)^T (s + dd) / x^Ts
        # dp = dp + 0.5 * xs / e^Tx, dd = dd + 0.5 * xs / e^Ts
        # x = x + dp,  s = s + dd
        dp = -1.5 * min(x)
        dd = -1.5 * min(s)
        xs = numpy.dot(x + dp, s + dd)

        self.x = x + dp + 0.5 * xs / sum(x)
        self.y = y
        self.s = s + dd + 0.5 * xs / sum(s)

    def xi(self):
        self.mu = numpy.inner(self.x, self.s) / self.n
        self.xib = self.b - numpy.dot(self.A, self.x)
        self.xic = self.c - numpy.dot(self.At, self.y) - self.s

    def info(self):
        print
        print "Problem solved in %d iterations." % self.iter

    def solve(self):
        self.read()
        self.init()

        A, b, c = self.A, self.b, self.c
        self.xi()

        print "Iter   alphap     alphad      xib        xic         mu"
        while self.mu > self.optol and self.iter < self.maxiters:
            self.iter += 1
            muhat = min(self.mu*self.mu, self.sigma*self.mu)
            (dx, dy, ds) = self.newton(muhat)
            alphap = 0.9995 * stepsize(self.x, dx)
            alphad = 0.9995 * stepsize(self.s, ds)

            self.x += alphap * dx
            self.y += alphad * dy
            self.s += alphad * ds
            self.xi()
            erb = linalg.norm(self.xib)
            erc = linalg.norm(self.xic)
            print "%3d %10.3e %10.3e %10.3e %10.3e %10.3e" % \
                  (self.iter, alphap, alphad, erb, erc, self.mu)

        self.info()

def main(argv = None):

    # change argv if we are not running from the interactive prompt
    if argv is None:
        argv = sys.argv[1:]

    problem = hippy()
    problem.solve()

if __name__ == "__main__":
    sys.exit(main())
