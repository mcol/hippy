#!/usr/bin/python
#
# scale.py
#
# Routines to improve the scaling of a problem.
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

from numpy import exp, log, maximum, zeros

class Scale:

    def __init__(self, A, b, c, u):
        self.initscaling(A)
        self.computescaling(A)
        self.applyscaling(A, b, c, u)
        print "Scaling done."

    def scalefactor(self, A):
        # scalefactor = Sum (log |Aij|)^2
        values = log(abs(A.data))**2
        return sum(values)

    def initscaling(self, A):
        rows, cols = A.shape
        rowlogs, rownnzs = zeros(rows), zeros(rows)
        collogs, colnnzs = zeros(cols), zeros(cols)

        factor = 0.0
        rowidx, colidx = A.nonzero()
        values = log(abs(A.data))
        for i in range(A.getnnz()):
            row, col = rowidx[i], colidx[i]
            value = values[i]
            rowlogs[row] += value
            collogs[col] += value
            rownnzs[row] += 1
            colnnzs[col] += 1
            factor += value**2

        self.rowlogs, self.rownnzs = rowlogs, rownnzs
        self.collogs, self.colnnzs = collogs, colnnzs

        return factor

    def __updatesk(self, residual, count):
        sk = sum(residual**2 / count)
        return sk

    def __updatefactors(self, factor, oldfac, residual, count, ee, qq):
        eeqq = ee / qq
        factor *= 1.0 + eeqq
        factor += residual / (qq * count) - oldfac * eeqq

    def __updateresidual(self, res1, res2, count, A, ek, qk):

        rowidx, colidx = A.nonzero()
        if A.shape[0] != len(res1): rowidx, colidx = colidx, rowidx

        res1 *= ek
        for i in range(A.getnnz()):
            row, col = rowidx[i], colidx[i]
            res1[row] += res2[col] / count[col]
        res1 *= -1.0 / qk

    def computescaling(self, A):
        rows, cols = A.shape
        nnzs  = A.getnnz()
        iters = 0
        toler = 0.01 * nnzs
        rowfactor1 = zeros(rows)
        colfactor1, colfactor2 = zeros(cols), zeros(cols)
        rownnzs, colnnzs = self.rownnzs, self.colnnzs

        rowfactor1 = self.rowlogs / rownnzs
        rowfactor2 = rowfactor1.copy()

        # initial residual
        rowres = zeros(rows)
        colres = self.collogs.copy()
        rowidx, colidx = A.nonzero()
        for i in range(nnzs):
            row, col = rowidx[i], colidx[i]
            colres[col] -= rowfactor1[row]

        sk1, sk = 0.0, self.__updatesk(colres, colnnzs)
        ek1, ek, qk1, qk = 0.0, 0.0, 0.0, 1.0

        while sk > toler:

            if iters % 2 == 0:
                factor, oldfac = rowfactor1, rowfactor2
                res1, res2, nnz1, nnz2 = rowres, colres, rownnzs, colnnzs
            else:
                factor, oldfac = colfactor1, colfactor2
                res1, res2, nnz1, nnz2 = colres, rowres, colnnzs, rownnzs

            if iters > 0:
                self.__updatefactors(factor, oldfac, res1,
                                     nnz1, ek * ek1, qk * qk1)
            self.__updateresidual(res1, res2, nnz2, A, ek, qk)

            sk1, sk = sk, self.__updatesk(res1, nnz1)
            ek1, ek = ek, qk * sk / sk1
            qk1, qk = qk, 1.0 - ek
            iters += 1

        # final update
        if iters > 0:
            if iters % 2 == 0:
                self.__updatefactors(rowfactor1, rowfactor2, rowres,
                                     rownnzs, ek * ek1, qk1)
            else:
                self.__updatefactors(colfactor1, colfactor2, colres,
                                     colnnzs, ek * ek1, qk1)

        # find the scaling factors
        self.rowfactor = maximum(exp(-rowfactor1), 1.0e-8)
        self.colfactor = maximum(exp(-colfactor1), 1.0e-8)

    def applyscaling(self, A, b, c, u):
        # scale the matrix
        rowidx, colidx = A.nonzero()
        A.data *= self.rowfactor[rowidx] * self.colfactor[colidx]

        # scale the right-hand side
        b *= self.rowfactor

        # scale the objective
        c *= self.colfactor

        # scale the upper bounds
        for i in range(len(u)):
            u[i] /= self.colfactor[u.idx[i]]
