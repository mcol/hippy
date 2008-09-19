#!/usr/bin/python
#
# scale.py
#
# Routines to improve the scaling of a problem.
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

from numpy import exp, log, zeros

class Scale:

    def __init__(self, A, b, c, u):
        self.initscaling(A)
        self.computescaling(A)
        self.applyscaling(A, b, c, u)
        print "Scaling done."

    def scalefactor(self, A):
        # scalefactor = Sum (log |Aij|)^2
        factor = 0.0

        # loop over the nonzero entries
        for i in range(A.getnnz()):
            value   = log(abs(A.getdata(i)))
            factor += value**2

        return factor

    def initscaling(self, A):
        rows, cols = A.shape
        rowlogs, rownnzs = zeros(rows), zeros(rows)
        collogs, colnnzs = zeros(cols), zeros(cols)

        factor = 0.0
        for i in range(A.getnnz()):
            row, col = A.rowcol(i)
            value = log(abs(A.getdata(i)))
            rowlogs[row] += value
            collogs[col] += value
            rownnzs[row] += 1
            colnnzs[col] += 1
            factor += value**2

        self.rowlogs, self.rownnzs = rowlogs, rownnzs
        self.collogs, self.colnnzs = collogs, colnnzs

        return factor

    def __updatesk(self, residual, count):
        sk = 0.0
        for i in range(len(residual)):
            sk += residual[i]**2 / count[i]
        return sk

    def __updatefactors(self, factor, oldfac, residual, count, ee, qq):
        eeqq = ee / qq
        for i in range(len(factor)):
            factor[i] *= 1.0 + eeqq
            factor[i] += residual[i] / (qq * count[i]) - oldfac[i] * eeqq

    def __updateresidual(self, res1, res2, count, A, ek, qk):

        len1 = len(res1)
        rows, cols = A.shape
        if rows == len1: swap = False
        else:            swap = True

        for i in range(len1):
            res1[i] *= ek

        for i in range(A.getnnz()):
            row, col = A.rowcol(i)
            if swap: row, col = col, row
            res1[row] += res2[col] / count[col]

        for i in range(len1):
            res1[i] *= -1.0 / qk

    def computescaling(self, A):
        rows, cols = A.shape
        nnzs  = A.getnnz()
        iters = 0
        toler = 0.01 * nnzs
        rowfactor1, rowfactor2 = zeros(rows), zeros(cols)
        colfactor1, colfactor2 = zeros(cols), zeros(cols)
        rownnzs, colnnzs = self.rownnzs, self.colnnzs

        for i in range(rows):
            rowfactor1[i] = self.rowlogs[i] / rownnzs[i]
            rowfactor2[i] = rowfactor1[i]

        # initial residual
        rowres = zeros(rows)
        colres = self.collogs.copy()
        for i in range(nnzs):
            row, col = A.rowcol(i)
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
        for row in range(rows):
            rowfactor1[row] = max(exp(-rowfactor1[row]), 1.0e-8)
        for col in range(cols):
            colfactor1[col] = max(exp(-colfactor1[col]), 1.0e-8)

        self.rowfactor = rowfactor1
        self.colfactor = colfactor1

    def applyscaling(self, A, b, c, u):
        # scale the matrix
        for i in range(A.getnnz()):
            row, col = A.rowcol(i)
            A.data[i] *= self.rowfactor[row] * self.colfactor[col]

        # scale the right-hand side
        for i in range(len(b)):
            b[i] *= self.rowfactor[i]

        # scale the objective
        for i in range(len(c)):
            c[i] *= self.colfactor[i]

        # scale the upper bounds
        for i in range(len(u)):
            u[i] /= self.colfactor[u.idx[i]]
