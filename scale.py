#!/usr/bin/python
#
# scale.py
#
# Routines to improve the scaling of a problem.
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

from numpy import log, zeros

class Scale:

    def __init__(self, A, b, c):
        self.scalefactor(A)
        self.applyscaling(A, b, c)
        self.scalefactor(A)

    def scalefactor(self, A):
        # scalefactor = Sum (log |Aij|)^2
        factor = 0.0

        # loop over the nonzero entries
        for i in range(A.getnnz()):
            value   = log(abs(A.getdata(i)))
            factor += value**2

        print "Scaling factor:", factor

    def initscaling(self, A):
        rows, cols = A.shape
        rowlogs, rownnzs = zeros(rows), zeros(rows)
        collogs, colnnzs = zeros(cols), zeros(cols)

        for i in range(A.getnnz()):
            row, col = A.rowcol(i)
            value = log(abs(A.getdata(i)))
            rowlogs[row] += value
            collogs[col] += value
            rownnzs[row] += 1
            colnnzs[col] += 1

        self.rowlogs, self.rownnzs = rowlogs, rownnzs
        self.collogs, self.colnnzs = collogs, colnnzs
        
    def applyscaling(self, A, b, c):
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

        print "Scaling done."
