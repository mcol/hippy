#!/usr/bin/python
#
# mps.py
#
# Routines for files in MPS format.
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

import string
from numpy import array
from scipy import sparse, matrix

class Mps:

    def __init__(self, mpsfile):
        self.mpsfile = mpsfile
        self.rowNames = {}
        self.rowTypes = {}
        self.objName  = None
        self.b = None

    def readMps(self):

        print "Reading file", self.mpsfile + "."

        mps = open(self.mpsfile, 'r')

        self.__parseRows(mps)
        self.__parseColumns(mps)
        self.__parseRhs(mps)

        mps.close()

    def getdata(self):
        A = sparse.csc_matrix((array(self.data), self.rows, self.ptrs))
        return A, matrix(self.rhs).T, matrix(self.obj).T

    def __parseRows(self, mps):

        rowIndex = 0

        for line in mps:

            line = string.split(line)

            if (line[0] == "COLUMNS"):
                return
            elif (line[0] == "N"):
                self.objName = line[1]
                continue
            elif (line[0] == "NAME" or line[0] == "ROWS"):
                continue

            self.rowNames[line[1]] = rowIndex
            self.rowTypes[line[1]] = line[0]
            rowIndex += 1

    def __parseColumns(self, mps):

        prev, nnnz, indx = 0, 0, 0
        data, rows, ptrs, obj = [], [], [], []
        for line in mps:

            line = string.split(line)
            if (line[0] == "RHS"):
                break

            if line[0] != prev:
                ptrs.append(nnnz)
                prev = line[0]
                indx += 1
                obj.append(0)

            if (line[1] == self.objName):
                obj[indx - 1] = float(line[2])
                continue

            rows.append(self.rowNames[line[1]])
            data.append(float(line[2]))
            nnnz += 1

            if len(line) > 3:
                if (line[3] == self.objName):
                    obj[indx - 1] = float(line[4])
                    continue
                rows.append(self.rowNames[line[3]])
                data.append(float(line[4]))
                nnnz += 1

        # add slacks for inequality constraints
        keys = self.rowTypes.keys()
        for key in keys:
            if self.rowTypes[key] is 'E':
                continue
            if self.rowTypes[key] is 'L':
                data.append(1.0)
            elif self.rowTypes[key] is 'G':
                data.append(-1.0)

            rows.append(self.rowNames[key])
            ptrs.append(nnnz)
            obj.append(0)
            nnnz += 1

       # add the last element
        ptrs.append(nnnz)

        self.data, self.rows, self.ptrs, self.obj = data, rows, ptrs, obj

    def __parseRhs(self, mps):
        # create a dense empty right-hand side
        rhs = [0]*len(self.rowNames)
        for line in mps:

            line = string.split(line)
            if (line[0] == "ENDATA"):
                break

            rhs[self.rowNames[line[1]]] = float(line[2])
            if len(line) > 3:
                rhs[self.rowNames[line[3]]] = float(line[4])

        self.rhs = rhs
