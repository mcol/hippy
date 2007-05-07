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
from scipy import sparse

class Mps:

    def __init__(self, mpsfile):
        self.mpsfile = mpsfile
        self.rowNames = {}
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
        return A, array(self.rhs), array(self.obj)

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
                rows.append(self.rowNames[line[3]])
                data.append(float(line[4]))
                nnnz += 1

        # add the last element
        ptrs.append(nnnz)

        self.data, self.rows, self.ptrs, self.obj = data, rows, ptrs, obj


    def __parseRhs(self, mps):
        rhs = []
        for line in mps:

            line = string.split(line)
            if (line[0] == "ENDATA"):
                break

            rhs.append(float(line[2]))

        self.rhs = rhs
