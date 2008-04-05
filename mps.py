#!/usr/bin/python
#
# mps.py
#
# Routines for files in MPS format.
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

from numpy import array, unique
from scipy import sparse
from string import split

class Mps:

    def __init__(self, mpsfile):
        '''Constructor.'''
        self.mpsfile = mpsfile
        self.rowNames = {}
        self.rowTypes = {}
        self.objName  = None

    def readMps(self):
        '''Read the MPS file.'''
        try:
            mps = open(self.mpsfile, 'r')

            print "Reading file", self.mpsfile + "."

            self.__parseRows(mps)
            self.__parseColumns(mps)
            self.__parseRhs(mps)

            self.__deleteEmptyRows()

            mps.close()

        except IOError:
            print "Could not open file '" + self.mpsfile + "'."
            raise IOError
        except IndexError:
            print "Parsing interrupted."
            raise IndexError

    def getdata(self):
        '''Get the coefficient matrix and vectors from the MPS data.'''
        A = sparse.csc_matrix((array(self.data), self.rows, self.ptrs))
        return A, array(self.rhs), array(self.obj)

    def __deleteEmptyRows(self):

        # create a list to reorder the rows in case empty rows are found:
        # in each position we put the correction to the row index
        # e.g.: rows = [0 1 0 1 3 1 3]
        #       u = [0 1 3] (row 2 is empty)
        #       adjust = [0 0 0 1]
        #       so that the reordered array is
        #       rows = [0 1 0 1 2 1 2]
        adjust  = []
        removed = 0

        # get an ordered list of the row indices
        u = unique(self.rows)

        # go through all row indices to compute the adjustment
        for i in range(len(u)):
            while (u[i] != i + removed):
                del self.rhs[i]
                removed += 1
                adjust.append(removed)
            adjust.append(removed)

        # from each row index subtract the number of removed rows
        try:
            self.rows = [i - adjust[i] for i in self.rows]
        except:
            print "Error in __deleteEmptyRows()."
            raise

        if (removed):
            print "Removed %d empty rows." % removed

    def __parseRows(self, mps):

        rowIndex = 0

        for line in mps:

            line = split(line)

            if (line[0] == "COLUMNS"):
                return
            elif (line[0] == "N"):
                self.objName = line[1]
                continue
            elif (line[0] == "NAME" or line[0] == "ROWS" or line[0] == "*"):
                continue

            if (len(line) != 2):
                print "Expected exactly 2 entries in the ROWS section."
                print "Read: ", line
                raise IndexError

            self.rowNames[line[1]] = rowIndex
            self.rowTypes[line[1]] = line[0]
            rowIndex += 1

    def __parseColumns(self, mps):

        prev, nnnz, indx = 0, 0, 0
        data, rows, ptrs, obj = [], [], [], []
        for line in mps:

            line = split(line)
            if (line[0] == "RHS"):
                break
            elif (line[0] == "*"):
                continue

            if (len(line) != 3 and len(line) != 5):
                print "Expected exactly 3 or 5 entries in the COLUMNS section."
                print "Read: ", line
                raise IndexError

            if line[0] != prev:
                ptrs.append(nnnz)
                prev = line[0]
                indx += 1
                obj.append(0)

            if (line[1] == self.objName):
                obj[indx - 1] = float(line[2])
            else:
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

            line = split(line)
            if (line[0] == "ENDATA"):
                break
            elif (line[0] == "*"):
                continue

            if (len(line) < 2 or len(line) > 5):
                print "Expected 1 or 2 pairs of entries in the RHS section."
                print "Read: ", line
                raise IndexError

            index = len(line) - 1
            while (index > 0):
                rhs[self.rowNames[line[index - 1]]] = float(line[index])
                line = line[:-2]
                index -= 2

        self.rhs = rhs
