#!/usr/bin/python
#
# mps.py
#
# Routines for files in MPS format.
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

from numpy import array, unique
from scipy import sparse

class Mps:

    # common error messages
    errNumEntries = "Expected exactly %s entries in the %s section."
    errUnknownRow = "Unknown row '%s' in the %s section."
    errUnknownCol = "Unknown column '%s' in the %s section."

    def __init__(self, mpsfile):
        '''Constructor.'''
        self.mpsfile = mpsfile
        self.rowNames = {}
        self.rowTypes = {}
        self.colNames = {}
        self.objName  = None

    def readMps(self):
        '''Read the MPS file.'''
        try:
            mps = open(self.mpsfile, 'r')

            print "Reading file", self.mpsfile + "."

            self.__parseRows(mps)
            self.__parseColumns(mps)
            self.__parseRhs(mps)
            self.__parseBounds(mps)

            self.__deleteEmptyRows()

            mps.close()

        except IOError:
            print "Could not open file '" + self.mpsfile + "'."
            raise IOError
        except (IndexError, NotImplementedError):
            print "Parsing of the MPS file interrupted."
            raise IndexError

    def getdata(self):
        '''Get the coefficient matrix and vectors from the MPS data.'''
        A = sparse.csc_matrix((array(self.data), self.rows, self.ptrs))
        bounds = self.bup, self.upx, self.blo, self.lox
        return A, array(self.rhs), array(self.obj), bounds

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

        # original length of the rhs vector
        rhslen = len(self.rhs)

        # go through all row indices to compute the adjustment
        for i in range(len(u)):
            while (u[i] != i + removed):
                del self.rhs[i]
                removed += 1
                adjust.append(removed)
            adjust.append(removed)

        # check that we have removed all empty rows from the rhs
        rem = rhslen - (len(u) + removed)

        # remove the empty rows at the end of the rhs vector
        if (rem > 0):
            del self.rhs[-rem:]

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

            line = line.split()
            try:
                if line[0] == "COLUMNS":
                    break
                if line[0] == "N":
                    self.objName = line[1]
                    continue
                if line[0] == "NAME" or line[0] == "ROWS" or line[0][0] == "*":
                    continue
            except IndexError:
                # skip an empty line
                continue

            if (len(line) != 2):
                print self.errNumEntries % ("2", "ROWS")
                print "Read: ", line
                raise IndexError

            self.rowNames[line[1]] = rowIndex
            self.rowTypes[line[1]] = line[0]
            rowIndex += 1

        if len(self.rowNames) == 0:
            print "Empty ROWS section in MPS file."
            raise ValueError

    def __parseColumns(self, mps):

        prev, nnnz, indx = 0, 0, 0
        data, rows, ptrs, obj = [], [], [], []
        for line in mps:

            line = line.split()
            try:
                if line[0] == "RHS":
                    break
                if line[0][0] == "*":
                    continue
            except IndexError:
                # skip an empty line
                continue

            if (len(line) != 3 and len(line) != 5):
                print self.errNumEntries % ("3 or 5", "COLUMNS")
                print "Read: ", line
                raise IndexError

            if line[0] != prev:
                self.colNames[line[0]] = indx
                ptrs.append(nnnz)
                prev = line[0]
                indx += 1
                obj.append(0)

            if (line[1] == self.objName):
                obj[indx - 1] = float(line[2])
            else:
                try:
                    rows.append(self.rowNames[line[1]])
                    data.append(float(line[2]))
                    nnnz += 1
                except KeyError:
                    print self.errUnknownRow % (line[1], "COLUMNS")
                    raise ValueError

            if len(line) > 3:
                if (line[3] == self.objName):
                    obj[indx - 1] = float(line[4])
                    continue
                try:
                    rows.append(self.rowNames[line[3]])
                    data.append(float(line[4]))
                    nnnz += 1
                except KeyError:
                    print self.errUnknownRow % (line[3], "COLUMNS")
                    raise ValueError

        if len(self.colNames) == 0:
            print "Empty COLUMNS section in MPS file."
            raise ValueError

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

        # report the number of slacks added
        nslacks = len(obj) - len(self.colNames)
        if (nslacks > 0):
            print "Added %d slacks variables." % nslacks

        self.data, self.rows, self.ptrs, self.obj = data, rows, ptrs, obj

    def __parseRhs(self, mps):
        # create a dense empty right-hand side
        rhs = [0]*len(self.rowNames)

        # declare the variable to be set inside a try block but used outside
        row = 0

        for line in mps:

            line = line.split()
            try:
                if line[0] == "ENDATA" or line[0] == "BOUNDS":
                    break
                if line[0] == "*":
                    continue
            except IndexError:
                # skip an empty line
                continue

            if (len(line) < 2 or len(line) > 5):
                print self.errNumEntries % ("3 or 5", "RHS")
                print "Read: ", line
                raise IndexError

            index = len(line) - 1
            while (index > 0):
                try:
                    val = float(line[index])
                    row = self.rowNames[line[index - 1]]
                except KeyError:
                    # ignore the assignment of right-hand side to the objective
                    if (self.objName == line[index - 1]): pass
                    else:
                        print self.errUnknownRow % (line[index - 1], "RHS")
                        raise ValueError
                except ValueError:
                    # line[index] is not a numerical value
                    print "Unexpected field ordering in the RHS section."
                    print "Read: ", line
                    raise IndexError

                rhs[row] = float(line[index])
                line = line[:-2]
                index -= 2

        self.rhs = rhs

    def __parseBounds(self, mps):

        # create sparse vectors for the bounds
        bup, upx = [], []
        blo, lox = [], []

        for line in mps:

            line = line.split()
            try:
                if line[0] == "ENDATA":
                    break
                if line[0] == "*":
                    continue
            except IndexError:
                # skip an empty line
                continue

            if (line[0] == "FR"):
                print "Bound type", line[0], "not supported."
                raise NotImplementedError

            if (len(line) < 3 or len(line) > 4):
                print self.errNumEntries % ("4", "BOUNDS")
                print "Read: ", line
                raise IndexError

            try:
                value = float(line[-1])
                index = self.colNames[line[-2]]
            except KeyError:
                print self.errUnknownCol % (line[-2], "BOUNDS")
                raise ValueError

            if (line[0] == "UP"):
                bup.append(value)
                upx.append(index)

            elif (line[0] == "LO"):
                blo.append(value)
                lox.append(index)

            elif (line[0] == "FX"):
                bup.append(value)
                upx.append(index)
                blo.append(value)
                lox.append(index)

        self.bup, self.upx = bup, upx
        self.blo, self.lox = blo, lox
