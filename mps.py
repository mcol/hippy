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

from collections import OrderedDict
from numpy import array, unique
from scipy import sparse

class Mps:

    # common error messages
    errNumEntries = "Expected exactly %s entries in the %s section."
    errFieldOrder = "Unexpected field ordering in the %s section."
    errUnknownRow = "Unknown row '%s' in the %s section."
    errUnknownCol = "Unknown column '%s' in the %s section."
    errLastParsed = "\nLast line parsed: %s."

    def __init__(self, mpsfile):
        '''Constructor.'''
        self.mpsfile = mpsfile
        self.rowNames = OrderedDict()
        self.rowTypes = {}
        self.colNames = {}
        self.rhsNames = {}
        self.objName  = None

        try:
            # read the MPS file
            mps = open(self.mpsfile, 'r')

            self.__parseRows(mps)
            self.__parseColumns(mps)
            self.__parseRhs(mps)
            self.__parseBounds(mps)

            self.__deleteEmptyRows()

            mps.close()

        except IOError:
            msg = "Could not open file '" + self.mpsfile + "'."
            raise IOError(msg)

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
        idxempty = [x for x in range(len(self.rowNames)) if x not in u]

        # go through all row indices to compute the adjustment
        keys, idx = self.rowNames.keys(), 0
        for key in keys:
            if idx in idxempty:
                del self.rowNames[key]
                del self.rowTypes[key]
                try:
                    del self.rhsNames[key]
                except KeyError:
                    pass
                removed += 1
            adjust.append(removed)
            idx += 1

        # from each row index subtract the number of removed rows
        try:
            self.rows = [i - adjust[i] for i in self.rows]
        except:
            print "Error in __deleteEmptyRows()."
            raise

        # update the indices stored in self.rowNames and create a dense rhs
        self.rhs = [0] * len(self.rowNames)
        idx = 0
        for key in self.rowNames.iterkeys():
            self.rowNames[key] = idx
            try:
                self.rhs[idx] = self.rhsNames[key]
            except:
                pass
            idx += 1

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
                msg = self.errNumEntries % ("2", "ROWS")
                msg += self.errLastParsed % str(line)
                raise IndexError(msg)

            self.rowNames[line[1]] = rowIndex
            self.rowTypes[line[1]] = line[0]
            rowIndex += 1

        if len(self.rowNames) == 0:
            msg = "Empty ROWS section in MPS file."
            raise ValueError(msg)

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
                msg = self.errNumEntries % ("3 or 5", "COLUMNS")
                msg += self.errLastParsed % str(line)
                raise IndexError(msg)

            if line[0] != prev:
                self.colNames[line[0]] = indx
                ptrs.append(nnnz)
                prev = line[0]
                indx += 1
                obj.append(0)

            # no need to access the variable name anymore
            line = line[1:]
            while len(line) > 0:
                rowname = line[0]
                try:
                    if rowname == self.objName:
                        obj[indx - 1] = float(line[1])
                    else:
                        rows.append(self.rowNames[rowname])
                        data.append(float(line[1]))
                        nnnz += 1
                except KeyError:
                    msg = self.errUnknownRow % (rowname, "COLUMNS")
                    raise ValueError(msg)
                except ValueError:
                    msg = self.errFieldOrder % "COLUMNS"
                    msg += self.errLastParsed % str(line)
                    raise IndexError(msg)

                # remove the parts of the line just parsed
                line = line[2:]

        if len(self.colNames) == 0:
            msg = "Empty COLUMNS section in MPS file."
            raise ValueError(msg)

        # add the last element
        ptrs.append(nnnz)
        self.data, self.rows, self.ptrs, self.obj = data, rows, ptrs, obj

    def __parseRhs(self, mps):

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
                msg = self.errNumEntries % ("3 or 5", "RHS")
                msg += self.errLastParsed % str(line)
                raise IndexError(msg)

            index = len(line) - 1
            while (index > 0):
                try:
                    row = line[index - 1]
                    self.rowNames[row] # ensure that this row name exists
                    self.rhsNames[row] = float(line[index])
                except KeyError:
                    # ignore the assignment of right-hand side to the objective
                    if (self.objName == line[index - 1]): pass
                    else:
                        msg = self.errUnknownRow % (line[index - 1], "RHS")
                        raise ValueError(msg)
                except ValueError:
                    # line[index] is not a numerical value
                    msg = self.errFieldOrder % "RHS"
                    msg += self.errLastParsed % str(line)
                    raise IndexError(msg)

                line = line[:-2]
                index -= 2

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

            if line[0] == "RANGES":
                msg = "RANGES section not supported."
                raise NotImplementedError(msg)

            if (line[0] == "FR"):
                msg = "Bound type FR not supported."
                raise NotImplementedError(msg)

            if (len(line) < 3 or len(line) > 4):
                msg = self.errNumEntries % ("4", "BOUNDS")
                msg += self.errLastParsed % str(line)
                raise IndexError(msg)

            try:
                value = float(line[-1])
                index = self.colNames[line[-2]]
            except KeyError:
                msg = self.errUnknownCol % (line[-2], "BOUNDS")
                raise ValueError(msg)

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
