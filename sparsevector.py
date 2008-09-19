#!/usr/bin/python
#
# sparsevector.py
#
# Implementation of a sparse vector class.
#
# Copyright (c) 2008 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

from numpy import array

class Sparsevector:

    def __init__(self, size, values, indices):
        '''Constructor.'''
        self.dim = size
        self.val = array(values)
        self.idx = indices

    def __getitem__(self, index):
        return self.val[index]

    def __setitem__(self, index, value):
        self.val[index] = value

    def __len__(self):
        return len(self.idx)

    def __eq__(self, other):
        ret = (self.idx == other.idx and
               self.dim == other.dim and
               (self.val == other.val).all())
        return ret

    def __abs__(self):
        return Sparsevector(self.dim, abs(self.val), self.idx)

    def __add__(self, other):
        if hasattr(other, "val"):
            return Sparsevector(self.dim, self.val + other.val, self.idx)
        else:
            return Sparsevector(self.dim, self.val + other, self.idx)

    def __sub__(self, other):
        if hasattr(other, "val"):
            return Sparsevector(self.dim, self.val - other.val, self.idx)
        else:
            return Sparsevector(self.dim, self.val - other, self.idx)

    def __mul__(self, other):
        if hasattr(other, "val"):
            return Sparsevector(self.dim, self.val * other.val, self.idx)
        else:
            return Sparsevector(self.dim, self.val * other, self.idx)

    def __rmul__(self, other):
        return self * other

    def __div__(self, other):
        if hasattr(other, "val"):
            return Sparsevector(self.dim, self.val / other.val, self.idx)
        else:
            return Sparsevector(self.dim, self.val / other, self.idx)

    def __neg__(self):
        return Sparsevector(self.dim, -self.val, self.idx)

    def __repr__(self):
        fmt = "(%f, %d) "
        str = "[ "
        for i in range(len(self)):
            str += fmt % (self.val[i], self.idx[i])
        str += "]"
        return str

    def todense(self):
        '''Return a dense version of the vector.'''
        v = [0.0] * self.dim
        for i in range(len(self)):
            v[self.idx[i]] = self.val[i]
        return array(v)

    def array(self):
        '''Return the array of sparse values.'''
        return self.val
