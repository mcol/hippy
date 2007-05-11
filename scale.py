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

from numpy import log

class Scale:

    def __init__(self, A):
        self.factor = 0.0
        self.scalefactor(A)
        print "Scale factor:", self.factor

    def scalefactor(self, A):
        # scalefactor = Sum (log |Aij|)^2
        # loop over the nonzero entries
        for i in range(A.getnnz()):
            a = abs(A.getdata(i))
            self.factor += log(a)**2
