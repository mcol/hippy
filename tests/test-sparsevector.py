#!/usr/bin/python
#
# test-sparsevector.py
#
# Tests for the Sparsevector class.
#
# Copyright (c) 2008, 2012 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

import sys
sys.path.append("..")
import unittest
from sparsevector import Sparsevector
from numpy import array

class TestSparsevector(unittest.TestCase):

    idx = [2, 3, 8, 9]
    lls = [1, 0, 2, 1]
    v = Sparsevector(10, [1.0, 2.0, 3.0, 4.0], idx)
    x = Sparsevector(10, [1.0, 2.0, 3.0, 4.0], idx)
    w = Sparsevector(10, [4, 3, 2, 1], idx)
    z = Sparsevector(10, [5, 5, 5, 5], idx)
    v2 = Sparsevector(10, [2, 4, 6, 8], idx)
    vv = Sparsevector(10, [2, 3, 4, 5], idx)
    vw = Sparsevector(10, [4, 6, 6, 4], idx)
    vn = Sparsevector(10, [-1, -2, -3, -4], idx)
    vlbef = Sparsevector(10, [1.0, 2.0, 3.0, 4.0], [2, 3, 8, 9])
    vldel = Sparsevector(10, [1.0, 2.0, 4.0], [2, 3, 9])
    vladd = Sparsevector(10, [2, 2, 5, 5], idx)
    vlsub = Sparsevector(10, [0, 2, 1, 3], idx)
    vlmul = Sparsevector(10, [1, 0, 6, 4], idx)
    vldiv = Sparsevector(10, [1, float('inf'), 1.5, 4], idx)
    vdense = array([0, 0, 1, 2, 0, 0, 0, 0, 3, 4])

    def test_del01(self):
        res = self.vlbef
        del res[2]
        self.assertEqual(res, self.vldel)

    def test_add01(self):
        res = self.v + 1
        self.assertEqual(res, self.vv)

    def test_add02(self):
        res = self.v + self.w
        self.assertEqual(res, self.z)

    def test_add03(self):
        res = self.v + self.lls
        self.assertEqual(res, self.vladd)

    def test_sub01(self):
        res = self.vv - 1
        self.assertEqual(res, self.v)

    def test_sub02(self):
        res = self.z - self.v
        self.assertEqual(res, self.w)

    def test_sub03(self):
        res = self.v - self.lls
        self.assertEqual(res, self.vlsub)

    def test_mul01(self):
        res = self.v * 2
        self.assertEqual(res, self.v2)

    def test_mul02(self):
        res = self.v * self.w
        self.assertEqual(res, self.vw)

    def test_mul03(self):
        res = self.v * self.lls
        self.assertEqual(res, self.vlmul)

    def test_mul04(self):
        res = 2 * self.v
        self.assertEqual(res, self.v2)

    def test_div01(self):
        res = self.v2 / 2
        self.assertEqual(res, self.v)

    def test_div02(self):
        res = self.vw / self.v
        self.assertEqual(res, self.w)

    def test_div03(self):
        res = self.v / self.lls
        self.assertEqual(res, self.vldiv)

    def test_neg01(self):
        res = -self.v
        self.assertEqual(res, self.vn)

    def test_abs01(self):
        res = abs(self.vn)
        self.assertEqual(res, self.v)

    def test_len01(self):
        res = len(self.v)
        self.assertEqual(res, 4)

    def test_assignment01(self):
        new = self.v
        new = new + self.w
        self.assertEqual(self.v, self.x)

    def test_assignment02(self):
        new = self.v
        new = new * self.z
        self.assertEqual(self.v, self.x)

    def test_assignment03(self):
        new = self.v
        new = self.x
        self.assertEqual(self.v, self.x)

    def test_remove01(self):
        a = Sparsevector(10, [1.0, 2.0, 4.0], [2, 3, 9])
        self.assertRaises(IndexError, a.remove, 4)

    def test_remove02(self):
        a = Sparsevector(10, [1.0, 2.0, 4.0], [2, 4, 9])
        b = Sparsevector(9, [2.0, 4.0], [3, 8])
        a.remove(0)
        self.assertEqual(a, b)

    def test_remove03(self):
        a = Sparsevector(10, [1.0, 2.0, 4.0], [2, 4, 9])
        b = Sparsevector(9, [1.0, 4.0], [2, 8])
        a.remove(1)
        self.assertEqual(a, b)

    def test_remove04(self):
        a = Sparsevector(10, [1.0, 2.0, 4.0], [2, 4, 9])
        b = Sparsevector(9, [1.0, 2.0], [2, 4])
        a.remove(2)
        self.assertEqual(a, b)

    def test_todense01(self):
        res = self.v.todense()
        self.assertEqual(res.all(), self.vdense.all())


if __name__ == '__main__':
    unittest.main()
