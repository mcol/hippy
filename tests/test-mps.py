#!/usr/bin/python
#
# test-mps.py
#
# Tests for the Mps class.
#
# Copyright (c) 2012 Marco Colombo
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# See http://www.gnu.org/licenses/gpl.txt for a copy of the license.
#

import sys
sys.path.append('..')
from mps import Mps
import unittest

class TestMps(unittest.TestCase):

    def test_NoFile(self):
        self.assertRaises(IOError, Mps, "")

    def test_Error000(self):
        self.assertRaises(ValueError, Mps, "mps/er000")

    def test_Error001(self):
        self.assertRaises(ValueError, Mps, "mps/er001")

    def test_Error002(self):
        self.assertRaises(ValueError, Mps, "mps/er002")

    def test_Error003(self):
        self.assertRaises(ValueError, Mps, "mps/er003")

    def test_Error004(self):
        self.assertRaises(ValueError, Mps, "mps/er004")

    def test_Error005(self):
        self.assertRaises(ValueError, Mps, "mps/er005")

    def test_Error006(self):
        self.assertRaises(ValueError, Mps, "mps/er006")

    def test_Error007(self):
        self.assertRaises(ValueError, Mps, "mps/er007")

    def test_Error008(self):
        self.assertRaises(IndexError, Mps, "mps/er008")

    def test_NotImplemented000(self):
        self.assertRaises(NotImplementedError, Mps, "mps/un000")

    def test_NotImplemented001(self):
        self.assertRaises(NotImplementedError, Mps, "mps/un001")

    def test_DeleteEmptyRows000(self):
        mps = Mps("mps/em000")
        self.assertEqual(mps.rowNames.keys(), ["R09", "R10"])


if __name__ == '__main__':
    unittest.main()
