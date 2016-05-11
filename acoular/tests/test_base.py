# -*- coding: utf-8 -*-
"""
Created on Mon Nov 30 20:22:25 2015

@author: sarradj
"""
from numpy.testing import *

class MyTestCase(TestCase):

    def test1(self):
        self.assertEqual(1,2)

    def test2(self):
        self.assertEqual(1,1)

    def test3(self):
        self.assertAlmostEqual(1.00000001,1.0)
        
