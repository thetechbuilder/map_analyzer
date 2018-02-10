#! /usr/bin/env python3
#
#test_pyextra.py
#
"""Unit tests for pyextra.py"""

#standard modules:
import unittest
from random import randint
#local(included) modules:
from pyextra import Validators 

#global parameters:
_NUM = 10 #number of invokes per test


class TestSequenceHelper(unittest.TestCase):
    def test_arrayindex(self):
        #apply inappropriate values:
        
        for x in range(_NUM):
            rnd = randint(0, 1000)
            self.assertRaises(IndexError, Validators.arrayindex, 
                    -rnd, rnd - 1)
            self.assertRaises(IndexError, Validators.arrayindex, 
                    -rnd - 1, rnd)
            self.assertRaises(IndexError, Validators.arrayindex, 
                    rnd, rnd)
            self.assertRaises(IndexError, Validators.arrayindex, 
                    rnd, 2)
            self.assertRaises(TypeError, Validators.arrayindex, 
                    str(rnd), rnd)

            rnd2 = randint(0, 1000)
            self.assertEqual(
                    Validators.arrayindex(-rnd, rnd + rnd2), rnd2)
            self.assertEqual(
                    Validators.arrayindex(rnd, rnd + rnd2), rnd)

    def test_bfileobject(self):
        self.assertRaises(TypeError, Validators.bfileobject, [])
        self.assertRaises(FileNotFoundError, Validators.bfileobject, 
                "/xml_jj_test_unreal_path/ss.d")

##########################################################################
### Run tests
##########################################################################

def _test():
    unittest.main(verbosity = 2)

#run tests
if __name__ == '__main__':
    _test()
