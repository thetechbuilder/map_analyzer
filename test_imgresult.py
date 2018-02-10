#!/usr/bin/env python3
#
#test_imgresult.py
#
"""Unit test for imgresult.py"""

#standard modules:
import unittest
from random import randint
#local(included) modules:
from imgresult import MatrixResult

class TestImageResult(unittest.TestCase):
    def setUp(self):
        self.shape = randint(1, 100), randint(1, 100)
        self.initlist = [[randint(1, 10) for x in range(self.shape[0])] 
                for y in range(self.shape[1])]

    def test_initialize(self):
        self.assertRaises(TypeError, MatrixResult, 1)
        self.assertRaises(TypeError, MatrixResult, [[1,3,4], 4])
        self.assertRaises(ValueError, MatrixResult, [[1,3,4], [1,2,3,4]])
        self.assertRaises(ValueError, MatrixResult, [[], []])
        self.assertRaises(ValueError, MatrixResult, [[]])
        self.assertRaises(ValueError, MatrixResult, [])
        
        w, h = self.shape
        M = MatrixResult(self.initlist)
        self.assertEqual(len(M), w*h)
        self.assertEqual(M.width, w)
        self.assertEqual(M.height, h)
        self.assertSequenceEqual(M.shape, (w, h))

    def test_get_set(self):
        for x in range(10):
            MR = MatrixResult(self.initlist)
            rand_index = (randint(0, MR.width - 1), 
                    randint(0, MR.height - 1))
            rand_value = randint(0, 100)
            MR[rand_index] = rand_value
            self.assertEqual(MR[rand_index], rand_value)

    def test_string_repr(self):
        MR = MatrixResult(self.initlist)
        mrstr = MR.to_string()
        self.assertSequenceEqual(self.initlist, MR[:,:])
        self.assertSequenceEqual(self.initlist, 
            MatrixResult.from_string(mrstr)._MatrixResult__matrix)

#######################################################################
### Run tests
#######################################################################

def _test():
    unittest.main(verbosity = 2)

#run tests:
if __name__ == '__main__':
    _test()
