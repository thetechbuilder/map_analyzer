#!/usr/bin/env python3
#
#test_imgselection.py
#
"""Unit tests for imgselection.py"""

#standard modules:
import os
import unittest
from random import randint
#third party modules
from pygame import image
#local modules:
from imgresult import MatrixResult
from imgselection import ImageSelection
from imgselection import ISSerializer

test_file = "test_image_sample.jpeg"
test_outs = "test_serialized_selection.xml"

class TestImageSelection(unittest.TestCase):
    def test_initialization(self):
        self.assertRaises(TypeError, ImageSelection, 10, 10, 10)
        self.assertRaises(TypeError, ImageSelection, "10", 10)
        self.assertRaises(TypeError, ImageSelection, 10, "10")
        self.assertRaises(ValueError, ImageSelection, 0, 1)
        self.assertRaises(ValueError, ImageSelection, 1, 0)

    def test_read_write(self):
        surface = image.load(test_file)
        width, height = surface.get_width(), surface.get_height()
        #test image size must be more than 50/50
        block_width, block_height = width//50, height//50
        sel = ImageSelection(block_width, block_height)
        res = MatrixResult([[randint(0, 10) for y in range(
            width//block_width)] for x in range(height//block_height)])

        self.assertRaises(FileNotFoundError, sel.append, 
                ('this_file_do_not_exist_in_this_directory', res))
        self.assertRaises(TypeError, sel.append, (test_file, []))

        sel.append((test_file, res))
        #get image:
        self.assertTrue(sel[0, lambda x:x])
        ISSerializer.write(sel, test_outs)
        parsed = ISSerializer.parse(test_outs)
        
        os.remove(test_outs)

##########################################################################
### Run tests
##########################################################################
def _test():
    unittest.main(verbosity = 2)

#run tests
if __name__ == '__main__':
    _test()
