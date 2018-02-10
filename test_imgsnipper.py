#!/usr/bin/env python3
#
#test_imgsnipper.py
#
"""Unit tests for imgsnipper.py"""

#standard modules:
import unittest
from os import remove
from operator import add
from random import randint
from functools import reduce
#third-party modules:
from pygame import image, PixelArray, Color, Surface
#local modules:
from imgsnipper import PixelParser
from imgsnipper import RectangularSnipper

class TestImageSnipper(unittest.TestCase):
    def setUp(self):
        #generate test values:
        self.surf = TestImageSnipper._generate_random_surface(
                randint(10, 100), randint(10, 100))

    def test_rectangular_snipper_initialize(self):
        self.assertRaises(ValueError, RectangularSnipper, self.surf, 
                self.surf.get_width() + 1, randint(1, 10)) #out of range
        self.assertRaises(ValueError, RectangularSnipper, self.surf, 
                randint(1, 10), self.surf.get_height() + 1) #out of range
        self.assertRaises(ValueError, RectangularSnipper, 
                self.surf, 10, 0) #zero height
        self.assertRaises(ValueError, RectangularSnipper, 
                self.surf, 0, 10) #zero width
        self.assertRaises(ValueError, RectangularSnipper, 
                self.surf, -1, 10) #negative width
        self.assertRaises(TypeError, RectangularSnipper, [], 1, 1)
    
    def test_rectangular_snipper_properties(self):
        w, h = self.surf.get_width(), self.surf.get_height()
        rs = RectangularSnipper(self.surf, w, h)
        self.assertEqual(rs.bheight, h)
        self.assertEqual(rs.bwidth, w)
        self.assertEqual(len(rs), 1)
        self.assertSequenceEqual(rs.bshape, (w, h))

    def test_rectangular_snipper_blocks(self):
        for x in range(20):
            w, h = self.surf.get_width(), self.surf.get_height() #pixel 
            #dimmensions
            parser = PixelParser.green
            bw, bh = randint(1, w), randint(1, h) #block dimmensions
            rs = RectangularSnipper(self.surf, bw, bh, parser)
            
            parr = PixelArray(self.surf)
            pixarr = [[parser(parr[x, y]) for x in range(w)] 
                    for y in range(h)]

            self.assertEqual(len(pixarr), h)
            self.assertEqual(sum(len(x) for x in pixarr), h*w)
            self.assertEqual(len(rs), (h//bh)*(w//bw))
            self.assertSequenceEqual(rs.shape, 
                    (len(pixarr[0])//bw, len(pixarr)//bh))
            self.assertSequenceEqual(rs.pxarray, pixarr)
            
            lw, lh = w//bw, h//bh #shape of the surface in blocks (maximum
            #width, maximum height)
            last_index = len(rs) - 1#last block
            
            self.assertEqual(last_index, lw*lh - 1)
            self.assertSequenceEqual(rs[last_index], #last block
                    reduce(add, [pixarr[y][(lw - 1)*bw:lw*bw]
                        for y in range((lh - 1)*bh, lh*bh)]))
            self.assertSequenceEqual(rs[last_index], #last block
                    rs.get_item((lw - 1), (lh - 1))) #last block
           
            self.assertRaises(ValueError, rs.get_item, 
                    lw, lh - 1) #out of range
            self.assertRaises(ValueError, rs.get_item, 
                    lw - 1, lh) #out of range
            self.assertRaises(IndexError, rs.__getitem__, len(rs))
            
            rnd_index = randint(0, last_index)
            x_index, y_index = rnd_index%lw, rnd_index//lw
            
            self.assertSequenceEqual(rs.__getitem__(rnd_index), 
                    reduce(add, (pixarr[y][bw*x_index:bw*x_index+bw]
                        for y in range(bh*y_index, bh*y_index+bh)), []))

            rnd_slice = slice(
                    randint(0, last_index), randint(0, last_index))
            w, h = rs.shape
            result = [reduce(add, (pixarr[y][bw*w:bw*(w+1)] 
                for y in range(bh*h, bh*(h+1))), [])
                for (w, h) in map(lambda x: (x%lw, x//lw), 
                    range(*rnd_slice.indices(len(rs))))]
            
            self.assertSequenceEqual(rs[rnd_slice], result) #select all

    @staticmethod
    def _generate_random_surface(width, height):
        surf = Surface((width, height))
        for h in range(height):
            for w in range(width):
                surf.set_at((h, w), Color(randint(0, 255), 
                    randint(0, 255), randint(0, 255)))
        return surf

#######################################################################
### Run tests
#######################################################################

def _test():
    unittest.main(verbosity = 2)

#run tests:
if __name__ == '__main__':
    _test()
