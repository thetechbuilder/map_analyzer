#Licence: Public Domain
#LGPL-parts have been used in the form of the related third party inputs
#
#imgsnipper.py
#
"""Implements facilities for chopping and snipping image parts as a means 
to provide further operations, image processing-wise.

Unit tests are in the current directory (test_imgsnipper.py)"""
#standard modules:
from functools import reduce
from operator import mul
from abc import abstractproperty
from collections.abc import Sequence as _Sequence
#third-party modules:
from pygame import Surface, PixelArray
#local(included) mosules:
from pyextra import Validators

__all__ = ["PixelParser", "ImageSnipperBase", "RectangularSnipper"]

class PixelParser:
    """Implements methods for parsing pixels"""
    def __init__(self, mode):
        """
        P.__init__(mode) -> PixelParser 
        
        Initialize pixel parser.
        Possible modes:
        1) 'red' -- parse red part
        2) 'green' -- parse green part
        3) 'blue' -- parse blue part
        """
        self.__parser = getattr(PixelParser, mode)

    def __call__(self, pix):
        return self.__parser(pix)

    def __str__(self):
        """P.__str__() -> str(P)"""
        return "<{}:{}>".format(
                self.__class__.__name__, 
                self.__parser.__name__)

    def __repr__(self):
        """P.__repr__() -> repr(P)"""
        return "{}('{}')".format(self.__class__.__name__, 
                self.__parser.__name__)

    @staticmethod
    def red(pix):
        """I.red(pix) -> int -- extract red agent"""
        return pix>>16

    @staticmethod
    def green(pix):
        """I.green(pix) -> int -- extract green agent"""
        return pix>>8&0xFF

    @staticmethod
    def blue(pix):
        """I.blue(pix) -> int -- extract blue agent"""
        return pix&0xFF

class ImageSnipperBase(_Sequence):
    """
    Abstract base class that is designed for all implementations snipping 
    image blocks
    """
    def __init__(self, surface, bwidth, bheight, pix_parser = lambda x:x):
        """
        I.__init__(surface, width, height) -> ImageSniper
        """
        if not isinstance(surface, Surface):
            raise TypeError("The specified surface source does not "
                    "implement pygame.Surface")
        bwidth, bheight = int(bwidth), int(bheight)
        if bwidth < 1 or bheight < 1:
            raise ValueError("The size of the block must be more "
                    "than one pixel, ({}, {}) specified.".format(
                        bwidth, bheight))
        if surface.get_width() < bwidth or surface.get_height() < bheight:
            raise ValueError("Snippet dimmensions are out of range. "
                    "Image size:{} < Block size:{}".format(
                        surface.get_size(), (bwidth, bheight)))
        self.__bwidth, self.__bheight = bwidth, bheight
        pxarr = PixelArray(surface)
        shape = pxarr.shape
        self.__pxarray = [
                [pix_parser(pxarr[x, y]) for x in range(shape[0])] 
                for y in range(shape[1])]

    @property
    def pxarray(self):
        """
        I.pxarray -> pygame.PixelArray
        
        Returns the pixel array of the source image
        """
        return self.__pxarray

    @property
    def bshape(self):
        """
        I.bshape -> tuple -- block dimmensions -- (width, height)

        Returns the block size by giving the length of each dimmension.
        It is the same as tuple(I.bwidth, I.bheight).
        """
        return self.bwidth, self.bheight

    @property
    def bwidth(self):
        """
        I.bwidth -> int -- block width

        Returns the width of the block in pixels
        """
        return self.__bwidth

    @property
    def bheight(self):
        """
        I.bheight -> int -- block height
        
        Returns the height of the block in pixels
        """
        return self.__bheight

    @abstractproperty
    def shape(self):
        """Represents the shape of the specified image in blocks"""
        raise NotImplementedError()

class RectangularSnipper(ImageSnipperBase):
    """
    Implements rectangular image snipper. 
    
    The snipper provides facilities for extruding rectangular subviews 
    from the specified image.
    """
    def __len__(self):
        """I.__len__() -> int -- the number of blocks"""
        return reduce(mul, self.shape)

    def __getitem__(self, key):
        self_length = len(self)
        if isinstance(key, _Sequence):
            raise NotImplementedError("Sequence index assigning is not "
                    "implemented")
        elif isinstance(key, slice):
            return [self.get_item(*self.__getdindex(k)) 
                    for k in range(*key.indices(self_length))]
        elif isinstance(key, int):
            return self.get_item(*self.__getdindex(
                Validators.arrayindex(key, self_length)))
        raise TypeError("Wrong type of the specified index.")

    def get_item(self, x, y):
        """
        I.get_item(x, y) -> new array
        """
        pix_width, pix_height = self.shape
        if x >= pix_width or y >= pix_height:
            raise ValueError("The index is out of range")
        x_start, y_start = x*self.bwidth, y*self.bheight
        result = []
        for y in range(y_start,y_start + self.bheight):
            result.extend(self.pxarray[y][x_start:x_start + self.bwidth])
        return result
    
    def __getdindex(self, i):
        """Get double index from single one"""
        w, h = self.shape
        return i%w, i//w

    @property
    def shape(self):
        """
        I.shape -> tuple -- image dimmensions in blocks (width, height)
        """
        return (len(self.pxarray[0])//self.bwidth, 
                len(self.pxarray)//self.bheight)
