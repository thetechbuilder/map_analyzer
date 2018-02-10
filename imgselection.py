#License: Public Domain
#
#imgselection.py
#
"""Provides the implementation for representing an image learning 
selection

Unit tests are in the current directory (test_imgselection.py)"""
#standard modules:
import imghdr
from abc import ABCMeta
from abc import abstractmethod
import xml.etree.ElementTree as ET
from collections.abc import MutableSequence
#third-party modules:
from pygame import image
#local(included) modules:
import imgresult
import imgsnipper
from pyextra import Validators
from imgsnipper import ImageSnipperBase
from iserializer import ISerializer
from imgsnipper import RectangularSnipper

__all__ = ["ImageSelection", "ISSerializer"]

class ImageSelection(MutableSequence):
    """
    This class provides the concrete representation of image training data 
    that is used to maintain and manage image learning selection.
    """
    def __init__(self, bwidth, bheight, snipper = RectangularSnipper):
        """
        ImageSelection.__init__(bwidht, bheight, snipper = 
        RectangularSnipper) -> ImageSelection
        
        Construncts image selection object.
        """
        if not issubclass(snipper, ImageSnipperBase):
            raise TypeError("The specified snipper must be a subclass "
                    "that derived from imgsnipper.ImageSnipperBase")
        if not isinstance(bwidth, int) or not isinstance(bheight, int):
            raise TypeError("Both the block width and the block height "
                    "must be integers, not ({}, {})".format(
                        type(bwidth), type(bheight)))
        elif bwidth < 1 or bheight < 1:
            raise ValueError("Both the block width and the block height "
                    "must be more than one, not ({}, {})".format(
                        bwidth, bheight))
        self.__bwidth = bwidth
        self.__bheight = bheight
        self.__snipper = snipper
        self.__items = []
        self.__results = []
    
    def __repr__(self):
        """
        S.__repr__() <==> repr(S) -- "official" string representation
        """
        return "{}({}, {}, {})".format(self.__class__.__name__, 
                self.__snipper.__name__, self.__bwidth, self.__bheight)

    def __len__(self):
        """
        S.__len__() <==> len(S)
        """
        return len(self.__items)

    def __getitem__(self, key):
        """
        S.__getitem__((index, *parser)) -> tuple
        Return value: results, ImageSnipper [,ImageSniper, ...]
        
        * 'parser' -- any one-argument function for mapping pixels
        * 'index' -- index of the desired item
        """
        if len(key) < 2:
            raise IndexError("The specified key can not be allowed. "
                    "The key must include one index and at least one "
                    "parser function")
        index  = Validators.arrayindex(key[0], len(self))
        surface = image.load(self.__items[index])
        return ((self.__results,) + tuple(self.__snipper(surface, 
            self.__bwidth, self.__bheight, pix_parser = parser) 
            for parser in key[1:]))

    def __setitem__(self, index, value):
        """S.__setitem__(i, v) <==> S[i] = v"""
        index = Validators.arrayindex(index, len(self))
        index = self.insert(index, value)
        del self[index + 1]
    
    def __delitem__(self, index):
        """S.__delitem__(index) <==> del I[index] -- delete item"""
        del self.__items[index]
        del self.__results[index]

    def insert(self, index, value):
        """
        S.insert item before the index
        """
        path, result = value
        if not isinstance(result, imgresult.ImageResultBase):
            raise TypeError("The 'result' must implement "
                    "imgresult.ImgageResultBase")
        index = Validators.arrayindex(index, len(self) + 1)
        if not imghdr.what(path) in ('jpeg', 'png', 'gif', 
                'bmp', 'tiff', 'pbm', 'bgm', 'ppm' ):
            return TypeError("The specified file is not supported")
        self.__items.insert(index, path)
        self.__results.insert(index, result)
        return index

    def paths(self):
        """S.paths() -> tuple -- return all paths"""
        return tuple(self.__items)

    def results(self):
        """S.results() -> results -- return all results"""
        return tuple(self.__results)

    @property
    def snipper(self):
        """
        S.snipper -> ImageSnipper, int (block width), int (block height)
        """
        return self.__snipper, self.__bwidth, self.__bheight

    @property
    def bshape(self):
        """S.bshape -> tuple(width, height) -- block shape"""
        return self.__bwidth, self._bheight

class ISSerializer(ISerializer):
    """
    Provides a simple implementation for serialization to and from 
    standard XML. It can be used for parsing/writing ImageSelection 
    objects from/to an XML-file
    """
    TAG_ROOT = "imgselection"
    TAG_IMG = "image"
    ATTR_PATH = "path"
    ATTR_SNIPPER = "snipper"
    ATTR_RESULT_FUNC = "result_function"
    ATTR_BLOCK_WIDTH = "bwidht"
    ATTR_BLOCK_HEIGHT = "bheight"
    
    @staticmethod
    def write(source, target):
        """
        ISSerializer.write(source, target) -> None
        
        Writes your image selection to the specified file
        * "source" is an instance of ImageSelection or of a subclass
        thereof
        * "target" is either the name of a file of a binary file object
        """
        if not isinstance(source, ImageSelection):
            raise TypeError("Invalid source type. ImageSelection "
                    "implementations are only acceptible")
        Validators.bfileobject(target)
        snipfunc, bwidth, bheight = source.snipper
        #root attributes contain snipper parameters
        root = ET.Element(ISSerializer.TAG_ROOT, attrib = {
            ISSerializer.ATTR_SNIPPER : snipfunc.__name__,
            ISSerializer.ATTR_BLOCK_WIDTH : str(bwidth),
            ISSerializer.ATTR_BLOCK_HEIGHT : str(bheight)})
        #Sub elements provide image links and result values
        for path, result in zip(source.paths(), source.results()):
            sub = ET.SubElement(root, ISSerializer.TAG_IMG, attrib = {
                ISSerializer.ATTR_PATH : path,
                ISSerializer.ATTR_RESULT_FUNC : result.__class__.__name__})
            sub.text = result.to_string()
        ET.ElementTree(root).write(target)

    @staticmethod
    def parse(target, outresult = ImageSelection):
        """
        ISSerializer.parse(target) -> ImageSelection
        
        Loads an image selection from its XML-representation.
        * "target" is either a filename or a file object
        * "outresult" is either the ImageSelection class of a subclass
        thereof
        """
        if not issubclass(outresult, ImageSelection):
            raise TypeError("'outresult' argument be either the"
                    "ImageSelection class or a subclass thereof.")
        Validators.bfileobject(target)
        root =  ET.parse(target).getroot()
        snipfunc = root.get(ISSerializer.ATTR_SNIPPER)
        bwidth = int(root.get(ISSerializer.ATTR_BLOCK_WIDTH))
        bheight = int(root.get(ISSerializer.ATTR_BLOCK_HEIGHT))
        snipfunc = getattr(imgsnipper, snipfunc)
        
        S = outresult(bwidth, bheight, snipper = snipfunc)
        for image in root:
            path = image.get(ISSerializer.ATTR_PATH)
            rfunc = image.get(ISSerializer.ATTR_RESULT_FUNC)
            result = getattr(imgresult, rfunc).from_string(image.text)
            S.append((path, result))
        return S
