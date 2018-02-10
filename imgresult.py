#License: Public Domain
#
#imgresult.py
#
"""This module provides classes that can be used to represent image results

Unit tests are in the current directory (test_imgresult.py)"""
#standard modules
from abc import ABCMeta
from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Sequence
#local(included) modules:
from pyextra import Validators

__all__ = ["ImageResultBase", "ImageResult"]

class ImageResultBase(Sequence):
    """
    Abstract base class for all implementations representing image results.
    This class can be used to test whether a class provides the interface.
    """

    @abstractmethod
    def to_string(self):
        """
        R.to_string() -> str -- cram results into a string
        """
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def from_string(text):
        """
        R.from_string(text) -> str -- extract results from a string
        """
        raise NotImplementedError()

class MatrixResult(ImageResultBase):
    """
    Implements a two dimmensional array for storing classification results.
    """
    def __init__(self, array):
        """
        CategoricalResult.__init__(ctg, length) -> CategoricalResult
        """
        if not isinstance(array, Iterable):
            raise TypeError("The initialization array must be iterable")

        matrix = []
        for row in array:
            if not isinstance(row, Iterable):
                raise TypeError("Initialization values are not iterable")
            matrix.append(list(row))
        
        #check consistency:
        if len(matrix) == 0 or len(matrix[0]) < 1:
            raise ValueError("Both the width and height of the specified " 
                    "source arrya must be more than one.")
        
        if sum(len(row) for row in matrix) != len(matrix)*len(matrix[0]):
            raise ValueError("The specified source array is not "
                    "consistent.")

        self.__matrix = matrix

    def __getitem__(self, key):
        """R.__getitem__((w, h)) <==> R[(w, h)] <==> R[w, h]"""
        if isinstance(key, Sequence):
            w, h = key
            rows = self.__matrix[h]
            if isinstance(h, slice):
                return tuple(r[w] for r in self.__matrix[h])
            return self.__matrix[h][w]
        elif isinstance(key, slice):
            return self.__matrix[key]
        elif isinstance(key, int):
            return tuple(self.__matrix[key])
        raise TypeError("The specified key cannot be allowed,")

    def __setitem__(self, key, value):
        """R.__setitem((w, h), value) <==> R[w, h] = value"""
        if not isinstance(value, int):
            return TypeError("Integer values are only acceptable.")
        w, h = key
        w = Validators.arrayindex(w, self.width)
        h = Validators.arrayindex(h, self.height)
        self.__matrix[h][w] = value

    def __len__(self):
        """R.__len__() <==> len(R) -- number of elements"""
        return self.width*self.height

    @property
    def width(self):
        """R.width -> int -- the width of the result matrix"""
        return len(self.__matrix[0])

    @property
    def height(self):
        """R.height -> int -- the height of the result matrix"""
        return len(self.__matrix)

    @property
    def shape(self):
        """
        R.shape -> tuple(width, height)
        
        Returns the width and height of the result matrix
        """
        return (self.width, self.height)

    def to_string(self):
        """R.to_string() -> str -- cram the current object into sting"""
        string = ""
        for row in self.__matrix:
            string += ", ".join(map(str, row)) + ";"
        return string

    @staticmethod
    def from_string(string):
        """R.from_string() -> MatrixResult"""
        return MatrixResult([list(map(int, row.split(','))) 
            for row in string.split(";")[:-1]])
