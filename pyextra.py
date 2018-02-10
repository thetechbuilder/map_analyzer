#Licence: Public Domain
#
#pyextra.py
#
"""Additional helping facilities

Unit tests are in the current directory (test_pyextra.py)"""

#standard modules:
import time
from os import path
from sys import stdout
#BufferedIOBase is a base class for binary streams that support some 
#kind of buffering. It inherits IOBase:
from io import BufferedIOBase

__all__ = ["Timer", "PrintHelper", "Validators"]

class Timer(object):
    """Implements timer running inside 'with' statement"""
    def __enter__(self):
        self.__start = time.time()
        self.__busy = True
    
    def __exit__(self, type, value, traceback):
        # Error handling here
        self.__finish = time.time()
        self.__busy = False
        
    def duration_in_seconds(self):
        return (time.time() if self.__busy 
                else self.__finish) - self.__start

class PrintHelper:
    """Helping class designed as a means to consolidating and providing
    auxiliary high-usage code for forming textual doohickeys"""
    class Progress(Timer):
        def update(self, progress):
            stdout.write("\r[{0:=3}%]".format(int(progress)))
            stdout.flush()

class Validators:
    """Helping class for handling input values"""
    #This class has been designed to eliminate repeated code
    
    #In many implementations, sequence-wise, handling of indices runs by 
    #the same way; consequently, I wrote a snippet of the high-usage 
    #code for validation:
    @staticmethod
    def arrayindex(index, length):
        """
        SequenceHelper.handleindex(index, length) -> verified index

        * inverses netagive indices so that 'index = (index + length)'
        * raises IndexError if the specified index is out of range
        * raises TypeError if the specified index does not implement 
        integer class(int)
        """
        if isinstance(index, int):
            if index < 0: #handle negative indices
                index += length
            if index < 0 or not index < length:
                raise IndexError("The specified index "
                        "{} is out of range [{},{}]".format(
                            index, 0, length))
        else:
            raise TypeError(
                    'invalid type of the specified index, indices must be '
                    'integers, not {} ({} given)'.format(
                        type(index), index))
        return index

    #My XML-serializing code ofthen includes this checking
    @staticmethod
    def bfileobject(target):
        """
        Validates binary file object.

        * raises FileNotFoundError if the specified file directory does
        not exist
        * raises TypeError if the specified target object is not a file 
        name or a binary stream (BufferedIOBase)
        """
        if isinstance(target, str):
            target = path.split(target)
            if target[0] and not path.isdir(target[0]):
                raise FileNotFoundError(
                        "The specified directory does not exist.")
            target = path.join(*target)
        elif not isinstance(target, BufferedIOBase):
            raise TypeError("The type of the target value is wrong. "
                    "'target' must be either a filename string or a "
                    "binary file object, not {}".format(type(target)))
