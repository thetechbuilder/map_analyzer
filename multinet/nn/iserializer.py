#Licence: Public Domain
#
#iserializer.py
#
"""Abstract base classes for serialization"""

#standard modules:
from abc import ABCMeta, abstractmethod

__all__ = ["ISerializer"]

class ISerializer(metaclass = ABCMeta):
    """Abstract base class for all implementations, serialization-wise"""
    
    @staticmethod
    @abstractmethod
    def write(source, target):
        """Serialize the source value into the target object"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def parse(target, outclass):
        """Parse the specified object, which has been serialized, into the 
        initial view"""
        raise NotImplementedError()
