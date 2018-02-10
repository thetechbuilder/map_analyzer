#License: Public Domain
#
#nnxs.py
#
"""Implements neural network XML-serializer using xml.etree.ElementTree

Unit tests are in the current directory (test_nnxs.py)
"""
#standard modules:
import xml.etree.ElementTree as ET
#internal modules:
from pyextra import Validators
from iserializer import ISerializer
from neuro import Network, NeuronView, Algorithms

__all__ = ["NNXSerializer"]

class NNXSerializer(ISerializer):
    """
    Provides a trivial API for serialization to and from standard XML
    It helps to parse/write your neural network (neuro.Network) object 
    from/to a specified file
    """
    #attributes:
    TAG_ROOT = "net"
    TAG_LAYER = "layer"
    TAG_NEURON = "neuron"
    ATTR_SPEED = "speed"
    ATTR_BIAS = "bias"
    ATTR_WEIGHTS = "weights"
    ATTR_ALG = "algorithm"

    @staticmethod
    def write(source, target):
        """
        NNXSerializer.write(source, target) -> None

        Writes your neural network to the specified file
        * "source" is an instance of "Network" or of a subclass thereof
        * "target" is either a file name or a binary file object
        """
        if not isinstance(source, Network):
            raise TypeError("The neural network to serialize must be "
                    "an instance of Network, or of a subclass thereof, "
                    "Not %s" % type(source))
        Validators.bfileobject(target)
        #root attributes countain overall parameters and layers
        root = ET.Element(NNXSerializer.TAG_ROOT, attrib = {
                    NNXSerializer.ATTR_SPEED : str(source.speed),
                    NNXSerializer.ATTR_ALG : source.algorithm.__name__})
        #SubElement function provides a way to create new sub-elements for
        #a given element.
        for layer in source._links: #traverse all layers
            sub = ET.SubElement(root, NNXSerializer.TAG_LAYER)
            for neuron in map(NeuronView, layer):
                ET.SubElement(sub, NNXSerializer.TAG_NEURON, 
                        attrib = {
                            NNXSerializer.ATTR_WEIGHTS: ", ".join(
                                map(str, neuron.weights)), 
                            NNXSerializer.ATTR_BIAS: str(neuron.bias)})
        #When encoding is US-ASCII or UTF-8 ET's output is binary!
        #Because the output is binary only BufferedIOBase-like objects 
        #are accepted.
        ET.ElementTree(root).write(target)

    @staticmethod
    def parse(target, outnetwork = Network):
        """
        NNXSerializer.parse(target) -> Network object

        Loads an external XML-preserved neural network into its "Network" 
        object
        * "target" is either a filename or a file object
        * "outnetwork" defines the output wrapper (output class)
        """
        if not issubclass(outnetwork, Network):
            raise TypeError("The argument 1 must be a class derived from "
                    "Network")
        Validators.bfileobject(target)
        root = ET.parse(target).getroot()
        L = [
                [
                    NeuronView.merge(
                        map(float, neuron.get(
                            NNXSerializer.ATTR_WEIGHTS).split(",")), 
                        float(neuron.get(NNXSerializer.ATTR_BIAS))) 
                    for neuron in layer]
                for layer in root]
        speed = float(root.get(NNXSerializer.ATTR_SPEED))
        alg = root.get(NNXSerializer.ATTR_ALG)
        if hasattr(Algorithms, alg):
            return Network(*L, n = speed, 
                    algorithm = getattr(Algorithms, alg)) 
        return outnetwork(*L, n = speed)
