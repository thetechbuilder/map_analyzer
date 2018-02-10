#Licence: Public Domain
#
#multinet.py
#
"""Implements a multi-network solver which allows you to combine a number of
neural networks into the single solver.

Unit tests are in the current directory (test_multinet.py)
"""
#standard modules:
from sys import path
from itertools import chain
from abc import ABCMeta, abstractmethod
#included modules:
path.append("nn")
from neuro import Network
from neuro import NetworkBase as _NetworkBase
from netfeeder import NetFeederBase as _NetFeederBase
from netfeeder import LearningManagerBase as _LearningManagerBase

__all__ = ["MultiNetBase", "MultiNet"]

class MultiNetBase(_NetworkBase):
    """
    The base class for all neural multi-network realizations.
    """
    def __init__(self, *feednets, unifier):
        """
        MN.__init__(*feednets) -> MultiNet

        Constructs a multi-network from
         - sequence of any objects implementing netfeeder.NetFeederBase
         - iterable yielding object implementing netfeeder.NetFeederBase
        """
        feednets = tuple(feednets)
        if not all(isinstance(f, _NetFeederBase) 
                for f in feednets):
            raise TypeError(
                    "Can't construct multi-network from these objects")
        
        #check unifier
        if not isinstance(unifier, _NetworkBase):
            raise TypeError("Unifier must implement _NetworkBase")
        else:
            #desired number of inputs:
            nouts = sum(x.network.nouts for x in feednets) 
            if unifier.ninps != nouts:
                raise ValueError("Unifier doesn't match inputs of the "
                        "specified networks. {} != {}".format(
                            unifier.ninps, ninps))
        self.__feednets = feednets
        self.__unifier = unifier

    def activate(self, ins):
        """MN.activate(ins) -> map object -- activate network"""
        return self.unifier.activate(
                tuple(chain(*self._activate_networks(ins))))

    def learn(self, selection, target):
        """
        MN.learn(selection, target)

        Arguments:
        1) 'selection' is a lerning selection (sequence)
        2) 'target' is a desired value of input nodes (sequence)
        """
        self._learn_networks(selection, target)
        #Acivates assistant networks and learn unificatory network
        return self.unifier.learn(
                tuple(chain(*self._activate_networks(selection))), target)

    @abstractmethod
    def learn_netfeeds(manager):
        raise NotImplementedError()
    
    @abstractmethod
    def _activate_networks(self, ins):
        raise NotImplementedError()

    @abstractmethod
    def _learn_networks(self, selection, target):
        raise NotImplementedError()
    
    @property
    def feednets(self):
        """MN.networks -> tuple -- separated networks"""
        return self.__feednets
    
    @property
    def unifier(self):
        """MN.unifier -> Network -- unificatory network"""
        return self.__unifier

    @property
    def shape(self):
        """MN.shape -> tuple -- shape of created network"""
        return tuple(len(nets), 1)

    @property
    def size(self):
        """MN.size -> int -- number of networks"""
        return sum(self.shape)

    @property
    def ninps(self):
        """MN.ninps -> int -- number of inputs"""
        return sum(x.ninps for x in self.networks)

    @property
    def nouts(self):
        """MN.nouts -> int -- number of outputs"""
        return self.unifier.nouts

class MultiNet(MultiNetBase):
    """Implements a multi-netwok"""
    
    def learn_netfeeds(self, manager, *params):
        for net in self.feednets:
            print("NET FEED LEARNING:{}".format(net))
            res = net.learn(manager(*params))
    
    #protected methods:
    def _activate_networks(self, ins):
        return (net.network.activate(i) 
                for net, i in zip(self.feednets, ins))

    def _learn_networks(self, selection, target):
        for net, slc in zip(self.feednets, selection):
            net.network.learn(slc, target)

