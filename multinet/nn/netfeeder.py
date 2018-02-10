#Licence: Public Domain
#
#netfeeder.py
#
"""The netfeeder.py implements handling and learning extensions for 
neural networks"""
#standard modules:
from abc import ABCMeta, abstractmethod
#included modules:
from neuro import NetworkBase as _NetworkBase

__all__ = ["LearningManager", "PeriodManager", "NetFeederBase"]

class LearningManagerBase(metaclass = ABCMeta):
    """
    The abstract base class for learing managers. 
    The main purpose of learning managers is to stop running process and
    produce some palpable indication about how it has been done.
    """
    @abstractmethod
    def __enter__(self):
        """Enter the runtime context and return either this object or
        another object related to the runtime context"""
        raise NotImplementedError()

    @abstractmethod
    def __exit__(self):
        """Exit the runtime context"""
        raise NotImplementedError()

    @abstractmethod
    def dump(self):
        """Get additional information if it has been accumulated"""
        raise NotImplementedError()

    @abstractmethod
    def evaluate(self, results):
        """
        Evaluates results of network execution and returns either True 
        of False value whether the current process must be terminated.
        """
        raise NotImplementedError()

class PeriodManager(LearningManagerBase):
    """
    Represents a learning manager.
    """
    def __init__(self, periods):
        """
        pm.__init__(periods) -> class -- initialize period manager

        *'periods' - a number of executions
        """
        self.periods = periods
        self.__current = 0

    def __repr__(self):
        """
        pm.__repr__() <==> repr(pm)
        """
        return "{}(periods = {})".format(
                self.__class__.__name__, 
                self.periods)

    def __str__(self):
        """
        pm.__str__() <==> str(pm)
        """
        return "<{} periods={}, current={}>".format(
                self.__class__.__name__, self.periods, self.current)

    def __enter__(self):
        return self
        pass
    def __exit__(self, type, value, traceback):
        pass
    def evaluate(self, results):
        """
        pm.evaluate(results) -> bool -- estimate execution results
        
        Returning value indicates whether the current process must
        be terminated.
        """
        self.__current += 1
        return self.__current < self.__periods

    def dump(self):
        """
        pm.dump() -> count of executions, duration in seconds

        Produces an indication so that it is possible to get some extra
        information about learning process
        """
        return self.__current

    @property
    def periods(self):
        return self.__periods

    @periods.setter
    def periods(self, value):
        self.__periods = int(value)

    @property
    def current(self):
        """Current period"""
        return self.__current

class NetFeederBase(metaclass = ABCMeta):
    """
    The abstract base class for handling and learning extensions.
    
    Originally, this class has been aimed to describe further stream-like 
    implementations that provide the way of learning network when training
    set is not loaded in the memory completely.
    """
    def __init__(self, network):
        """f.__init__(network) -> NetFeederBase -- initialize feeder"""
        self.network = network

    def learn(self, manager = PeriodManager(1)):
        """
        f.learn() -> condition.dump()

        Executes series of learning rounds and return runtime dump that
        gets extra information about how computations have been perforemed.
        
        Notice that runtime dump is formed by the specified learning 
        manager which main purpose is to terminate the learning process 
        when it is needed and produce some palpable indication that tells
        you how it has been done.
        """
        if not isinstance(manager, LearningManagerBase):
            raise TypeError("'condition' expected to be derived from "
                    "LearningManagerBase")
        with manager as t:
            while t.evaluate(self.network.learn(*self.pick())):
                pass
        return t.dump()

    @abstractmethod
    def pick(self):
        """
        Pick an item up from the specified data set
        """
        raise NotImplementedError()

    @property
    def network(self):
        return self.__network

    @network.setter
    def network(self, value):
        if not isinstance(value, _NetworkBase):
            raise TypeError("Can't assign {}".format(type(value)))
        self.__network = value

