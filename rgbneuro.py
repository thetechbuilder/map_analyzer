#Licence: Public Domain
#
#rgbneuro.py
#
"""Provides classes for setting a multi-network off.

Unit tests are in the current directory (test_rgbneuro.py)
"""
#standard modules:
from sys import path
from random import randint
from collections.abc import Sequence as _Sequence
from collections.abc import MutableSequence as _MuatableSeqence
#related modules:
from pygame import image, PixelArray, Surface, Color
#included (local) modules:
path.append("multinet")
path.append("multinet/nn")
from neuro import Network
from multinet import MultiNet
from imgresult import MatrixResult
from imgresult import ImageResultBase
from netfeeder import NetFeederBase
from netfeeder import PeriodManager
from imgselection import ISSerializer
from imgselection import ImageSelection
from imgsnipper import PixelParser
from imgsnipper import ImageSnipperBase
from imgsnipper import RectangularSnipper

__all__ = ["RandomFeeder"]

class RandomFeeder(NetFeederBase):
    """
    Class for feeding neural network
    """
    def __init__(self, network):
        NetFeederBase.__init__(self, network)
        self.__results = None
        self.__snipper = None

    def __load_snipper(self):
        self.__snipper = self.selection[
                randint(0, len(self.selection) - 1), parser]

    def pick(self):
        shp = self.snipper.shape
        x = randint(0, shp[0] - 1)
        y = randint(0, shp[1] - 1)
        result = [0]*3 #WEEEEEEEEEEEEEEEEEEEEEEEEEAAAAAaKKKKKKKKKKK
        result[self.__results[x, y]] = 1
        return self.__snipper.get_item(x, y), result

    def set_selection(self, snipper, results):
        if not isinstance(snipper, ImageSnipperBase):
            return TypeError("Wrong 'snipper' argument")
        if not isinstance(results, ImageResultBase):
            return TypeError("Wrong 'results' argument")
        self.__results = results
        self.__snipper = snipper

    @property
    def results(self):
        return self.__results

    @property
    def snipper(self):
        return self.__snipper


def set_new_multinet():
    print()
    learning_file = "maps/learning_map.bmp"
    result_file = "maps/learning_map_result.bmp"
    testing_map = "maps/test_map.bmp"
    testing_result = "test_result.bmp"
    NCATEG = 3
    BLOCK = (15, 15)
    number_inputs = BLOCK[0]*BLOCK[1]
    #obtaining results:
    surface = image.load(result_file)
    result_snipper = RectangularSnipper(surface, *BLOCK)
    result_shape = result_snipper.shape
    
    R = []
    pindx = BLOCK[0]*BLOCK[1]//2
    for y in range(result_shape[1]):
        row = []
        for x in range(result_shape[0]):
            b = PixelParser.blue(
                    result_snipper.get_item(x, y)[pindx])
            g = PixelParser.green(
                    result_snipper.get_item(x, y)[pindx])
            if b == 131:
                row.append(1)
            elif g == 71:
                row.append(2)
            else:
                row.append(0)
        R.append(row)
    result = MatrixResult(R)
    print("the result has been generated")

    selection = ImageSelection(*BLOCK)
    selection.append((learning_file, result))
    #network setting new:
    layers = [number_inputs, number_inputs//2, number_inputs//4, NCATEG]
    red_network = RandomFeeder(
            Network(*layers, n = 0.25))
    red_network_snipper = selection[0, lambda x: PixelParser.red(x)/100][1]
    red_network.set_selection(red_network_snipper, result)

    blue_network = RandomFeeder(
            Network(*layers, n = 0.25))
    blue_network_snipper = selection[0, lambda x: PixelParser.blue(x)/100][1]
    blue_network.set_selection(blue_network_snipper, result)

    green_network = RandomFeeder(
            Network(*layers, n = 0.25))
    green_network_snipper = selection[0, lambda x: PixelParser.green(x)/100][1]
    green_network.set_selection(green_network_snipper, result)

    unifier_network = Network(green_network.network.nouts + blue_network.network.nouts + green_network.network.nouts, NCATEG, n = 0.15)

    input("PRESS A BUTTON TO BEGIN LEARING PROCESS")
    #initialize multi network:
    mulnet = MultiNet(red_network, green_network, blue_network, unifier = unifier_network)
    print("MULNET SHAPES: RED:{} BLUE:{} GREEN:{} RESULT:{}".format(
        red_network_snipper.shape, blue_network_snipper.shape, 
        green_network_snipper.shape, result.shape))
    for x in range(10):
        mulnet.learn_netfeeds(PeriodManager, 10000)
        print("STAGE:", x)

    #learn netfeed
    sel = selection[0, lambda x: PixelParser.red(x)/100, 
            lambda x: PixelParser.green(x)/100, lambda x: PixelParser.blue(x)/100]
    print("SHAPES: RED:{} BLUE:{} GREEN:{} RESULT:{}".format(
        sel[1].shape, sel[2].shape, sel[3].shape, result.shape))

    for x in range(100000):
        if x%100 == 0:
            print("The period {} has been completed".format(x))
        rx = randint(0, result_shape[0] - 1)
        ry = randint(0, result_shape[1] - 1)
        r = [0]*3 #WEEEEEEEEEEEEEEEEEEEEEEEEEKKKKKKKKKKK
        r[result[rx, ry]] = 1
        mulnet.learn([sel[1].get_item(rx, ry), 
            sel[2].get_item(rx, ry), sel[3].get_item(rx, ry)], r)

    #learning is completed
    surface = Surface(result_shape)
    for y in range(result_shape[1]): 
        for x in range(result_shape[0]):
            outs = tuple(mulnet.activate([sel[1].get_item(x, y), 
                sel[2].get_item(x, y), sel[3].get_item(x, y)]))
            if outs[0] > outs[1] and outs[0] > outs[2]:
                color = Color(255,255,255)
            elif outs[1] > outs[0] and outs[1] > outs[2]:
                color = Color(0, 0, 131)
            else:
                color = Color(0, 70, 0)
            surface.set_at((x, y), color)
    print("SAVING RESULTS")
    image.save(surface, testing_result)
    
    sursur = image.load(testing_map)
    sursur_shape = surface.get_width(), surface.get_height()
    test_snipper_red = RectangularSnipper(sursur, *BLOCK, pix_parser = lambda x: PixelParser.red(x)/100)
    test_snipper_green = RectangularSnipper(sursur, *BLOCK, pix_parser = lambda x: PixelParser.green(x)/100)
    test_snipper_blue = RectangularSnipper(sursur, *BLOCK, pix_parser = lambda x: PixelParser.blue(x)/100)
    surface = Surface(sursur_shape)
    for y in range(test_snipper_red.shape[1]): 
        for x in range(test_snipper_red.shape[0]):
            outs = tuple(mulnet.activate([test_snipper_red.get_item(x, y), 
                test_snipper_green.get_item(x, y), test_snipper_blue.get_item(x, y)]))
            if outs[0] > outs[1] and outs[0] > outs[2]:
                color = Color(255,255,255)
            elif outs[1] > outs[0] and outs[1] > outs[2]:
                color = Color(0, 0, 131)
            else:
                color = Color(0, 70, 0)
            surface.set_at((x, y), color)
    print("SAVING RESULTS")
    image.save(surface, "pokemon.bmp")

if __name__ == '__main__':
    #set_network
    set_new_multinet()
