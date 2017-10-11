#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""conv.py:   Module containing the class implementations of the logic gates using
              Neural Networks"""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "6th September, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
* Module Description:
This module contains the class implementations of AND, OR, NOT and EXOR gates using the
NeuralNetwork Class.

Each of these classes inherits from the class NeuralNetwork and has the following methods:
__init__()     - Class Constructor
__call__()     - Generate output for specified input
-----------------------------------------------------------------------------"""
import time
import math
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------

from neural_network import NeuralNetwork

class AND(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of AND Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,1])
        layer = self.getLayer(0)
        self.network[0] = torch.from_numpy(np.array([20.0, 20.0, -30.0]))
        
    def __call__(self, in_1, in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor
#-------------------------------------------------------------------------------

class OR(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of AND Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,1])
        layer = self.getLayer(0)
        self.network[0] = torch.from_numpy(np.array([20.0, 20.0, -10.0]))
        
    def __call__(self, in_1, in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor
#-------------------------------------------------------------------------------

class NOT(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of NOT Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[1,1])
        layer = self.getLayer(0)
        self.network[0] = torch.from_numpy(np.array([-20.0, 10.0]))
        
    def __call__(self, in_1):
        input_tensor = torch.ones(1,1)
        input_tensor[0] = in_1
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor
#-------------------------------------------------------------------------------

class EXOR(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of EXOR Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,2,1])
        layer = []
        layer.append(self.getLayer(0))
        layer.append(self.getLayer(1))
        
        self.network[0] = torch.from_numpy(np.array([[ 20.0,-20.0],
                                                     [ 20.0,-20.0],
                                                     [-10.0, 30.0]]))
        self.network[1] = torch.from_numpy(np.array([20.0,20.0,-30.0]))
        
    def __call__(self, in_1,in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    ANDCurr = AND()
    print ANDCurr(False,False)
    print ANDCurr(False,True)
    print ANDCurr(True,False)
    print ANDCurr(True,True),'\n'
                
    ORCurr = OR()
    print ORCurr(False,False)
    print ORCurr(False,True)
    print ORCurr(True,False)
    print ORCurr(True,True),'\n'

    NOTCurr = NOT()
    print NOTCurr(False)
    print NOTCurr(True),'\n'

    EXORCurr = EXOR()
    print EXORCurr(False,False)
    print EXORCurr(False,True)
    print EXORCurr(True,False)
    print EXORCurr(True,True),'\n'
