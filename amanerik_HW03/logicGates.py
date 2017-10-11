#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""conv.py:   Module containing the class implementations of the logic gates using
              Neural Networks"""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "14th September, 2017"
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
train()        - Train the NN
-----------------------------------------------------------------------------"""
import time
import math
import random
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------

from NeuralNetwork import NeuralNetwork

class AND(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of AND Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,1])
        self.truth_table = [[False, False],
                       [False, True],
                       [True, False],
                       [True, True]]        
        self.target_val = [x[0] and x[1] for x in self.truth_table]
        
    def __call__(self, in_1, in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor.numpy()[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor
    
    def train(self):
        err_val = 1
        while err_val > 0:
            err_val = 0
            for k in range(4):
                self.__call__(self.truth_table[k][0],self.truth_table[k][1])
                self.backward(self.target_val[k])
                self.updateParams(0.05)
                err_val += abs(self.target_val[k] - self.__call__(self.truth_table[k][0],self.truth_table[k][1])[0])
#-------------------------------------------------------------------------------

class OR(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of AND Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,1])
        self.truth_table = [[False, False],
                       [False, True],
                       [True, False],
                       [True, True]]        
        self.target_val = [x[0] or x[1] for x in self.truth_table]

        
    def __call__(self, in_1, in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor.numpy()[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor

    def train(self):
        err_val = 1
        while err_val > 0.1:
            err_val = 0
            for k in range(4):
                self.__call__(self.truth_table[k][0],self.truth_table[k][1])
                self.backward(self.target_val[k])
                self.updateParams(0.01)
                err_val += abs(self.target_val[k] - self.__call__(self.truth_table[k][0],self.truth_table[k][1])[0])

#-------------------------------------------------------------------------------

class NOT(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of NOT Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[1,1])
        self.truth_table = [False, True]        
        self.target_val = [not x for x in self.truth_table]
        
    def __call__(self, in_1):
        input_tensor = torch.ones(1,1)
        input_tensor[0] = in_1
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor.numpy()[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor

    def train(self):
        err_val = 1
        while err_val > 0:
            err_val = 0
            for k in range(2):
                self.__call__(self.truth_table[k])
                self.backward(self.target_val[k])
                self.updateParams(0.01)
                err_val += abs(self.target_val[k] - self.__call__(self.truth_table[k])[0])

#-------------------------------------------------------------------------------

class EXOR(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class Implementation of EXOR Gate
    Args:   -
    Returns: - 
    ---------------------------------------------------------------------"""

    def __init__(self):
        NeuralNetwork.__init__(self,[2,2,1])
        self.truth_table = [[False, False],
                       [False, True],
                       [True, False],
                       [True, True]]        
        self.target_val = [x[0] ^ x[1] for x in self.truth_table]
        
    def __call__(self, in_1,in_2):
        input_tensor = torch.ones(2,1)
        input_tensor[0] = in_1
        input_tensor[1] = in_2
        output_tensor = self.forward(input_tensor)

        o_tensor = []
        
        for k in range(output_tensor.numpy().shape[0]):
            if output_tensor.numpy()[k] > 0.5:
                o_tensor.append(True)
            else:
                o_tensor.append(False)

        return o_tensor

    def train(self):
        err_val = 1
        while err_val > 0:
            err_val = 0
            for k in range(4):
                self.__call__(self.truth_table[k][0],self.truth_table[k][1])
                self.backward(self.target_val[k])
                self.updateParams(0.1)
                err_val += abs(self.target_val[k] - self.__call__(self.truth_table[k][0],self.truth_table[k][1])[0])

#-------------------------------------------------------------------------------

if __name__ == "__main__":

    print "AND Gate:"
    print "Before Training"
    ANDCurr = AND()
    print ANDCurr(False,False)
    print ANDCurr(False,True)
    print ANDCurr(True,False)
    print ANDCurr(True,True),'\n'

    ANDCurr.train()

    print "After Training"
    print ANDCurr(False,False)
    print ANDCurr(False,True)
    print ANDCurr(True,False)
    print ANDCurr(True,True),'\n'

    print "Trained weights:\n"
    print ANDCurr.theta,'\n\n'

    print "OR Gate:"
    print "Before Training"                
    ORCurr = OR()
    print ORCurr(False,False)
    print ORCurr(False,True)
    print ORCurr(True,False)
    print ORCurr(True,True),'\n'

    ORCurr.train()
    print "After Training"

    print ORCurr(False,False)
    print ORCurr(False,True)
    print ORCurr(True,False)
    print ORCurr(True,True),'\n'

    print "Trained weights:\n"
    print ORCurr.theta,'\n\n'


    print "NOT Gate:"
    print "Before Training"
    NOTCurr = NOT()
    print NOTCurr(False)
    print NOTCurr(True),'\n'

    NOTCurr.train()
    print "After Training"
    
    print NOTCurr(False)
    print NOTCurr(True),'\n'

    print "Trained weights:\n"
    print NOTCurr.theta,'\n\n'

    print "EXOR Gate:"
    print "Before Training"
    EXORCurr = EXOR()
    print EXORCurr(False,False)
    print EXORCurr(False,True)
    print EXORCurr(True,False)
    print EXORCurr(True,True),'\n'

    EXORCurr.train()
    print "After Training"

    print EXORCurr(False,False)
    print EXORCurr(False,True)
    print EXORCurr(True,False)
    print EXORCurr(True,True),'\n'
    
    print "Trained weights:\n"
    print EXORCurr.theta,'\n\n'
