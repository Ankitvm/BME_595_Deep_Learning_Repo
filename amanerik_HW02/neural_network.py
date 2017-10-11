#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""neural_network.py:   Module containing the class NeuralNetwork for generating  
                        an n-layer neural net for forward propagation pass."""

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
This module contains the class to generate a neural netwok with specified parameters.
The parameters that can be set using this class are:
NeuralNetwork(in_size_list)

where in_size_list = [in, h1, h2, ..., out]

in - size of input layer 
h1 - size of output layer
h2 - size of 2nd layer
hn - size of nth layer
out - size of output layer

The methods for this class are:
__init__()    - Class Constructor
getLayer()    - Display the NN layer of specified index
forward()     - Generate output for the NN

-----------------------------------------------------------------------------"""

import time
import math
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------

class NeuralNetwork(object):

    def __init__(self, in_size_list):
        """---------------------------------------------------------------------
        Desc.:  Class Constructor
        Args:   in_size_list - size of the layers of the NN
        Returns: - 
        ---------------------------------------------------------------------"""
        
        size_list = map(lambda x:x+1,in_size_list)
        size_list[-1] -= 1
        
        input_size  = size_list[0]
        output_size = size_list[-1]
        h_size      = [size_list[1:-1]]
        self.size_list = size_list

        self.network = dict()
        
        for i in range(len(size_list)-1):
            self.network[i]  = torch.rand(size_list[i],size_list[i+1]-1)
            self.network[i] -= 0.5*torch.ones(size_list[i],size_list[i+1]-1)
            self.network[i] *= (1/np.sqrt(size_list[i]*(size_list[i+1])))

#-------------------------------------------------------------------------------
                                         
    def getLayer(self,layer_no):
        """---------------------------------------------------------------------
        Desc.:  Display the nth layer of NN
        Args:   layer_no - index of NN layer
        Returns: NN layer of specified index
        ---------------------------------------------------------------------"""

        return self.network[layer_no]
#-------------------------------------------------------------------------------

    def forward(self,input_tensor):
        """---------------------------------------------------------------------
        Desc.:  generate the output of NN
        Args:   input_tensor - input for generating output
        Returns: output value generated
        ---------------------------------------------------------------------"""

        input_vect = input_tensor

        for k in range(len(self.network)):
            in_val = torch.ones(input_vect.shape[0]+1,
                                input_vect.shape[1])
            in_val[0:-1] = input_vect
            
            out_val = np.dot(self.network[k].numpy().T, in_val.numpy())
            input_vect  = torch.sigmoid(torch.from_numpy(out_val))
        
        return input_vect
#-------------------------------------------------------------------------------

if __name__ == '__main__':
    curr_nn = NeuralNetwork([2,2,1])
    print "Layer 0", curr_nn.getLayer(0)
    print "Layer 1", curr_nn.getLayer(1)
    print "Output", curr_nn.forward(torch.from_numpy(np.array([[1.0,1.0],
                                                               [1.0,0.0]])))
