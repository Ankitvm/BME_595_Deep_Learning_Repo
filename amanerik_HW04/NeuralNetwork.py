#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""neural_network.py:   Module containing the class NeuralNetwork for generating  
                        an n-layer neural net for forward and backward propagation
                        pass."""

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
This module contains the class to generate a neural network with specified parameters.
The parameters that can be set using this class are:
NeuralNetwork(in_size_list)

where in_size_list = [in, h1, h2, ..., out]

in - size of input layer 
h1 - size of output layer
h2 - size of 2nd layer
hn - size of nth layer
out - size of output layer

The methods for this class are:
__init__()     - Class Constructor
getLayer()     - Display the NN layer of specified index
forward()      - Generate output for the NN
backward()     - Generate backward propagation pass for the NN
updateParams() - Update weights for NN 
-----------------------------------------------------------------------------"""

import time
import math
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------

class NeuralNetwork(object):

    def __init__(self, size_list):
        """---------------------------------------------------------------------
        Desc.:  Class Constructor
        Args:   in_size_list - size of the layers of the NN
        Returns: - 
        ---------------------------------------------------------------------"""
        
##        size_list = map(lambda x:x+1,in_size_list)
##        size_list[-1] -= 1
        
        input_size  = size_list[0]
        output_size = size_list[-1]
        h_size      = [size_list[1:-1]]
        self.size_list = size_list

        self.theta = []
        
        for i in range(len(size_list)-1):
            size_1 = size_list[i]+1
            size_2 = size_list[i+1]
                
            theta  = torch.rand(size_1,size_2)
            
            theta -= 0.5*torch.ones(size_1,size_2)
            theta *= (1/np.sqrt(size_1))
            self.theta.append(theta)

#-------------------------------------------------------------------------------
                                         
    def getLayer(self,layer_no):
        """---------------------------------------------------------------------
        Desc.:  Display the nth layer of NN
        Args:   layer_no - index of NN layer
        Returns: NN layer of specified index
        ---------------------------------------------------------------------"""

        return self.theta[layer_no]
#-------------------------------------------------------------------------------

    def forward(self,input_tensor):
        """---------------------------------------------------------------------
        Desc.:  generate the output of NN
        Args:   input_tensor - input for generating output
        Returns: output value generated
        ---------------------------------------------------------------------"""
        self.a = []
        self.z = []
        self.ahat = []
        input_vect = input_tensor

        for k in range(len(self.theta)):
            in_val = torch.ones(input_vect.shape[0]+1,
                                input_vect.shape[1])
            in_val[1:] = input_vect
            
            self.a.append(input_vect)
            self.ahat.append(in_val)
            
            out_val = np.dot(self.theta[k].numpy().T, in_val.numpy())
            self.z.append(torch.from_numpy(out_val))
            input_vect  = torch.sigmoid(torch.from_numpy(out_val))

        self.a.append(input_vect)
        
        return input_vect
#-------------------------------------------------------------------------------

    def backward(self, target):
        """---------------------------------------------------------------------
        Desc.:  perform back-propagation pass
        Args:   target - target output value 
        Returns: -
        ---------------------------------------------------------------------"""

        self.dE_dtheta = []
        
        delta = (self.a[-1] - target)* (self.a[-1]* (torch.ones(self.a[-1].shape) - self.a[-1]))
        init_flag = True
        for curr_a, curr_ahat, curr_z, curr_theta in zip(self.a[-2::-1],
                                                         self.ahat[::-1],
                                                         self.z[::-1],
                                                         self.theta[::-1]):    
            if init_flag:
                curr_dtheta = torch.mm(curr_ahat, delta.transpose(0,1))
                m1 = torch.mm(curr_theta,delta)
                m2 = curr_ahat*(1-curr_ahat)
                init_flag = False
                
            else:
                curr_dtheta = torch.mm(curr_ahat, delta[1:].transpose(0,1))
                m1 = torch.mm(curr_theta,delta[1:])
                m2 = curr_ahat*(1-curr_ahat)

            self.dE_dtheta.append(curr_dtheta)
            delta = torch.zeros(m1.shape[0],m1.shape[1])
            for i in range(m1.shape[1]):
                delta[:,i] = m1[:,i:i+1]*m2

        self.dE_dtheta = list(reversed(self.dE_dtheta))
        
#-------------------------------------------------------------------------------

    def updateParams(self, eta):
        """---------------------------------------------------------------------
        Desc.:  update Theta matrix after back-propagation pass
        Args:   eta - learning parameter
        Returns: -
        ---------------------------------------------------------------------"""
        
        for i in range(len(self.theta)):
            self.theta[i] = self.theta[i] - eta*torch.mul(self.dE_dtheta[i],eta)

#-------------------------------------------------------------------------------
            
if __name__ == '__main__':
    curr_nn = NeuralNetwork([2,2,1])
    print "Layer 0", curr_nn.getLayer(0)
    print "Layer 1", curr_nn.getLayer(1)
    print "\n Forward Pass:"
    in_tensor = torch.ones(2,1)
    in_tensor[0] -= 1
    print "Input:", in_tensor
    print "Output", curr_nn.forward(in_tensor)
    print '\nBackward:\t'
    
    curr_nn.backward(torch.ones(1,1))
    print "Theta:\t", curr_nn.theta
    print "dTheta:\t", curr_nn.dE_dtheta
    
    curr_nn.updateParams(0.1)
    print "\nUpdated Theta:\t", curr_nn.theta

    
    
    
