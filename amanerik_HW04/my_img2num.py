#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""my_img2um.py:   Module containing the class implementations of the MNIST classifiers
                   using Neural Networks"""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "21st September, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
* Module Description:
This module contains the class implementations of MNIST Classifier using the
NeuralNetwork Class.

Each of these classes inherits from the class NeuralNetwork andhas the following
methods:

__init__()     - Class Constructor
forward()      - Generate output for specified input
train()        - Train the NN
-----------------------------------------------------------------------------"""
import time
import math
import random
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.datasets.mnist import MNIST, read_image_file, read_label_file

#-------------------------------------------------------------------------------

from NeuralNetwork import NeuralNetwork

class MyImg2Num(NeuralNetwork):
    """---------------------------------------------------------------------
    Desc.:  Class for implementing a Neural network to classify digits based
            on the MNIST Dataset. The NN is built using the NeuralNetworkClass
            and contains methods for training and testing the classifier.
            
    Returns: -    __init__()  - constructor
                  forward()   - generate output
                  train()     - train NN classifier
    ---------------------------------------------------------------------"""

    def __init__(self):
        """---------------------------------------------------------------------
        Desc.:   Class Constructor  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""

        NeuralNetwork.__init__(self,[28*28,10*10,10])

        print "----------------------------------------------------------------"
        print "Digit Classfier using pytorch nn module"
        print "Author: Ankit Manerikar"
        print "Written on: 09-21-2017"
        print "----------------------------------------------------------------"
        print "Loading MNIST Dataset ..."
        self.train_images = read_image_file('./data/raw/train-images-idx3-ubyte')
        self.target_val = read_label_file('./data/raw/train-labels-idx1-ubyte')
        print "Dataset Loaded"
        print "\nClass initialized"
        
#-------------------------------------------------------------------------------
        
    def forward(self, img):
        """---------------------------------------------------------------------
        Desc.:  function for forward propagation pass (generates output for an
                input tensor x)
        Args:   img - input tensor for forward propagation
        Returns: - 
        ---------------------------------------------------------------------"""

        input_tensor = img.view(28*28,1)
        input_tensor = input_tensor.type(torch.FloatTensor)
        output_tensor = NeuralNetwork.forward(self, input_tensor)

        o_value = np.argmax(output_tensor.numpy())
        
        return output_tensor.numpy()
#-------------------------------------------------------------------------------
        
    def train(self):
        """---------------------------------------------------------------------
        Desc.:   Method for training the NN using an SGD optimizer and MSE Cross
                 Entropy Criterion  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""
        print "\nTraining Neural network using the MNIST Dataset ..."
        print "\nEpoch-wise Performance:"

        err_val = np.zeros(10)
        batch_ctr = 0
        ctr = 10
        mse_val = 10
        
        while mse_val > 0.4:
            err_val = np.zeros(10)
            for k in range(batch_ctr, batch_ctr+10):
                o_val = self.forward(self.train_images[k])
                target_vect = np.zeros(10)
                target_vect[self.target_val[k]] = 1.
                bkward_vect = torch.rand(10,1)
                bkward_vect[:,0] = torch.from_numpy(target_vect).type(torch.FloatTensor)
                self.backward(bkward_vect)
                o_val = self.forward(self.train_images[k])
                self.updateParams(0.05)
                err_val = err_val + (o_val[0]-target_vect)
            
            mse_val = sum(err_val)/10
            print "Epoch:\t", batch_ctr, "\tError:\t", mse_val
            
            if batch_ctr == self.train_images.shape[0]/10:
                batch_ctr = 0
            else:
                batch_ctr += 1
            ctr += 10

        print 'MNIST CLassifier Trained ...'
#-------------------------------------------------------------------------------
# Class Ends
#-------------------------------------------------------------------------------

################################################################################
if __name__ == "__main__":
    CurrNN = MyImg2Num()
    CurrNN.forward(CurrNN.train_images[0])
    CurrNN.train()
