#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""nn_img2num.py:   Module containing the class implementations of the MNIST classifiers
                   using Neural Networks (with pytorch nn)"""

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
This module contains the class implementations of MNIST Classifier using pytorch nn module.

Each of these classes inherits from the class NeuralNetwork and has the following methods:
__init__()     - Class Constructor
__call__()     - Generate output for specified input
train()        - Train the NN
-----------------------------------------------------------------------------"""

import torch
import torchvision
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#------------------------------------------------------------------------------
class NNImg2Num(nn.Module):
    """---------------------------------------------------------------------
    Desc.:  Class for implementing a Neural network to classify digits based
            on the MNIST Dataset. The NN is built using the pytorch nn module
            and contains methods for training and testing the classifier.
            
    Attributes:   transform   - pytorch object to transform images
                  trainset    - MNIST training dataset
                  trainloader - MNIST training dataset loader     
                  testset     - MNIST testing dataset
                  testloader  - MNIST testing dataset loader
    Returns: -    __init__()  - constructor
                  forward()   - generate output
                  train()     - train NN classifier
    ---------------------------------------------------------------------"""

## Attributes -----------------------------------------------------------------
    # Transform to Normalize the MNIST images 
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.1307,), (0.3081,)) ])
    
    trainset    =  torchvision.datasets.MNIST(root='./data',
                                              train=True,
                                              download=True,
                                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              num_workers=1,
                                              batch_size = 64,
                                              shuffle=True)

    testset     = torchvision.datasets.MNIST( root='./data',
                                              train=False,
                                              download=True,
                                              transform=transform)

    testloader  = torch.utils.data.DataLoader(testset,
                                              num_workers=1,
                                              batch_size = 1000,
                                              shuffle=True)

## Methods ---------------------------------------------------------------------
    
    def __init__(self):
        """---------------------------------------------------------------------
        Desc.:  Class Constructor
        Args:   -
        Returns: - 
        ---------------------------------------------------------------------"""

        super(NNImg2Num, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        print "----------------------------------------------------------------"
        print "Digit Classfier using pytorch nn module"
        print "Author: Ankit Manerikar"
        print "Written on: 09-21-2017"
        print "----------------------------------------------------------------"
        print "Class initialized"
#------------------------------------------------------------------------------

    def forward(self, x):
        """---------------------------------------------------------------------
        Desc.:  function for forward propagation pass (generates output for an
                input tensor x)
        Args:   x - input tensor for forward propagation
        Returns: - 
        ---------------------------------------------------------------------"""

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
#------------------------------------------------------------------------------

    def train(self):
        """---------------------------------------------------------------------
        Desc.:   Method for training the NN using an SGD optimizer and MSE Cross
                 Entropy Criterion  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""
        print "\nTraining Neural network using the MNIST Dataset ..."
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        print "\nEpoch-wise Performance:"
        
        for epoch in range(2):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(self.trainloader):

                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.__call__(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]
                if i % 200 == 199:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('MNIST classifier trained ...')
#------------------------------------------------------------------------------
# Class Ends
#------------------------------------------------------------------------------

###############################################################################        
if __name__ == '__main__':
    
    CurrNN = NNImg2Num()
    CurrNN.train()
