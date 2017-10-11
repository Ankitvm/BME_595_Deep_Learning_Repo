#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""img2num.py:   Module containing classes for comparing the CIFAR100 classifiers
                 using Convolutional Neural Network """

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "24st September, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
* Module Description:
This module contains the class implementation of a CIFAR100 Classifier
LeNet5 neural network.

__init__()     - Class Constructor
__call__()     - Generate output for specified input
train()        - Train the NN
view()         - View the current test input and its label
-----------------------------------------------------------------------------"""

import time
import torch
import torchvision
import cv2
import argparse
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#------------------------------------------------------------------------------

class Img2Num(nn.Module):
    """---------------------------------------------------------------------
    Desc.:  Class for implementing a CONVOLUTIONAL Neural Network to classify digits
            based on the CIFAR100 Dataset. The NN is built using the pytorch nn
            module and contains methods for training and testing the classifier.

    Attributes:   transform   - pytorch object to transform images
                  trainset    - CIFAR100 training dataset
                  trainloader - CIFAR100 training dataset loader     
                  testset     - CIFAR100 testing dataset
                  testloader  - CIFAR100 testing dataset loader
    Returns: -    __init__()  - constructor
                  forward()   - generate output
                  train()     - train NN classifier
    ---------------------------------------------------------------------"""

## Attributes -----------------------------------------------------------------
    # Transform to Normalize the CIFAR100 images 
    transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ])

    # objects for training and test datasets
    trainset    =  torchvision.datasets.CIFAR100(root='./CIFAR100_dataset',
                                              train=True,
                                              download=True,
                                              transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset,
                                              num_workers=2,
                                              batch_size = 50,
                                              shuffle=True)

    testset     = torchvision.datasets.CIFAR100( root='./CIFAR100_dataset',
                                              train=False,
                                              download=True,
                                              transform=transform)

    testloader  = torch.utils.data.DataLoader(testset,
                                              num_workers=2,
                                              batch_size = 1000,
                                              shuffle=True)
    
## Methods ---------------------------------------------------------------------
    
    def __init__(self):
        """---------------------------------------------------------------------
        Desc.:  Class Constructor
        Args:   -
        Returns: - 
        ---------------------------------------------------------------------"""

        super(Img2Num, self).__init__()

        # LeNet5 layers
        self.conv1 = nn.Conv2d(3,  16, 5) # 1st convolution layer
        self.conv2 = nn.Conv2d(16, 10, 5) # 2nd convolution layer
        self.conv2_drop = nn.Dropout2d()
        
        self.fc1 = nn.Linear(250, 500)    
        self.fc2 = nn.Linear(500, 200)
        self.fc3 = nn.Linear(200, 100)
        
        print "----------------------------------------------------------------"
        print "LeNet-5 Neural Network using pytorch nn module"
        print "Author: Ankit Manerikar"
        print "Written on: 09-24-2017"
        print "----------------------------------------------------------------"
        print "Class initialized"
#------------------------------------------------------------------------------

    def forward(self, x):
        """---------------------------------------------------------------------
        Desc.:  Function for forward propagation pass (generates output for an
                input tensor x)
        Args:   x - input tensor for forward propagation
        Returns: - 
        ---------------------------------------------------------------------"""

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 250)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, training=self.training)
        x = self.fc3(x)
        
        return F.log_softmax(x)
#------------------------------------------------------------------------------

    def train(self):
        """---------------------------------------------------------------------
        Desc.:   Method for training the NN using an SGD optimizer and MSE Cross
                 Entropy Criterion  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""
        print "\nTraining Neural network using the CIFAR100 Dataset ..."
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=0.02, momentum=0.9)
        print "\nEpoch-wise Performance:"
        
        for epoch in range(50):  # loop over the dataset for 50 epochs

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
                if i % 200 == 199:    # print every 200 mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / 200))
                    running_loss = 0.0

        print('CIFAR100 classifier trained ...')
#------------------------------------------------------------------------------

    def view(self, img):
        """---------------------------------------------------------------------
        Desc.:   View the current test inp-ut and its predicted as well as ground truth
                 label  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""
        img = img / 2 + 0.5
        npimg = img.numpy()
        plt.figure(1)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
#-------------------------------------------------------------------------------

    def cam(self,idx=0):
        """---------------------------------------------------------------------
        Desc.:   Continuously produce labels for a camera feed  
        Args:    -
        Returns: - 
        ---------------------------------------------------------------------"""        
        cam = cv2.VideoCapture(idx)

        while(True):
            if not cam.isOpened():
                print 'Cannot Connect to Camera! Please check the interface and interface drivers ...'
                return None
        
            ret, frame = cam.read()            
            curr_image = cv2.resize(frame, (32,32))
            curr_tensor = torch.Tensor(1,3,32,32)
            
            curr_tensor[0] = torch.from_numpy(curr_image)
            output = self.__call__(Variable(curr_tensor,volatile=True))
            cv2.imshow('Current Camera Frame', frame)
                

            print 'Frame Time:', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime()), \
                  '\tIdentified Object Label:\t', output.data.max(1, keepdim=True)[1].numpy()
            time.sleep(0.1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()        
#------------------------------------------------------------------------------
# Class Ends
#------------------------------------------------------------------------------

###############################################################################        
if __name__ == '__main__':

    print '===================================================================='
    print 'Convolutional Neural Network for CIFAR100 Dataset'
    print '====================================================================\n'

    print '\nProgram Start Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())

    CurrCNN = Img2Num()
    start_time = time.time()
    CurrCNN.train()
    end_time = time.time()
    
##-------------------------------------------------------------------------------
    print '\nTesting Neural Network for CIFAR100 Dataset...'
    correct = 0
    loss = 0
    for data, target in CurrCNN.testloader:

        data, target = Variable(data, volatile=True), Variable(target)
        output = CurrCNN(data)
        loss += F.nll_loss(output, target, size_average=False).data[0] 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(CurrCNN.testloader.dataset)

    print('\nConvolutional Neural Network:')
    print('Test Accuracy = {:.3f}%'.format(100. * correct / len(CurrCNN.testloader.dataset)))
    print('Loss: %.3f'%loss)
    print 'Time Taken for Training: %.3f s'%(end_time-start_time)

#--------------------------------------------------------------------------------
    print "\nOperational Example:"
    dataiter = iter(CurrCNN.testloader)
    images,labels = dataiter.next()
    pred = CurrCNN(Variable(images, volatile=True))
    label_no = input('Enter Label Number to test:\t')
    print 'Ground truth: \t', labels[label_no]
    print 'Predicted Label:\t', pred.data.max(1, keepdim=True)[1].numpy()[label_no]
    print 'Displaying Image (Close figure for execution of Part B)... '
    CurrCNN.view(images[label_no])
#--------------------------------------------------------------------------------
    
    print '\n--------------------------------------------------------------------'
    print 'Real-time Visual Classification:'
    print 'Capturing live feed from camera ...'
    print 'Press Ctrl+C to halt program ...\n'
    time.sleep(5)
    CurrCNN.cam()
#================================================================================
