#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""train.py:   Module containing classes for using a pre-trained AlexNet to train
               on the Tiny ImageNet dataset."""

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "16th October, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
* Module Description:
This module contains the implementation to train the last layer of a pre-
trained AlexNet on the tiny ImageNet dataset. 
-----------------------------------------------------------------------------"""

import time, os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import cv2

import torch

import torchvision
import torchvision.transforms as transforms
import torchvision.models     as models
import torchvision.datasets   as datasets

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

#------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='PyTorch AlexNet Training')

parser.add_argument('--data', metavar='DIR', default='./tiny-imagenet-200/',
                    help='path to Tiny ImageNet dataset')
parser.add_argument('--save', metavar='DIR', default='./result_model/'     ,
                    help='path to directory where the model is to be saved')


if __name__ == "__main__":

    print '===================================================================='
    print 'Transfer Learning: AlexNet for Tiny ImageNet Dataset'
    print '====================================================================\n'

    print '\nProgram Start Time : ', time.strftime('%m-%d-%Y %H:%M:%S', time.localtime())

    global args, best_prec1
    args = parser.parse_args()

    print "\nLoading Tiny ImageNet Dataset from the directory, ", args.data,"..."

    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')

    # Load Datasets--------------------------------------------- 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([ transforms.RandomSizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           normalize,])

    test_transform = transforms.Compose([  transforms.Scale(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(),
                                           normalize,])
    
    train_loader = torch.utils.data.DataLoader( datasets.ImageFolder(traindir, train_transform),
                                                batch_size=50,
                                                shuffle=True,
                                                pin_memory=True)

    val_loader   = torch.utils.data.DataLoader( datasets.ImageFolder(valdir, test_transform),
                                                batch_size=50,
                                                shuffle=False,
                                                pin_memory=True)
    #-----------------------------------------------------------
    
    print "Loading AlexNet ..."
    model = models.alexnet(pretrained=True)
    model.classifier._modules['6'] = nn.Linear(4096, 1000)
    print "Dataset and DeepNet Loaded ...\n"
    
    print "\nTraining Last Layer of AlexNet using the Tiny ImageNet Dataset ..."
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.classifier._modules['6'].parameters(), lr=0.001, momentum=0.9)
    print "\nEpoch-wise Performance:"
    
    start_time = time.time()    
    for epoch in range(1):  # loop over the dataset for 1 epoch

        running_loss = 0.0
        for i, data in enumerate(train_loader):

            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)
            
            # zero the parameter gradients
            optimizer.zero_grad()
        
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 50 == 49:    # print every 200 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 50))
                running_loss = 0.0
    end_time = time.time()

    print "Classifier Trained ... "
    print "-----------------------------------"
    print "\n Testing Classifier with the validation dataset ..."
    correct = 0
    loss = 0
    ctr = 0

    for data, target in val_loader:
        ctr +=1 
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        loss += F.nll_loss(output, target, size_average=False).data[0] 
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss /= len(val_loader.dataset)
    
    print('\nConvolutional Neural Network:')
    print('Test Accuracy = {:.3f}%'.format(10.*correct))
    print 'Time Taken for Training: %.3f s'%(end_time-start_time)    
    
    model_file_name = os.path.join(args.save, 'model.pyc')
    torch.save(model,model_file_name)
    print "\nClassifier saved as: ", model_file_name
    print "======================================================================="
