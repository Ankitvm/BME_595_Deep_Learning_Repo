#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""train.py:   Module containing classes for using a pre-trained AlexNet to predict labels
               coming from a live camera feed."""

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

parser = argparse.ArgumentParser(description='PyTorch AlexNet Testing')
parser.add_argument('--data', metavar='DIR', default='./tiny-imagenet-200/',
                    help='path to Tiny ImageNet dataset')
parser.add_argument('--model', metavar='DIR', default='./result_model/' ,
                    help='path to directory where the model is to be loaded')

def alexnet_camfeed(model,idx=0):
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
        curr_image = cv2.resize(frame, (224,224))
        
        curr_tensor = torch.Tensor(1,3,224,224)        
        curr_tensor[0] = torch.from_numpy(curr_image)
        output = model(Variable(curr_tensor,volatile=True))

        image_text = "Label: ",output.data.max(1, keepdim=True)[1].numpy()
        cv2.putText(frame,'OpenCV',(10,200), font, 4,(255,255,255),2,cv2.LINE_AA)        
        cv2.imshow('Current Camera Frame', frame)
            

        print 'Frame Time:', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime()), \
              '\tIdentified Object Label:\t', output.data.max(1, keepdim=True)[1].numpy()
        
        time.sleep(0.1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":

    print '===================================================================='
    print 'Transfer Learning: AlexNet for Tiny ImageNet Dataset'
    print '====================================================================\n'

    print '\nProgram Start Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())

    global args, best_prec1
    args = parser.parse_args()

    print "\nLoading AlexNet from the directory, ", args.model,"..."

    model_file = os.path.join(args.model, 'model.pyc' )
    model = torch.load(model_file)
    print "AlexNet Loaded ..."
    
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    test_transform = transforms.Compose([  transforms.Scale(256),
                                           transforms.CenterCrop(224),
                                           transforms.ToTensor(), normalize,])

    val_loader   = torch.utils.data.DataLoader( datasets.ImageFolder(valdir, test_transform),
                                                batch_size=50,
                                                shuffle=False,
                                                pin_memory=True)
    
    print '\nTesting AlexNet with Live Camera Feed...'

    print 'Real-time Visual Classification:'
    print 'Capturing live feed from camera ...'
    print 'Press Ctrl+C to halt program ...\n'
    time.sleep(5)
    alexnet_camfeed(model, 0)
    print '===================================================================='
