#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""conv.py:   Module containg the class Conv2D for implementing 2D convolution on
              a set of images using different kernels and """

__author__      = "Ankit Manerikar"
__copyright__   = "Copyright (C) 2017, Purdue University"
__date__        = "28th August, 2017"
__credits__     = ["Ankit Manerikar"]
__license__     = "Public Domain"
__version__     = "1.0"
__maintainer__  = "Ankit Manerikar"
__email__       = "amanerik@purdue.edu"
__status__      = "Prototype"
#-------------------------------------------------------------------------------

"""-----------------------------------------------------------------------------
* Module Description:
This module contains the class Conv2D that allows 2D convolution on a set of images
while specifying different parameters. These parameters include:

in_channel  - number of input channels
o_channel   - number of output channels
kernel_size - kernel size
stride      - number of pixels to skip
mode        - two modes:
              'known' - chooses kernels from a preset list of kernels
              'rand'  = randomly initializes a kernel of a given size

The methods for this class are:
__init__()    - Constructor for Conv2D
forward()     - method for implementing convolution on a 2D image
-----------------------------------------------------------------------------"""
import time
import math
import os, sys
import numpy as np
##import pytorch
import cv2
import matplotlib.pyplot as plt

class Conv2D(object):
    
    # Data Attributes ----------------------------------------------------------
    kernel_size_3_list =   [  np.array([[-1,-1,-1],
                                        [ 0, 0, 0],
                                        [ 1, 1, 1]]),
                              np.array([[-1, 0, 1],
                                        [-1, 0, 1],
                                        [-1, 0, 1]]),
                              np.array([[ 1, 1, 1],
                                        [ 1, 1, 1],
                                        [ 1, 1,1]]) ]
    
    kernel_size_5_list =   [  np.array([[-1,-1,-1,-1,-1],
                                        [-1,-1,-1,-1,-1],
                                        [ 0, 0, 0, 0, 0],
                                        [ 1, 1, 1, 1, 1],
                                        [ 1, 1, 1, 1, 1]]),
                              np.array([[-1,-1, 0, 1, 1],
                                        [-1,-1, 0, 1, 1],
                                        [-1,-1, 0, 1, 1],
                                        [-1,-1, 0, 1, 1],
                                        [-1,-1, 0, 1, 1]]) ]
    #---------------------------------------------------------------------------

    def __init__(self, in_channel, o_channel, kernel_size, stride, mode):
        """---------------------------------------------------------------------
        Desc.:  Constructor Method
        Args:   in_channel  - number of input channels
                o_channel   - number of output channels
                kernel_size - kernel size
                stride      - number of pixels to skip
                mode        - two modes:
                              'known' - chooses kernels from a preset list of kernels
                              'rand'  = randomly initializes a kernel of a given size
        Returns: -
        ---------------------------------------------------------------------"""

        self.in_channel = in_channel
        self.o_channel  = o_channel
        self.kernel_size = kernel_size
        
        if   mode == 'known':

            if   kernel_size == 3:
                self.k_curr = self.kernel_size_3_list
            elif kernel_size == 5:
                self.k_curr = self.kernel_size_5_list
            else:
                print 'Option not recognized ...'
                self.k_curr = self.kernel_size_3_list

        elif mode == 'rand':
            self.k_curr = []
            
            for k in range(o_channel):
                self.k_curr.append(np.random.rand(kernel_size,kernel_size))

        self.padding = (kernel_size - kernel_size%2)/2

        self.stride = stride
    #---------------------------------------------------------------------------
            
    def forward(self,input_image):
            """---------------------------------------------------------------------
            Desc.:  Method for implementing 2D convolution on an input image
            Args.:  input_image - the image to be convolved - should passed as a
                                 2D numpy array
            Returns: opno - number of operations required
                     float_tensor - output channels of convolved image
            ---------------------------------------------------------------------"""
            opno = 0
            [W,H,C] = input_image.shape
            float_tensor_3D = []
            
            for out_ch in range(self.o_channel):

                curr_chnl_img = np.zeros(shape=[W,H])
                
                for inp_ch in range(C):

                    for i in range(self.padding, W-self.padding,self.stride):
                        for j in range(self.padding, H-self.padding,self.stride):

                            curr_mat = input_image[i-self.padding:i+self.padding+1,
                                                   j-self.padding:j+self.padding+1,
                                                   inp_ch]

                            curr_chnl_img[i,j] += np.multiply(self.k_curr[out_ch], curr_mat).sum()

                            opno += 2*((self.kernel_size)**2) - 1 

                float_tensor_3D.append(curr_chnl_img)
            
            return [opno, np.asarray(float_tensor_3D)]


if __name__ == '__main__':
    curr_img    = cv2.imread('./images/image_1.jpg',cv2.IMREAD_COLOR)
    
    conv2d = Conv2D(3,1,3,1,'rand')
    [op_count, float_tensor] = conv2d.forward(curr_img)
    for k in range(len(float_tensor)):
        fname = str('./ch%i.jpg'%k)
        cv2.imwrite(fname, float_tensor[k])

# Class Ends -------------------------------------------------------------------
# ------------------------------------------------------------------------------
    
    
