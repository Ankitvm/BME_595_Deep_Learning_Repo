#!/usr/bin/env python

#-------------------------------------------------------------------------------
"""main.py:   Module for executing and testing the class Conv2D for performing
              2D convolution on a set of images"""

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
This program executes the all the Parts of Homework 1 for a pair of images in the
folder ./images/ and saves the results in the folder ./results/. It makes use of
the class Conv2D to carry out convoultion for different parameters. The outputs generated
are as follows:

Part A:
The output images for each of the three tasks in Part A as saved as images in the
following format:
- image_<image_number>_task_<task_number>_ch_<channel_number>.jpg -
  This is the output image for given channel and for the give task in Part A

Part B:
The plot for Part B graphing Time v/s. Output Channels is saved as follows:
- image_<image_number>_part_b_plot.jpg - 
  This is the output plot of Time Taken v/s. Number of Output Channels for Part B
  of Homework 1

Part C:
The plot for Part C graphing Operations v/s. Kernel Size is saved as follows:
- image_<image_number>_part_c_plot.jpg - 
  This is the output plot of Number of Operations v/s. Kernel Size for Part C
  of Homework 1

Part D:
The plot for Part D graphing Time v/s. Output Channels is saved as follows:
- image_<image_number>_part_d_plot.jpg - 
  This is the output plot of Time Taken v/s. Number of Output Channels for Part D
  of Homework 1
This is text file containing values of the time taken for C implementation
- part_d_image_<image_number>_time_taken.txt -
  .txt file containing the output for implementation of Part D

-----------------------------------------------------------------------------"""

import time
import math
import os, sys
import numpy as np
import cv2
import matplotlib.pyplot as plt

from conv import Conv2D
from subprocess import call

if __name__ == "__main__":

    print '---------------------------------------------------------------------'
    print 'Course: \tBME 595'
    print 'Homework:\t1'
    print 'Date: \t08/28/2017'
    print 'Performer: Ankit Manerikar'
    print 'Inst.: \t Purdue University'
    print '---------------------------------------------------------------------'
    
    print '\nProgram Start Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())

    im_no           = 2 
    input_image_loc = str('./images/')
    result_loc      = str('./results/')
    
    # Loading Images
    print 'Loading Images from File Location,', input_image_loc
    image_list = []
    
    for i in range(im_no):
        img_name    = input_image_loc+str('image_%i.jpg'%(i+1))
        curr_img    = cv2.imread(img_name,cv2.IMREAD_COLOR)
        image_list.append(curr_img)

    curr_no = 1
    task_no = 1
        
    for current_image in image_list:
        
        print '\nImage No: ', curr_no
    # Part A--------------------------------------------------------------------
        print '\nPART A: 2D Convolution'
    
    # Task 1
        conv2d = Conv2D(3,1,3,1,'known')
        [op_count, float_tensor] = conv2d.forward(current_image)

        for i in range(len(float_tensor)):
            result_file = result_loc + str('image_%i_task_%i_ch_%i.jpg'%(curr_no,task_no,i))
            cv2.imwrite(result_file, float_tensor[i])
            print '\nOutput Channel:\t',i,'\t Result saved at:\t', result_file

        print 'Task 1 completed ...'
        task_no += 1
        
    # Task 2
        conv2d = Conv2D(3,2,5,1,'known')
        [op_count, float_tensor] = conv2d.forward(current_image)

        for i in range(len(float_tensor)):
            result_file = result_loc + str('image_%i_task_%i_ch_%i.jpg'%(curr_no,task_no,i))
            cv2.imwrite(result_file, float_tensor[i])
            print '\nOutput Channel:\t',i,'\t Result saved at:\t', result_file

        print 'Task 2 completed ...'
        task_no += 1
    # Task 3
        conv2d = Conv2D(3,3,3,2,'known')
        [op_count, float_tensor] = conv2d.forward(current_image)

        for i in range(len(float_tensor)):
            result_file = result_loc + str('image_%i_task_%i_ch_%i.jpg'%(curr_no,task_no,i))
            cv2.imwrite(result_file, float_tensor[i])
            print '\nOutput Channel:\t',i,'\t Result saved at:\t', result_file

        print 'Task 3 completed ...'

    # Part B -------------------------------------------------------------------
        print '\nPART B: Output Channels v/s Time'

        time_taken = []
        
        for i in range(11):
            conv2d = Conv2D(3,2**i,3,1,'rand')
            start_time = time.time()
            [op_count, float_tensor] = conv2d.forward(current_image)
            time_elapsed = time.time() - start_time

            print 'Output Channels:\t%i \tTime Elapsed:\t%.5f s'%(2**i,time_elapsed)
            time_taken.append(time_elapsed)

        plt.figure(1)
        plt.plot(range(11),time_taken)
        plt.title('Part B: Number of Output Channels v/s. Time Taken')
        plt.xlabel('Number of Output Channels (2^i)')
        plt.ylabel('Time Taken (seconds)')
        plt.xticks(np.arange(len(time_taken)), [str('%i'%2**i) for i in range(len(time_taken))])

        result_file = result_loc + str('image_%i_part_b_plot.jpg'%(curr_no))
        plt.savefig(result_file)
        print '\nPART B completed ...'
        print 'Result saved in ', result_file

    # Part C -------------------------------------------------------------------
        print '\nPART C: Kernel Size v/s Time'

        no_of_operations = []
        
        for i in range(3,13,2):
            conv2d = Conv2D(3,2,i,1,'rand')
            [op_count, float_tensor] = conv2d.forward(current_image)
            no_of_operations.append(op_count)
            print 'Kernel Size:\t%i \tNumber of Operations:\t %i'%(i,op_count)

        plt.figure(1)
        plt.plot(range(len(no_of_operations)), no_of_operations)
        plt.title('Part C: Kernel Size v/s. Number of Operations')
        plt.xlabel('Kernel Size')
        plt.ylabel('Number of Operations')
        plt.xticks(np.arange(len(no_of_operations)), [str('%i'%i) for i in range(3,13,2)])
        
        result_file = result_loc + str('image_%i_part_c_plot.jpg'%(curr_no))
        plt.savefig(result_file)

        print '\nPart C completed ...'
        print 'Result saved in ', result_file


    # Part D --------------------------------------------------------------------
        print '\nPART D: C Implementation for Part B'

        os.system('./main ./images/image_1.jpg ./image_2.jpg')
        file_name = str('./results/part_d_image_%i_time_taken.txt'%curr_no)

        while(not os.path.isfile(file_name)):
            print 'Waiting for Part D to be complete ...'

        with open(file_name) as part_d_file:

            d_time_taken = part_d_file.read().splitlines()
            print d_time_taken
            plt.figure(curr_no)
            plt.plot(range(10),d_time_taken)
            plt.title('Part D: Number of Output Channels v/s. Time Taken')
            plt.xlabel('Number of Output Channels (2^i)')
            plt.ylabel('Time Taken (seconds)')
            plt.xticks(np.arange(len(d_time_taken)), [str('%i'%2**i) for i in range(len(d_time_taken))])

            result_file = result_loc + str('image_%i_part_d_plot.jpg'%(curr_no))
            plt.savefig(result_file)
            print '\nPART D completed ...'
            print 'Result saved in ', result_file
            
        
        
        print 'Program Completed'
        print '\nProgram End Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())
        curr_no += 1

# Program Ends ------------------------------------------------------------------
#--------------------------------------------------------------------------------
