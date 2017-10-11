"""test.py:   Module fore testing the NeuralNetwork class as well as the logic gates
              generated from this class"""

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

import time
import math
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
#-------------------------------------------------------------------------------

from neural_network import NeuralNetwork
from logic_gates    import AND,OR,NOT,EXOR

if __name__ == "__main__":

    print '---------------------------------------------------------------------'
    print 'Course: \tBME 595'
    print 'Homework:\t2'
    print 'Date: \t09/07/2017'
    print 'Performer: Ankit Manerikar'
    print 'Inst.: \t Purdue University'
    print '---------------------------------------------------------------------'
    
    print '\nProgram Start Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())

    # Part A -------------------------------------------------------------------

    print '\n PART A: \t Neural Network - Forward Propagation'

    print "Initializing Neural Network Class ..."
    model = NeuralNetwork([2,2,1])

    print "Neural Network Object Initialized."

    print "NN Specifications:"
    print "Number of Inputs:", model.size_list[0]-1
    print "Number of Outputs:", model.size_list[-1]
    print "Number of Layers:", len(model.size_list)-1, '\n'

    print 'Displaying NN Layers ...'

    for k in range(len(model.network)):

        print '\n Layer %i:'%k
        print model.getLayer(k)


    test_inp = torch.ones(2,1)
    print '\nTesting Output ...'

    print 'Test Input:', test_inp
    print 'Output:', model.forward(test_inp)
    
    print 'PART A Completed'
    print '---------------------------------------------------------------------'

    print '\nPART B: \t Neural Networks - Logic Gates'
    # Test Inputs
    test_input_1 = [[False, False],
                   [False,  True],
                   [True,  False],
                   [True,   True]]
    
    test_input_2 = [False,  True]

    # AND Gate -----------------------------------------------------------------
    print '\nInitializing Object for AND Gate ...'
    ANDCurr = AND()

    print 'Testing Output ...'

    for test_inp in test_input_1:
        print 'Input 0: ', test_inp[0],'  \tInput 1: ', test_inp[1], '  \tOutput: ', \
              ANDCurr(test_inp[0],test_inp[1])

    # OR Gate -----------------------------------------------------------------
    print '\nInitializing Object for OR Gate ...'
    ORCurr = OR()

    print 'Testing Output ...'

    for test_inp in test_input_1:
        print 'Input 0: ', test_inp[0],'  \tInput 1: ', test_inp[1], '  \tOutput: ', \
              ORCurr(test_inp[0],test_inp[1])

    # NOT Gate -----------------------------------------------------------------
    print '\nInitializing Object for NOT Gate ...'
    NOTCurr = NOT()

    print 'Testing Output ...'

    for test_inp in test_input_2:
        print 'Input 0: ', test_inp, '  \tOutput: ', \
              NOTCurr(test_inp)


    # EXOR Gate -----------------------------------------------------------------
    print '\nInitializing Object for EXOR Gate ...'
    EXORCurr = EXOR()

    print 'Testing Output ...'

    for test_inp in test_input_1:
        print 'Input 0: ', test_inp[0],'  \tInput 1: ', test_inp[1], '  \tOutput: ', \
              EXORCurr(test_inp[0],test_inp[1])
    
    print '\nPART B Completed'
    print '---------------------------------------------------------------------'
    print '\nProgram End Time : ', time.strftime('%d-%m-%Y %H:%M:%S', time.localtime())
