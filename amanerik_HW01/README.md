amanerik_HW01
=======
(Homework No 1 - Course: BME 595)

#### Synopsis

This project contains code and drivers for implementing 2D convolution on a set 
of images with the options of input channels, output channels, kernel size, 
stride and mode. The files that form a part of this project are explained as follows: 

conv.py:
This file contains the class Conv2D() that implements the convolution

main.py:
This is the main program that calls the Conv2D() and executes convolution on a
set of images providing as output the convolved image channels as well as plots 
show analysis of convolution operation for varyng parameters.

main.c:
This is the C implementation of the Part D of Homework 1. 

/images/:
This folder is used to store the images used as input for convolution - the images 
are stored sequentially as "image_<image_number>.jpg".
(When changing the number of images in the folder - change the variable im_no in 
main.py to the desired number of images)

/results/:
This folder contains the results of the execution of the program main.py and 
consists of the following images:

- image_<image_number>_task_<task_number>_ch_<channel_number>.jpg -
  This is the output image for given channel and for the give task in Part A

- image_<image_number>_part_b_plot.jpg - 
  This is the output plot of Time Taken v/s. Number of Output Channels for Part B
  of Homework 1

- image_<image_number>_part_c_plot.jpg - 
  This is the output plot of Number of Operations v/s. Kernel Size for Part C
  of Homework 1

- image_<image_number>_part_c_plot.jpg - 
  This is the output plot of Time Taken v/s. Number of Output Channels for Part D
  of Homework 1

- part_d_image_<image_number>_time_taken.txt -
  .txt file containing the output for implementation of Part D

### Install 
(For Linux OS only)
- Open the terminal and navigate to the directory
- run the command:
  $ make
- Once the code is built, it is ready for exceution
  (main.py makes a system call to main.c for exceution of Part D, hence the above command is necessary
   for execution) 

### Pre-requisites:
To run this program your machine should support:
- Python 2.7
- OpenCV 2.x.x/3.x.x
- CMake

### Code Example

Python:
- To run the program main.py on a set of images:
  -> Save the images into the folder ./images/ renamed as "image_<image_number>.jpg"
  -> change the variable im_no in main.py to the desired number of images
  -> run the command on terminal:
     $ python main.py

- To use the class Conv2D in a program, type the following commands:
>> from conv import Conv2D
>> import cv2
>> curr_image = cv2.imread(<image_location>, <image_type>)
>> [opcount, float_tensor] = conv2d = Conv2D(in_channel=3, o_channel=3, kernel_size=3, stride=1, mode= 'known')
>> imshow("Image Channel 1", float_tensor[0])
>> imshow("Image Channel 2", float_tensor[1])
>> imshow("Image Channel 3", float_tensor[2])

This will display the output channels of the convolved image independently. 

C:
To test output, Run the command:
$ ./main ./images/image_1.jpg ./images/image_2.jpg

and check the images in folder /results/ 
for results

#### Class Description
Python Class Conv2D:
This is the class that executes 2D convolution on an image. Its methods are described below:

- __init__(in_channel, o_channel, kernel_size, stride, mode): (Constructor)

in_channel - no of input channels
o_channel - number of output channels (less than 3)
kernel_size - size of kernel (either 3 or 5)
	      if value is 3, the kernels are chosen as:
	      K1 = [-1, -1, -1; 0, 0, 0; 1, 1, 1]
	      K2 = [-1, 0, 1; -1, 0, 1; -1, 0, 1]
	      K3 = [ 1, 1, 1; 1, 1, 1; 1, 1, 1]
   	      if value is 5, the kernels are chosen as:
	      K4 = [-1, -1, -1, -1, -1; -1, -1, -1, -1, -1; 0, 0, 0, 0, 0; 1, 1, 1, 1, 1; 1, 1, 1, 1, 1]
	      K5 = [-1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1; -1, -1, 0, 1, 1]
              
stride - no of pixels to skip
mode - can be set to two values: 
       'known' - in this case, the kernels are selected from k1,k2,k3,k4,k5
       'rand'  - in this case, a kernel of size kernel_size is randomly initialized 

[opno, float_tensor] forward(current_image):

current_image - the input image on which convolution is performed

Returns:
- opno - Number of operations 
- float_tensor - output convolved with number of channels equal to o_channels

/results_old/:
These contain sample results of the program main.py for a previous run.

### Authors
Ankit Manerikar (Robot Vision Lab, Purdue)
