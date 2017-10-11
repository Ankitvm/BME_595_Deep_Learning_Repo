# **LeNet-5 Network for CIFAR-100 Dataset**


### **Synopsis**
This project contains code and drivers for implementing a LeNet-5 Neural Network in Python that is used to compare its performnace with a simple Neural network with regard to the MNIST dataset as well as is used to create an image classifier that is trained using the CIFAR100 dataset.  The files that form a part of this project are explained as follows: 

**PART A**: *Comparison of Simple and Convolutional Neural Networks* (**File**: *img2num*.py): 
This file contains the classes for creating neural networks for training on MNIST dataset for diigt classification. The two class are the Img2Num() which creates an instance of a fully connected neural network for building the MNIST classifier while the class CNNImg2Num() that creates an instance of a LeNet-5 convolutional neural network for MNIST classifcation. Both class objects can be trained and tested on the MNIST dataset and contain methods for training and generating an output instance. Running this program will allow the two classes to be instanced, trained on the MNIST Dataset and tested for accuracy for Speed and Accuracy on MNISt Test set.

**PART B**: *LeNet-5 CIFAR-100 Classifier* - (**File**: *img2obj*.py) :
This file contains the neural network class Img2Num() which is used to create an instance of a LeNet-5 CNN trained on a CIFAR-100 dataset. This 5-layer CNN instanced by this class can be used for classification of 32x32 pixel images according to the 100 labels defined by the CIFAR-100 dataset. In addition methods for training and testing the CNN, the class also methods for viewing input images with their labels as well as labeling images streaming from a camera. 

### Principle of Operation - Fully Connected and Convolutional Neural Networks:
LeNet-5 Neural Nets are 5-layer convolutional neural networks that can be used for simple image classifcation utilizing the advantage of the depth provided by the convolutional layers to train better classifiers for a dense feature set. The LeNet-5 Neural Net consists of 2 convolutional layers followed by 3 fully-connected layers in its design structure. When training a neural network for image classification, it is often observed that the pixelwise features that are extracted from images are locally similar, hence, a dense feature set can be more efficiently trained by convolving the local features by a suitable filter or kernel. This notion goes into the use fo convolutional layers for implementing neural nets for multi-channel data such as color images. This project explores the LeNet-5 CNN for its effectiveness in implementation for different datasets.

### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- OpenCV (installed with WITH_FFMPEG=ON)
- PyTorch 

### Installation Procedure 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files img2num.py, img2obj.py into the folder where the program file exists and import the same in the program code.

To run the program:

1 *img2num*.py:
- Open the terminal and navigate to the directory
- run the command:
  > python img2num.py
  
(The program should create and train a fully-connected neural network and a convolutional network for the MNIST dataset amd display the training and test accuracy and speed for the two at the output.)

2 img2obj.py:
- Open the terminal and navigate to the directory
- run the command:
  $ python img2obj.py

(The program should create and train  a convolutional network for the MNIST dataset amd display the training and test accuracy and speed for the same. The program will then prompt the user to enter the index of the test input to be predicted - the program will then output the test image as well as its predicted and ground truth labels. followinf this, the program will stream images from the camera and display the predicted labels for the object present in the frame.)

### Code Example

- To use the class Img2Num() in a program, type the following commands:

```from img2obj import Img2Num 
NNModel = Img2Num()
 # To train CNN
print NNModel.train()
 # To get output
output = NNModel(Input_32_x_32_Image_Tensor)
print 'Predicted Label:\t', pred.data.max(1, keepdim=True)[1].numpy()[label_no]
 # To view Output
 NNModel.view(Input_32_x_32_Image_Tensor)
 # To stream input from camera wit hdivce index idx
 NNModel.cam(idx)
 ```

###Author

Ankit Manerikar, Robot Vision Lab, Purdue University
