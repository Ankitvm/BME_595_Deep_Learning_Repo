# **Homework 5 - LeNet-5 Network for CIFAR-100 Dataset**

#### **Course**: BME 595
####  **Name**: Ankit Manerikar
####  **PUID**: 0028119077
####  **Date**: 26th September, 2017

### **Synopsis**
This project contains code and drivers for implementing a LeNet-5 Neural Network in Python that is used to compare its performnace with a simple Neural network with regard to the MNIST dataset as well as is used to create an image classifier that is trained using the CIFAR100 dataset.  The files that form a part of this project are explained as follows: 

**PART A**: *Comparison of Simple and Convolutional Neural Networks* (**File**: *img2num*.py): 
This file contains the classes for creating neural networks for training on MNIST dataset for diigt classification. The two class are the Img2Num() which creates an instance of a fully connected neural network for building the MNIST classifier while the class CNNImg2Num() that creates an instance of a LeNet-5 convolutional neural network for MNIST classifcation. Both class objects can be trained and tested on the MNIST dataset and contain methods for training and generating an output instance. Running this program will allow the two classes to be instanced, trained on the MNIST Dataset and tested for accuracy for Speed and Accuracy on MNISt Test set.

**PART B**: *LeNet-5 CIFAR-100 Classifier* - (**File**: *img2obj*.py) :
This file contains the neural network class Img2Num() which is used to create an instance of a LeNet-5 CNN trained on a CIFAR-100 dataset. This 5-layer CNN instanced by this class can be used for classification of 32x32 pixel images according to the 100 labels defined by the CIFAR-100 dataset. In addition methods for training and testing the CNN, the class also methods for viewing input images with their labels as well as labeling images streaming from a camera. 

### Principle of Operation - Fully Connected and Convolutional Neural Networks:
LeNet-5 Neural Nets are 5-layer convolutional neural networks that can be used for simple image classifcation utilizing the advantage of the depth provided by the convolutional layers to train better classifiers for a dense feature set. The LeNet-5 Neural Net consists of 2 convolutional layers followed by 3 fully-connected layers in its design structure. When training a neural network for image classification, it is often observed that the pixelwise features that are extracted from images are locally similar, hence, a dense feature set can be more efficiently trained by convolving the local features by a suitable filter or kernel. This notion goes into the use fo convolutional layers for implementing neural nets for multi-channel data such as color images. This project explores the LeNet-5 CNN for its effectiveness in implementation for different datasets.

### Implementation:

### Part A: Comparison between Fully Connected and Convolutional Neural Networks:
Part A for the project is implemented in the program img2num.py and has the following specifcations:

This code contains the neural network class Img2Num() which is trained on the MNIST dataset to create a classifier - the class Img2Num() is inherited from the pytorch.nn module and instances a fully-connected NN for training on an MNIST dataset. The NN is trained on batches of size 10 for 2 epochs with the MSE error threshold of 0.01. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The SGD optimizer is used for training with a learning rate of 0.01 and momentum of 0.9. 

This code also  contains the neural network class CNNImg2Num() which is trained on the MNIST dataset to create a classifier - the class CNNImg2Num() is inherited from the pytorch.nn module and instances a LeNet-5 NN for training on an MNIST dataset. The NN is trained on batches of size 10 for 2 epochs with the MSE error threshold of 0.01. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The SGD optimizer is used for training with a learning rate of 0.01 and momentum of 0.9. 

### Part B: LeNet-5 CNN for CIFAR100 Dataset:
Part B of the project is implemented in the code img2obj.py which contains the neural network class Img2Num() which is trained on the CIFAR-100 dataset to create a classifier - the class Img2Num() is inherited from the pytorch.nn module and instances a LeNet-5 NN for training on a CIFAR-100 dataset. The NN is trained on batches of size 50 for 50 epochs using SGD optimization and a CrossEntropyLoss criterion for the loss function. The training dataset has a total of 6000 samples each with a size of 3 x 32 x 32 thus giving a 3072-size feature vector. The SGD optimizer is used for training with a learning rate of 0.02 and momentum of 0.9. 

## Operational Details:
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

### **Results**:

The output logs for PART A (img2num.py) and PART B (img2obj.py) are generated as follows - the speed and accuracy of the Neural Nets implemented can be observed from these output logs.:

#### PART A:
 ```
 ====================================================================
Comparison between Simple and Convolutional Neural Networks
====================================================================


Program Start Time :  26-09-2017 23:05:56
----------------------------------------------------------------
Digit Classfier using pytorch nn module
Author: Ankit Manerikar
Written on: 09-24-2017
----------------------------------------------------------------
Class initialized

Training Neural network using the MNIST Dataset ...

Epoch-wise Performance:
[1,   200] loss: 0.129
[1,   400] loss: 0.053
[1,   600] loss: 0.043
[1,   800] loss: 0.038
[2,   200] loss: 0.033
[2,   400] loss: 0.032
[2,   600] loss: 0.031
[2,   800] loss: 0.029
MNIST classifier trained ...
Testing Neural Network for MNIST Dataset...

Simple Neural Network:
Accuracy = 92.280%
Loss: 0.275
Time Taken for Training: 10.139 s 

----------------------------------------------------------------
CONVOLUTIONAL Neural Network using pytorch nn module
Author: Ankit Manerikar
Written on: 09-24-2017
----------------------------------------------------------------
Class initialized

Training Neural network using the MNIST Dataset ...

Epoch-wise Performance:
[1,   200] loss: 0.223
[1,   400] loss: 0.157
[1,   600] loss: 0.092
[1,   800] loss: 0.071
[2,   200] loss: 0.057
[2,   400] loss: 0.051
[2,   600] loss: 0.047
[2,   800] loss: 0.043
MNIST classifier trained ...
Testing Neural Network for MNIST Dataset...

Convolutional Neural Network:
Accuracy = 88.260%
Loss: 0.378
Time Taken for Training: 21.912 s
 ```
 
 PART B:
 ``` 
  Files already downloaded and verified
Files already downloaded and verified
====================================================================
Convolutional Neural Network for CIFAR100 Dataset
====================================================================


Program Start Time :  26-09-2017 23:12:06
----------------------------------------------------------------
LeNet-5 Neural Network using pytorch nn module
Author: Ankit Manerikar
Written on: 09-24-2017
----------------------------------------------------------------
Class initialized

Training Neural network using the CIFAR100 Dataset ...

Epoch-wise Performance:
[1,   200] loss: 4.605
[1,   400] loss: 4.591
[1,   600] loss: 4.543
[1,   800] loss: 4.464
[1,  1000] loss: 4.393
[2,   200] loss: 4.328
[2,   400] loss: 4.289
[2,   600] loss: 4.251
[2,   800] loss: 4.219
[2,  1000] loss: 4.186
.
.
.
[49,   200] loss: 3.793
[49,   400] loss: 3.831
[49,   600] loss: 3.839
[49,   800] loss: 3.836
[49,  1000] loss: 3.854
[50,   200] loss: 3.826
[50,   400] loss: 3.818
[50,   600] loss: 3.825
[50,   800] loss: 3.846
[50,  1000] loss: 3.853
CIFAR100 classifier trained ...

Testing Neural Network for CIFAR100 Dataset...

Convolutional Neural Network:
Test Accuracy = 30.506%
Loss: 3.892
Time Taken for Training: 997.896 s

Operational Example:
Enter Label Number to test:	5
Ground truth: 	28
Predicted Label:	[27]
Displaying Image (Close figure for execution of Part B)... 

--------------------------------------------------------------------
Real-time Visual Classification:
Capturing live feed from camera ...
Press Ctrl+C to halt program ...

init done
Frame Time: 27-09-2017 00:37:57 	Identified Object Label:	[[41]]
Frame Time: 27-09-2017 00:37:57 	Identified Object Label:	[[73]]
.
.
.
```
