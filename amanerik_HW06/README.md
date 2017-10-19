# **AlexNet for Tiny ImageNet Dataset**


### **Synopsis**
This project contains code and drivers for using a pre-trained AlexNet from the pytorch and training only its last layer on the Tiny ImageNet dataset - the project thus is a demonstration of a transfer learning procedure for a deep net. The files that form a part of this project are explained as follows: 

**PART A**: *Transfer Learning using AlexNet* (**File**: *train*.py): 

This program contains code for training the last layer of the pre-trained AlexNet using transfer learning on the Tiny ImageNet dataset. The program reads in the arguments --data which holds the location of the dataset and --save which contains the location at which hte model is to be saved. The pre-trained AlexNet is loaded from the Pytorch module and the last fully-connected 4096 x 1000 layer of the Alexnet is then replaced by an untrained linear NN layer. This layer is trained using back-propagation pass using the ImageNet dataset with an SGD optimizer and a CrossEntropyLoss() criterion.

**PART B**: *Testing ALexNet on a Live CameraFeed* - (**File**: *test*.py) :

This program is used to test the trained AlexNet by predicting labels on image frames streaming from a camera.


### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- OpenCV (installed with WITH_FFMPEG=ON)
- PyTorch 

### Installation Procedure 
(For Linux OS only)
 
To run the program, download the folder and unzip it into the desired directory.

1 *train*.py:
- Open the terminal and navigate to the directory
- run the command:
```
  > python train.py --data /<path of the dataset> --save /<path to save trained model>
```
- By default, the dataset path is ./tiny-imagenet-200/ and the path for the trained model is ./result_model/

(The program should generate and save an AlexNet whose last layer is retrained using th TinyImagenet Dataset.)

2 *test*.py:
- Open the terminal and navigate to the directory
- run the command:

```
  $ python test.py --model /<path of the saved model>
```
- By default, the path to the saved model is ./result_model/

(The program should load the trained model and stream images from the system camera displaying the predicted labels for the object present in the frame.)

### Output:
```
====================================================================
Transfer Learning: AlexNet for Tiny ImageNet Dataset
====================================================================


Program Start Time :  18-10-2017 15:46:17

Loading Tiny ImageNet Dataset from the directory,  ./tiny-imagenet-200/ ...
Loading AlexNet ...
Dataset and DeepNet Loaded ...


Training Last Layer of AlexNet using the Tiny ImageNet Dataset ...

Epoch-wise Performance:
[1,    50] loss: 6.207
[1,   100] loss: 4.973
[1,   150] loss: 4.486
[1,   200] loss: 4.092
[1,   250] loss: 3.888
[1,   300] loss: 3.750
[1,   350] loss: 3.646
[1,   400] loss: 3.560
[1,   450] loss: 3.433
[1,   500] loss: 3.437
.
.
.

Classifier Trained ... 
-----------------------------------

 Testing Classifier with the validation dataset ...

Convolutional Neural Network:
Test Accuracy = 67.03%
Loss: 0.3253
Time Taken for Training: 25453.912 s

Classifier saved as:  ./result_model/model.pyc
=======================================================================
```


### Author

Ankit Manerikar, Robot Vision Lab, Purdue University
