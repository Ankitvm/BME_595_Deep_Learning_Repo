# **Homework No 4 - MNIST Neural Network Classifier**

### **Synopsis**
This project contains code and drivers for implementing a neural network class in Python to classify written character recognition in two ways - one using the NeuralNetwork() class and the other using th pytorch nn module.  The files that form a part of this project are explained as follows: 

### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- Anaconda
- PyTorch

### Install 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files neural_network.py, my_img2num.py nn_img2num.py into the folder where the program file exists and import the same in the program code.

To run the program:

1 NeuralNetwork.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython NeuralNetwork.py
  
(The program should generate a neural network with randomly initialized weights and display its layers and output.)

2 my_img2num.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython my_img2num.py

(The program should generate a NN instance of the MyImg2Num class and train it - the number of epochs and the error trends during the training period are displayed on the terminal.)

2 nn_img2num.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython nn_img2num.py

(The program should generate a NN instance of the NNImg2Num class and train it - the number of epochs and the error trends during the training period are displayed on the terminal.)


### Code Example

- To use the class NeuralNetwork() in a program, type the following commands:

```from neural_network import NeuralNetwork 
 NNModel = NeuralNetwork([input_layer_size, layer_1_size, ..., output_layer_size])
 To display layers
 print NNModel.getLayer(Layer_No)
 To get output
 print NNModel.forward(Input_Tensor)
 ```

- To use the MNIST CLassifier from my_img2num.py, type the following commands:
```
> from my_img2num.py import MyImg2Num
> MNISTNN = MyImg2Num()
> #To display output for specified input
> print MNISTNN.forward(<input_tensor>)
```

(NN_img2Num Classfier can be operated on similar grounds.)

### Authors
Ankit Manerikar, Robot Vision Lab, Purdue University
