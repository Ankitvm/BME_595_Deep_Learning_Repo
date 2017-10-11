# **Homework No 3 - Backpropagation Pass**

### **Synopsis**
This project contains code and drivers for implementing a neural network class in Python as well as derive logic gate implementations from the same. The files that form  a part of this project are explained as follows: 

*NeuralNetwork.py*:
This file contains the class NeuralNetwork([layer_size_list]) that creates an instance of a n-layer neuralnetwork as per the number of layers and layer sizes  provided as input - the class invokes functions for forward and backpropagation pass for training a neural network.

*logic_gates.py*:
This file contains the neural network implementations of the logic gates AND,OR, NOT and EXOR using the NeuralNetwork() Class from neural_network.py - the neural network  can be trained to perform with minimal expected error.

### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- Anaconda
- PyTorch

### Install 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files neural_network.py, logic_gates.py into the folder where the program file exists and import the same in the program code.

To run the program:

1. NeuralNetwork.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython NeuralNetwork.py
  
(The program should generate a neural network with randomly initialized weights and display its layers and output.)

2. logicGates.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython logicGates.py

(The program should generate a neural network implementations of AND,OR,NOT,EXOR 
gates and display their truth tables before and after training.)

### Code Example

- To use the class NeuralNetwork() in a program, type the following commands:

> from neural_network import NeuralNetwork
> NNModel = NeuralNetwork([input_layer_size, layer_1_size, ..., output_layer_size])
> To display layers
>print NNModel.getLayer(Layer_No)
>  To get output
> print NNModel.forward(Input_Tensor)

- To use the logic gate implementation,e.g., AND gates in a program, 
type the following commands:

> from logic_gates import AND
> ANDGate = AND()
> To display output for specified input
> print AND.train()
> print AND(False, False)

