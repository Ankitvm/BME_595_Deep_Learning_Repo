amanerik_HW02
=======
(Homework No 2 - Course: BME 595)

#### Synopsis

This project contains code and drivers for implementing a neural network class in Python 
as well as derive logic gate implementations from the same. The files that form 
a part of this project are explained as follows: 

neural_network.py:
This file contains the class NeuralNetwork([layer_size_list]) that creates an 
instance of a n-layer neural network as per the number of layers and layer sizes 
provided as input. 

logic_gates.py:
This file contains the neural network implementations of the logic gates AND,OR,
NOT and EXOR using the NeuralNetwork() Class from neural_network.py

test.py:
This is a test program which checks the class operations from neural_network.py
and logic_gates.py

### System Pre-requisites:
To run this program your machine should support:
- Python 2.7
- Anaconda
- PyTorch

### Install 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files neural_network.py, 
logic_gates.py into the folder where the program file exists and import the same 
in the program code.

To run the program:
-> neural_network.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython neural_network.py
(The program should generate a neural network with randomly initialized weights
and display its layers and output.)

-> logic_gates.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython logic_gates.py
(The program should generate a neural network implementations of AND,OR,NOT,EXOR 
gates and display their truth tables.)

-> test.py:
- Open the terminal and navigate to the directory
- run the command:
  $ ipython test.py
(The first part of the program should generate a neural network with randomly 
initialized weights and display its layers and output
 The second part of the prgoram generate a neural network implementations of 
 AND,OR,NOT,EXOR gates and display their truth tables.)


### Code Example

- To use the class NeuralNetwork() in a program, type the following commands:

>> from neural_network import NeuralNetwork
>> NNModel = NeuralNetwork([<input_layer_size, layer_1_size, ..., output_layer_size>])
>> # To display layers
>> print NNModel.getLayer(<Layer_No>)
>> # To get output
>> print NNModel.forward(<Input_Tensor>)

- To use the logic gate implementation,e.g., AND gates in a program, 
type the following commands:

>> from logic_gates import AND
>> ANDGate = AND()
>> # To display output for specified input
>> print AND(False, False)

#### Class Description
Python Class NeuralNetwork():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- __init__([in0, h1,h2,...,out]): (Constructor)

in0 - size of input layer
hn - size of nth layer
out - size of output layer

[DoubleTensor] getLayer(layer_no):
Get the layer corresponding to the index layer_no

layer_no - index of the layer

Returns: 
Tensor for the nth layer

[DoubleTensor] forward(input_tensor):
input_tensor - the input for which NN output is to be generated 

Returns:
Tensor for the output

Python Class AND():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- __init__(): (Constructor)


[DoubleTensor] __call__(input_1,input_2):
input_1, input_2 - the inputs for which NN output is to be generated 

Returns:
Tensor output


### Authors
Ankit Manerikar (Robot Vision Lab, Purdue)
