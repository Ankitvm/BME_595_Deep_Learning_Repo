# **Homework No 3 - Backpropagation Pass**

#### **Course**: BME 595
####  **Name**: Ankit Manerikar
####  **PUID**: 0028119077
####  **Date**: 14 September, 2017

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
> print AND(False, False)

#### Class Description
Python Class **NeuralNetwork**():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- _ _ init _ _([in0, h1,h2,...,out]): (Constructor)

*in0* - size of input layer
*hn* - size of nth layer
*out* - size of output layer

[DoubleTensor] **getLayer**(layer_no):
Get the layer corresponding to the index layer_no

*layer_no* - index of the layer

*Returns*: 
Tensor for the nth layer

[DoubleTensor] **forward**(input_tensor):
*input_tensor* - the input for which NN output is to be generated 

*Returns*:
Tensor for the output

[DoubleTensor] **backward**(target):
*target* - the input for which NN output is to be generated 

*Returns*:
Value for dE/dT

[DoubleTensor] **updateParams**(eta):
*input_tensor* - learning parameter

*Returns*:
Update weight matrix theta

Python Class **AND**():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- _ _ **init** _ _(): (Constructor)


[DoubleTensor] _ _ **call** _ _ (input_1,input_2):
*input_1, input_2* - the inputs for which NN output is to be generated 

*Returns*:
Tensor output

-[] **train**(): (Function for training the Neural Network)
*Args*- None
Returns - None

(The class for **AND, OR, NOT** and **XOR** is similar to the **AND** class described above)

### **Results**:

The following are the values for the weights trained using the Back-Propagation Algorithm and the manually entered weights:

*Back-Propagation Algorithm*
AND Gate:
>[
>1.00000e-02 *
>  8.6877
> -9.2074
> -0.0781
>[torch.FloatTensor of size 3x1]
>]
 
OR Gate:
>[
>-0.2415
 >0.0424
 >0.2092
>[torch.FloatTensor of size 3x1]
] 

NOT Gate:
>[
-0.2545
 0.1205
[torch.FloatTensor of size 2x1]
] 

EXOR Gate:
>[
-0.2061 -0.1393
 0.0883 -0.0449
 0.0990  0.1582
[torch.FloatTensor of size 3x2]
, 
-0.1162
-0.1857
 0.1618
[torch.FloatTensor of size 3x1]
] 

*Manually Entered Weights*:

AND Gate: 
>[20.0, 20.0, -30.0]

OR Gate:
>[20.0, 20.0, -10.0]

NOT Gate:
>[-20.0, 10.0]

EXOR Gate:
>Layer_1 : [[ 20.0,-20.0],[ 20.0,-20.0],[-10.0, 30.0]]
>Layer_2 : [20.0,20.0,-30.0]

We can see that the weights trained by backpropagation are smaller and therefore have smaller cost energy.


