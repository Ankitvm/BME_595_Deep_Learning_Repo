# **Homework No 4 - MNIST Neural Network Classifier**

#### **Course**: BME 595
####  **Name**: Ankit Manerikar
####  **PUID**: 0028119077
####  **Date**: 21st September, 2017


### **Synopsis**
This project contains code and drivers for implementing a neural network class in Python to classify written character recognition in two ways - one using the NeuralNetwork() class and the other using th pytorch nn module.  The files that form a part of this project are explained as follows: 

*NeuralNetwork.py*:
This file contains the class NeuralNetwork([layer_size_list]) that creates an instance of a n-layer neuralnetwork as per the number of layers and layer sizes  provided as input - the class invokes functions for forward and backpropagation pass for training a neural network.

*my_img2num.py*:
This file contains the neural network class MyImg2Num() which is trained on the MNIST dataset to create a classifier - this class is inherited from the NeuralNetwork() and therefore contains the forward() and train() functions defined for the class. The method forward() generates the output for the neural network for the input tensor or set of tensors. The method train() uses the Back-propagation algorithm with a Batch Gradient Descent Optimization strategy and an MSE error cost to train the classifier on the MNIST dataset. The NN is trained on batches of size 10 with the MSE error threshold of 0.4. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The NN designed is a two layer NN with the layer dimensions (784,100, 10).

*nn_img2num.py*:
This file contains the neural network class NNImg2Num() which is trained on the MNIST dataset to create a classifier - this class is gnerated using the pytorch.nn module and therefore contains all the functions defined for that class. The method forward() generates the output for the neural network for the input tensor or set of tensors. The method train() uses the Back-propagation algorithm with a Stochastic Gradient Descent Optimizer (using the optim package) and an MSE error cost to train the classifier on the MNIST dataset. The NN is trained on batches of size 10 with the MSE error threshold of 0.4. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The NN designed is a two layer NN with the layer dimensions (320, 50, 10).

### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- Anaconda
- PyTorch

### Install 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files neural_network.py, my_img2num.py, nn_img2num.py into the folder where the program file exists and import the same in the program code.

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

- To use the MNIST CLassifieer from my_img2num.py, type the following commands:
```
> from my_img2num.py import MyImg2Num
> MNISTNN = MyImg2Num()
> #To display output for specified input
> print MNISTNN.forward(<input_tensor>)
```

#### Class Description
Python Class **NeuralNetwork**():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- _ _ init _ _([in0, h1,h2,...,out]): (Constructor)

*in0* - size of input layer
*hn* - size of nth layer
*out* - size of output layer

- [DoubleTensor] **getLayer**(layer_no):
Get the layer corresponding to the index layer_no

*layer_no* - index of the layer

*Returns*: 
Tensor for the nth layer

- [DoubleTensor] **forward**(input_tensor): 
Method for forward propagation pass

*input_tensor* - the input for which NN output is to be generated 

*Returns*:
Tensor for the output

 -[DoubleTensor] **backward**(target):
 Method for backward propagation pass

*target* - the input for which NN output is to be generated 

*Returns*:
Value for dE/dT

- [DoubleTensor] **updateParams**(eta):
*input_tensor* - learning parameter

*Returns*:
Update weight matrix theta

Python Class **MyImg2Num**():
This is the class that implements an n-layer neural network for MNIST Classification. 
Its methods are described below:

- _ _ **init** _ _(): (Constructor)


[FloatTensor] **forward**(x):
x - 28 x 28 input float tensor corresponding to an MNIST image

*Returns*:
Tensor output - prediction label

-[] **train**(): (Function for training the Neural Network)
*Args*- None
Returns - None

(The class for **NNImg2Num** is similar to the **MyImg2Num** class described above)


### **Results**:

The following are the values for the computation time and MSE error for the two Neural Networks:

##### MyImg2Num():
Error:         	      0.03596  s

Computation Time:     18.62304 s

##### MyImg2Num():
Error:         		  0.05001 s

Computation Time:     2.63544 s

While the MSE error value is simlar for both the classes, we see that the NN implemented using nn_img2num.py has a faster response. 
# **Homework No 4 - MNIST Neural Network Classifier**

#### **Course**: BME 595
####  **Name**: Ankit Manerikar
####  **PUID**: 0028119077
####  **Date**: 21st September, 2017


### **Synopsis**
This project contains code and drivers for implementing a neural network class in Python to classify written character recognition in two ways - one using the NeuralNetwork() class and the other using th pytorch nn module.  The files that form a part of this project are explained as follows: 

*NeuralNetwork.py*:
This file contains the class NeuralNetwork([layer_size_list]) that creates an instance of a n-layer neuralnetwork as per the number of layers and layer sizes  provided as input - the class invokes functions for forward and backpropagation pass for training a neural network.

*my_img2num.py*:
This file contains the neural network class MyImg2Num() which is trained on the MNIST dataset to create a classifier - this class is inherited from the NeuralNetwork() and therefore contains the forward() and train() functions defined for the class. The method forward() generates the output for the neural network for the input tensor or set of tensors. The method train() uses the Back-propagation algorithm with a Batch Gradient Descent Optimization strategy and an MSE error cost to train the classifier on the MNIST dataset. The NN is trained on batches of size 10 with the MSE error threshold of 0.4. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The NN designed is a two layer NN with the layer dimensions (784,100, 10).

*nn_img2num.py*:
This file contains the neural network class NNImg2Num() which is trained on the MNIST dataset to create a classifier - this class is gnerated using the pytorch.nn module and therefore contains all the functions defined for that class. The method forward() generates the output for the neural network for the input tensor or set of tensors. The method train() uses the Back-propagation algorithm with a Stochastic Gradient Descent Optimizer (using the optim package) and an MSE error cost to train the classifier on the MNIST dataset. The NN is trained on batches of size 10 with the MSE error threshold of 0.4. The training dataset has a total of 6000 samples each with a size of 28 x 28 thus giving a 784-size feature vector. The NN designed is a two layer NN with the layer dimensions (320, 50, 10).

### System Pre-requisites:

To run this program your machine should support:

- Python 2.7
- Anaconda
- PyTorch

### Install 
(For Linux OS only)
 
To use the API in Python program, copy and paste the files neural_network.py, logic_gates.py into the folder where the program file exists and import the same in the program code.

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

- To use the MNIST CLassifieer from my_img2num.py, type the following commands:
```
> from my_img2num.py import MyImg2Num
> MNISTNN = MyImg2Num()
> #To display output for specified input
> print MNISTNN.forward(<input_tensor>)
```

#### Class Description
Python Class **NeuralNetwork**():
This is the class that implements an n-layer neural network. 
Its methods are described below:

- _ _ init _ _([in0, h1,h2,...,out]): (Constructor)

*in0* - size of input layer
*hn* - size of nth layer
*out* - size of output layer

- [DoubleTensor] **getLayer**(layer_no):
Get the layer corresponding to the index layer_no

*layer_no* - index of the layer

*Returns*: 
Tensor for the nth layer

- [DoubleTensor] **forward**(input_tensor): 
Method for forward propagation pass

*input_tensor* - the input for which NN output is to be generated 

*Returns*:
Tensor for the output

 -[DoubleTensor] **backward**(target):
 Method for backward propagation pass

*target* - the input for which NN output is to be generated 

*Returns*:
Value for dE/dT

- [DoubleTensor] **updateParams**(eta):
*input_tensor* - learning parameter

*Returns*:
Update weight matrix theta

Python Class **MyImg2Num**():
This is the class that implements an n-layer neural network for MNIST Classification. 
Its methods are described below:

- _ _ **init** _ _(): (Constructor)


[FloatTensor] **forward**(x):
x - 28 x 28 input float tensor corresponding to an MNIST image

*Returns*:
Tensor output - prediction label

-[] **train**(): (Function for training the Neural Network)
*Args*- None
Returns - None

(The class for **NNImg2Num** is similar to the **MyImg2Num** class described above)


### **Results**:

The following are the values for the computation time and MSE error for the two Neural Networks:

##### MyImg2Num():
Error:         	      0.03596  s

Computation Time:     18.62304 s

##### MyImg2Num():
Error:         		  0.05001 s

Computation Time:     2.63544 s

While the MSE error value is simlar for both the classes, we see that the NN implemented using nn_img2num.py has a faster response. 

