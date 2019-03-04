# Digit-Recognition

## Training Accuracy: 96.7%

Implementing the backpropagation algorithm for a neural network, and apply it to hand-written digit recognition. 

## Some theory about the project...

### Regularized Cost Function
The cost function for a neural network with 3 layers is given by:
![image](https://user-images.githubusercontent.com/41659296/53733136-055f1780-3e4e-11e9-9859-11ecc124bd26.png)

### Sigmoid Gradient
The backpropagation algorithm requires the gradient of the sigmoid function to be computed.
Sigmoid Function:
![image](https://user-images.githubusercontent.com/41659296/53733209-5111c100-3e4e-11e9-872b-da9b87d5c9d2.png)

Sigmoid Gradient:
![image](https://user-images.githubusercontent.com/41659296/53733231-61c23700-3e4e-11e9-9dd1-bc44b8a81eb2.png)

### Backpropagation Implementation
Gradients with Regularization:
![image](https://user-images.githubusercontent.com/41659296/53733447-03e21f00-3e4f-11e9-920c-d1bae40660ff.png)

Where delta equals:
![image](https://user-images.githubusercontent.com/41659296/53733500-283dfb80-3e4f-11e9-8cd3-9bf1825f2e1e.png)
and
![image](https://user-images.githubusercontent.com/41659296/53733537-4572ca00-3e4f-11e9-9459-3ff30cf047d1.png)

Used to compute:
![image](https://user-images.githubusercontent.com/41659296/53733574-60ddd500-3e4f-11e9-85ff-779fc46d6aa1.png)

### Gradient Checking
An import part of using a neural network is numerically validating your implementation to confirm accurate gradient results.
This numerical function uses an alternative approach to compute the derivatives:
![image](https://user-images.githubusercontent.com/41659296/53733706-d649a580-3e4f-11e9-9d82-84c4ce080f63.png)

Which gives:
![image](https://user-images.githubusercontent.com/41659296/53733717-dfd30d80-3e4f-11e9-8751-4d6449888adb.png)

The gradient check is used for a few test values, confirms the backpropagation algorithm implementation is correct, then turns off as it is computationally inefficient.

This implementation then uses fmincg to learn the parameter values.



Based off of Stanford's Machine Learning Course taught by professor Andrew Ng.
