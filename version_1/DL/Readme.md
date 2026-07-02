### Difference between cost function and loss function?

### cost function vs loss function vs forward propogation vs backward propogation vs activation function


### activation function - tanh,relu,leaky relu


#### $ f(x)=\tanh (x) =\frac {e^x - e^{-x}} {e^x+e^{-x}}   $ $\tanh(x) = 2\sigma(2x) - 1$
- Disadvantage
1. vanishing gradient for deep neural network
2. computation


#### $ ReLU=\max(0,x)$
- Rectified Linear Unit
- If derivative of ReLU output is 1 => weight updating will happen

- Disadvantage
1. If derivative of ReLU output is 0 => dead neuron. Solved by leaky ReLU and parametric ReLU


#### ELU ( Exponential Linear Unit ):
#### Softmax Activation Function:
- $ Softmax=\frac{e^{y_i}}{\sum^{n}_{k=0} c^{y_k}} $ 
#### derivative output for sigmoid ( 0 to .25), tanh (0 to 1)

#### When activation function to use?
- For binary class classification sigmoid
- For multi class classification softmax
- Never used tanh or sigmoid because of gradient vanishing. Always used ReLU or it's variant.


#### Loss Function and Cost Function for classification.
- Classification:  
    - Cross Entropy: 
        - Binary Cross Entropy:
        - Categorical Cross Entropy: Multiclass classification
        - Sparse Categorical Entropy: 

#### Right Combination:

|Hidden Layers|O/P Layers (Activation Function )|Problem Statement|Loss Function|
|-------------|----------|-----------------|-------------|
|ReLU  or It's Variance     | Sigmoid  | Binary Classification | Binary Cross Entropy|
|ReLU or It's Variance|Softmax|Multiclass|Categorical/Sparse Cross Entropy|  
|ReLU or It's Variance|Linear|Regression|MSE,MAE,Huber Loss,RMSE|


## Optimizers
#### Gradient Descent:
#### Stochastic Gradient Descent (SGD):
- Advantages:
    1. Solve infrastructure and resource usage issue.
- Disadvantages:
    1. Time Complexity. 
    2. Convergence will take more time.
    3. Noise gets introduced
#### Mini batch SGD
- Advantages:
    1. Compare to SGD convergence speed will increase
    2. noise will be less than compared to SGD
    3. efficient resource usage
- Disadvantages:
    1. Noise still exist
#### SGD with momentum
- Advantages:
    1. 
- Disadvantages:
    1. 
#### Adagrad: Adaptive Gradiant Descent  
- As the convergence happen the learning rate should change
- Advantages:
    1. 
- Disadvantages:
    1. $\eta$ -> possibility to become zero.
#### RMS Prop
- Advantages:
    1. 
- Disadvantages:
    1. 
#### Adam Optimizers
- Advantages:
    1. Best all of the optimizers.
- Disadvantages:
    1. 


### Weight Initiating Techniques
- Weights should be small
- Weights should not be same
- Weights should have good variance

#### Uniform Distribution:
#### Xavier/Glorot initialization
#### Kaiming He Initialization

### CNN - Convolutional NN
### ANN - Artificial NN
### RNN - Recurrent NN