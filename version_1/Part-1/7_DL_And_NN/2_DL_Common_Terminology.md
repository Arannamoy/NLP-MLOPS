
### Perceptron
A perceptron is the simplest type of neural network — just one neuron. It takes inputs, multiplies them by weights, adds a bias, and applies an activation function to make a decision (e.g., classify 0 or 1).
### Neural Network 
 A neural network is a collection of interconnected layers of perceptrons (neurons). Each layer transformsits inputs using weights, biases, and activation functions. Deep neural networks have multiple hidden layers and can model complex patterns.

### Hyperparameters 
 These are settings we configure before training a model. They are not learned from the data. 
### Learning Rate (η)
 This controls how much we adjust the weights after each training step. A learning rate that’s too high may overshoot the solution; too low may make training very slow.
### Training
 This is the process where the model learns patterns from data by updating weights based on errors between predicted and actual outputs.
### Backpropagation 
 Backpropagation is the algorithm used to update weights in a neural network. It calculates the gradient of the loss function with respect to each weight by applying the chain rule, allowing the model to learn from its mistakes.
### Inference 
 Inference is when the trained model is used to make predictions on new, unseen data.

### Activation Function 
This function adds non-linearity to the output of neurons, helping networks model complex patterns.

Common activation functions:

ReLU: Rectified Linear Unit
Sigmoid: squashes output between 0 and 1
Tanh: squashes output between -1 and 1
Step: Code given below

```py
def step(x):
    return 1 if x>=0 else 0
```
Perceptrons typically use a step function as the activation, but modern neural nets often use ReLU.

### Epoch 
 One epoch means one full pass over the entire training dataset. Multiple epochs are used so the model can keep refining its understanding.
