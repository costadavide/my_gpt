import numpy as np 



class Linear: 
    def __init__(self):
        pass 

class LayerNorm: 
    def __init__(self):
        pass 

class FeedForward: 
    def __init__(self):
        pass 

class SoftMax:
    def __init__(self, axis=-1):
        self.axis = axis
        self.output = None # need to store the output for the backward pass 

    def forward(self, x): 
        # compute softmax to get a probability distribution for the next token 
        # subtract the max value to avoid overflow in the exponential function
        shifted_x = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted_x)
        sigma = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        self.output = sigma
        return sigma

    def backward(self, grad_output): 
        # V_2: explicitly compute the Jacobian matrix (use numpy instead of for loops)
        J = np.zeros((self.output.shape[0], self.output.shape[0]))
        J = np.diag(self.output) - np.outer(self.output, self.output)

        grad_input = np.dot(J.T, grad_output)
        return grad_input 

class EmbeddingLayer: 
    def __init__(self):
        pass 

class MultiHeadAttention:
    def __init__(self):
        pass 


x = np.array([0.5, -1.0, 2.0])
grad_output = np.array([0.3, -0.7, 1.2])

softmax = SoftMax()

softmax.forward(x)
analytical = softmax.backward(grad_output)

def f(x):
    y = softmax.forward(x)
    return np.sum(y * grad_output)

eps = 1e-5
numerical = np.zeros_like(x)

for i in range(len(x)):
    x_plus = x.copy()
    x_minus = x.copy()

    x_plus[i] += eps
    x_minus[i] -= eps

    numerical[i] = (
        f(x_plus) - f(x_minus)
    ) / (2 * eps)

print("Analytical:", analytical)
print("Numerical: ", numerical)
print("Difference:", analytical - numerical)