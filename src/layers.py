import numpy as np 



class Linear: 
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = np.random.randn(input_dim, output_dim) / np.sqrt(input_dim)
        self.bias = np.zeros(output_dim) 
        self.input = None 
        self.weights_grad = None
        self.bias_grad = None 

    def forward(self, x): 
        self.input = x
        y = x @ self.weights + self.bias # x.shape = (B, T, input_dim), weights.shape = (input_dim, output_dim), bias.shape = (output_dim, )
        return y # y.shape = (B, T, output_dim)

    def backward(self, grad_output): 
        grad_input = grad_output @ self.weights.T # grad_output.shape = (B, T, output_dim), weights.shape = (input_dim, output_dim)
        self.bias_grad = np.sum(grad_output, axis=(0, 1)) # bias_grad.shape = (output_dim, )
        self.weights_grad = self.input.reshape(-1, self.input_dim).T @ grad_output.reshape(-1, self.output_dim)# x.shape = (B, T, input_dim), grad_output.shape = (B, T, output_dim), weights_grad.shape = (input_dim, output_dim)
        return grad_input # grad_input.shape = (B, T, input_dim)

class LayerNorm: 
    def __init__(self, input_dim, eps=1e-5):
        self.input_dim = input_dim
        self.eps = eps  
        self.gamma = np.ones(self.input_dim) 
        self.beta = np.zeros(self.input_dim)
        self.input = None # x.shape = (B, T, input_dim)
        self.output = None # output.shape = (B, T, input_dim)
        self.mean = None 
        self.variance = None 
        self.x_hat = None 

        # keep gradients for backward pass
        self.gamma_grad = None
        self.beta_grad = None


    def forward(self, x): 
        self.input = x 
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x-self.mean) / np.sqrt(self.variance + self.eps)
        y = self.gamma * self.x_hat + self.beta 
        self.output = y
        return y 

    def backward(self, grad_output): 
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
        grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=self.axis, keepdims=True))
        return grad_input 

class EmbeddingLayer: 
    def __init__(self):
        pass 

class MultiHeadAttention:
    def __init__(self):
        pass 