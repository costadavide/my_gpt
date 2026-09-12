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
        grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=self.axis, keepdims=True))
        return grad_input 

class EmbeddingLayer: 
    def __init__(self):
        pass 

class MultiHeadAttention:
    def __init__(self):
        pass 