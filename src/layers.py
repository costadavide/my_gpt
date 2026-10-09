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
        self.c = None
        self.r = None 

        # keep gradients for backward pass
        self.gamma_grad = None
        self.beta_grad = None


    def forward(self, x): 
        self.input = x 
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.variance = np.var(x, axis=-1, keepdims=True)
        self.c = x - self.mean
        self.r = 1/np.sqrt(self.variance + self.eps)
        self.x_hat = self.c * self.r
        y = self.gamma * self.x_hat + self.beta 
        self.output = y
        return y 

    def backward(self, grad_output): 
        self.gamma_grad = np.sum(grad_output * self.x_hat, axis=(0, 1)) # gamma_grad.shape = (input_dim, )
        self.beta_grad = np.sum(grad_output, axis=(0, 1)) # beta_grad.shape = (input_dim, )
        grad_x_hat = grad_output * self.gamma # grad_x_hat.shape = (B, T, input_dim)
        grad_c = grad_x_hat * self.r # grad_c.shape = (B, T, input_dim)
        grad_r = np.sum(grad_x_hat * self.c, axis=-1, keepdims=True) # grad_r.shape = (B, T, 1)
        grad_variance = grad_r * (-0.5) * (self.variance + self.eps)**(-1.5) # grad_variance.shape = (B, T, 1)
        grad_c += grad_variance * 2 * self.c / self.input_dim # grad_c.shape = (B, T, input_dim)
        grad_mean = -np.sum(grad_c, axis=-1, keepdims=True) # grad_mean.shape = (B, T, 1)
        grad_input = grad_c + grad_mean / self.input_dim # grad_input.shape = (B, T, input_dim)

        return grad_input

class GELU:
    def __init__(self): 
        self.x = None 
    def forward(self, x):
        self.x = x 
        gelu = 0.5 * x * (1 +np.tanh(np.sqrt(2/np.pi) * (x+0.044715*x**3)))
        return gelu 
    
    def backward(self, grad_output): 
        x = self.x 
        u = np.sqrt(2/np.pi)*(x+0.044715*x**3)
        tanh_u = np.tanh(u)
        grad_input = grad_output * (0.5 * (1 + tanh_u) + x/2 * (1 - tanh_u**2) * np.sqrt(2/np.pi) * (1 + 3*0.044715*x**2))
        return grad_input

class FeedForward: 
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.hidden_dim = 4 * input_dim # default in GPT2 architecture
        self.linear1 = Linear(self.input_dim, self.hidden_dim)
        self.gelu = GELU()
        self.linear2 = Linear(self.hidden_dim, self.input_dim)

    def forward(self, x):
        x = self.linear1.forward(x)
        x = self.gelu.forward(x)
        x = self.linear2.forward(x)
        return x 

    def backward(self, grad_output):
        grad = self.linear2.backward(grad_output)
        grad = self.gelu.backward(grad)
        grad = self.linear1.backward(grad)
        return grad
        

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
    def __init__(self, num_embeddings, embedding_dim):
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embedding_matrix = np.random.randn(num_embeddings, embedding_dim) # embedding_matrix.shape = (num_embeddings, embedding_dim)
        # consider changing the initializaton, as in GPT2 did something different 
        self.input = None # input.shape = (B, T) (these are indices of the tokens in the vocabulary)
        self.embedding_grad = None

    def forward(self, x):
        self.input = x
        return self.embedding_matrix[x]

    def backward(self, grad_output):
        x = self.input 
        # for each index in the input, accumulate the gradient for the corresponding embedding vector
        self.embedding_grad = np.zeros_like(self.embedding_matrix)
        np.add.at(self.embedding_grad, x, grad_output)
        # here no need to return anything 


class MultiHeadAttention:
    def __init__(self, input_dim, num_heads):
        assert input_dim % num_heads == 0 # ensure that the input dimension is divisible by the number of heads
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads
        self.Q = Linear(input_dim, input_dim)
        self.K = Linear(input_dim, input_dim)
        self.V = Linear(input_dim, input_dim)
        self.output_projection = Linear(input_dim, input_dim)
        self.softmax = SoftMax(axis=-1)
        self.Q_out = None
        self.K_out = None
        self.V_out = None
        self.attention_weights = None
        self.scale = 1 / np.sqrt(self.head_dim)

    def forward(self, x):
        self.Q_out = self.Q.forward(x) # shape = (B, T, input_dim)
        self.K_out = self.K.forward(x) # shape = (B, T, input_dim)
        self.V_out = self.V.forward(x) # shape = (B, T, input_dim)
        B, T, _ = x.shape
        # reshape Q, K, V to (B, num_heads, T, head_dim)
        self.Q_out = self.Q_out.reshape((B, T, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        self.K_out = self.K_out.reshape((B, T, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))
        self.V_out = self.V_out.reshape((B, T, self.num_heads, self.head_dim)).transpose((0, 2, 1, 3))

        scores = self.Q_out @ self.K_out.transpose((0, 1, 3, 2)) * self.scale # shape = (B, num_heads, T, T)
        causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1) # can do (T, T) as numpy will broadcast mask across all batches and heads
        scores = np.where(causal_mask, -np.inf, scores) # make -inf the attention scores of tokens after one considered (so when apply softmax result is 0)

        self.attention_weights = self.softmax.forward(scores) # attention_weights.shape = (B, num_heads, T, T), softmax acts only on last dimension

        head_outputs = self.attention_weights @ self.V_out # head_outputs.shape(B, num_heads, T, head_dim)
        concat_heads = head_outputs.transpose((0, 2, 1, 3)).reshape((B, T, self.input_dim)) # concat_heads.shape = (B, T, input_dim), concatenate heads together

        output = self.output_projection.forward(concat_heads) # output.shape = (B, T, input_dim)

        return output

    def backward(self, grad_output):
        grad_projection = self.output_projection.backward(grad_output) # shape = (B, T, input_dim)
        grad_concat_heads = grad_projection.reshape((grad_projection.shape[0], grad_projection.shape[1], self.num_heads, self.head_dim)) # shape = (B, T, num_heads, head_dim)
        grad_head_outputs = grad_concat_heads.transpose((0, 2, 1, 3)) # shape = (B, num_heads, T, head_dim)

        grad_attention_weights = grad_head_outputs @ self.V_out.transpose((0, 1, 3, 2)) # shape = (B, num_heads, T, T)
        grad_V_out = self.attention_weights.transpose((0, 1, 3, 2)) @ grad_head_outputs # shape = (B, num_heads, T, head_dim)

        grad_scores = self.softmax.backward(grad_attention_weights) # shape = (B, num_heads, T, T)

        grad_Q_out = grad_scores @ self.K_out * self.scale # shape = (B, num_heads, T, head_dim)
        grad_K_out = grad_scores.transpose((0, 1, 3, 2)) @ self.Q_out * self.scale # shape = (B, num_heads, T, head_dim)

        B, _, T, _ = grad_V_out.shape

        grad_V_out = grad_V_out.transpose((0, 2, 1, 3)).reshape((B, T, self.input_dim))
        grad_Q_out = grad_Q_out.transpose((0, 2, 1, 3)).reshape((B, T, self.input_dim))
        grad_K_out = grad_K_out.transpose((0, 2, 1, 3)).reshape((B, T, self.input_dim))

        grad_Q = self.Q.backward(grad_Q_out)
        grad_V = self.V.backward(grad_V_out)
        grad_K = self.K.backward(grad_K_out)

        grad_input = grad_Q + grad_V + grad_K
        return grad_input
