# AIM OF THE PROJECT

The aim of this project is to implement a mini-GPT from scratch using only `numpy`. Each class and function is also documented here to clearly explain how it works and make the project easier to understand and revisit.

Currently, the focus is on implementing all the layers of the decoder-only transformer. The training pipeline will be added afterward.

# TO DO

1. Tokenizer

# IMPLEMENTED

## SoftMax

`SoftMax.forward()` convert the input values into values between 0 and 1 that sum to 1 along the specified axis(last one by default).

It computes:

$$
\sigma(x)_i = \frac{e^{x_i}}{\sum_{j=0}^{N-1} e^{x_j}}
$$

where $N$ is the `input_dim`.

For numerical stability, the maximum value of $x$ is subtracted before computing the exponentials:

$$
\sigma(x)_i = \frac{e^{{x_i}-\max(x)}}{\sum_{j=0}^N e^{{x_j}-\max(x)}}
$$

This does not change the softmax result but prevents very large exponentials from causing overflows.

The output has the same shape as the input, so `output_dim` is equal to `input_dim`.

`SoftMax.backward()` computes the gradient of the loss wrt the input:

$$
\frac{\partial L}{\partial x}
$$

To get the gradient the first step is to derive the partial derivative of the output $y$ wrt to the input $x$. For each $y_i$ and each $x_j$ the $\frac{\partial y_i}{\partial x_j}$ is:

**If $i \neq j$:**

```math
\frac{\partial }{\partial x_j}\frac{e^{x_i}}{e^{x_j}+\sum_{k\neq j} e^{x_k}} = -\frac{e^{x_i}\cdot e^{x_j}}{(\sum_{k}e^{x_k})^2} = - y_i \cdot y_j
```

**If $i = j$:**

```math
\frac{\partial }{\partial x_j}\frac{e^{x_i}}{e^{x_i}+\sum_{k\neq i} e^{x_k}}= \frac{e^{x_i}(\sum_{k}e^{x_k}-e^{x_i})}{(\sum_{k}e^{x_k})^2} = y_i(1-y_i)
```

Thus each output element $y_i$ depends on all input elements $x_j$. Considering the two cases above the Jacobian can be calculated, being $y_i(1-y_i)$ along the main diagonal and $-y_i \cdot y_j$ in the rest of the matrix.

The last step is computing $\frac{\partial L}{\partial x}$:

```math
\frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
```

The first part is given as input to `SoftMax.backward()`. The second part can be calculated as seen above, selecting the correct case depending on whether $i=j$ or not.

By combining everything together the final formula for computing the gradient is:

```math
\boxed{
\frac{\partial L}{\partial x_i}
=
y_i
\left(
\frac{\partial L}{\partial y_i}
-
\sum_j
y_j\frac{\partial L}{\partial y_j}
\right)
}
```

Using `numpy` this can be done efficiently, without having to specifically build the Jacobian matrix. In actual code this translates to:

`grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=self.axis, keepdims=True))`

This computes the whole gradient using vectorized operations, without explicitly constructing the Jacobian matrix, making the computation more memory and computationally efficient.

## Linear

`Linear.forward()` takes as input a tensor $x$ and applies an affine transformation to its last dimension. Each output element is then computed as a weighted sum of the input elements, plus a bias.

The operation is defined as:

$$
y = xW + b
$$

where: $x$ is the input tensor, $W$ is the weight matrix of dimension (input_dim, output_dim), $b$ is the bias vector, and $y$ is the output tensor.

For each output element $y_j$, we have the following:

$$
y_j = \sum_{i=0}^{input\_dim-1} x_i W_{ij} + b_j
$$

For simplicity, the following backward pass derivations assume a single input vector. The extension to batch and sequence dimensions is discussed at the end.

`Linear.backward()` computes the gradients of the loss wrt. the input $x$, the weights $W$, and the bias $b$.

Let's start with the $\textbf{input}$. The partial derivative of $y_j$ wrt. $x_i$ is:

$$
\frac{\partial y_j}{\partial x_i} = \frac{\partial \sum_k x_k W_{kj} + b_j}{\partial x_i} = W_{ij}
$$

Applying the chain rule:

$$
\frac{\partial L}{\partial x_i}=\sum_j\frac{\partial L}{\partial y_j}\frac{\partial y_j}{\partial x_i} = \sum_j\frac{\partial L}{\partial y_j} \cdot W_{ij}
$$

We can exploit `numpy` and compute this with just one matrix multiplication as:

$$
\boxed{
\frac{\partial L}{\partial x}=\frac{\partial L}{\partial y} \cdot W^T
}
$$

Now let's consider the $\textbf{weights}$. The partial derivative of $y_j$ wrt. $W_ij$ is:

$$
\frac{\partial y_j}{\partial W_{ij}} = \frac{\partial \sum_k x_k W_{kj} + b_j}{\partial W_{ij}} = x_i
$$

Again using the chain rule we get:

$$
\frac{\partial L}{\partial W_{ij}}=\frac{\partial L}{\partial y_j}\frac{\partial y_j}{\partial W_{ij}} = \frac{\partial L}{\partial y_j} \cdot x_i
$$

Again using `numpy` this can be calculated just using matrix multiplication:

$$
\boxed{
\frac{\partial L}{\partial W}=x^T \cdot \frac{\partial L}{\partial y}
}
$$

At last we consider the $\textbf{bias}$. The partial derivative of $y_j$ wrt. $b_j$ is:

$$
\frac{\partial y_j}{\partial b_j} = \frac{\partial \sum_i x_i W_{ij} + b_j }{\partial b_j} = 1
$$

Using the chain rule we get:

$$
\boxed{
\frac{\partial L}{\partial b}=\frac{\partial L}{\partial y}
}
$$

$\textbf{Note on batch and sequence dimensions}$

The derivations above assume a single input vector of shape (input_dim,). However, in the GPT-2 architecture, the input to the Linear layer has shape (B, T, input_dim), where B represents the batch size and T the sequence length.

The same linear transformation is applied independently to every token in every batch. Consequently:

- The input gradient is computed independently for each token using the same weight matrix.
- The weight gradient is obtained by summing the contributions from all tokens across all batches.
- The bias gradient is obtained by summing the output gradients across the batch and sequence dimensions.

Using `numpy`, the three gradients can be calculated as:

`grad_input = grad_output @ self.weights.T`

`weights_grad = self.input.reshape(-1, self.input_dim).T @ grad_output.reshape(-1, self.output_dim)`

`bias_grad = np.sum(grad_output, axis=(0, 1))`

The batch and sequence dimensions are combined when computing the weight gradient to allow the use of a single matrix multiplication.

## Layer Norm

## GELU

## FeedForward

## Embedding Layer

## MultiHead Attention
