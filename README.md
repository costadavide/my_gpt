# TO DO

1. Implementation of the FeedNeuralNetwork layer
2. Implementation of the Embedding Layer
3. Implementation of the MultiHeadAttention layer
4. Backward pass for LayerNorm

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

- if $i\neq j$ :

  $$
  \frac{\partial }{\partial x_j}\frac{e^{x_i}}{e^{x_j}+\sum_{k\neq j} e^{x_k}} = -\frac{e^{x_i}\cdot e^{x_j}}{(\sum_{k}e^{x_k})^2} = - y_i \cdot y_j
  $$

- if $i = j$:

  $$
  \frac{\partial }{\partial x_j}\frac{e^{x_i}}{e^{x_i}+\sum_{k\neq i} e^{x_k}}= \frac{e^{x_i}(\sum_{k}e^{x_k}-e^{x_i})}{(\sum_{k}e^{x_k})^2} = y_i(1-y_i)
  $$

Thus each output element $y_i$ depends on all input elements $x_j$. Considering the two cases above the Jacobian can be calculated, being $y_i(1-y_i)$ along the main diagonal and $-y_i \cdot y_j$ in the rest of the matrix.

The last step is computing $\frac{\partial L}{\partial x}$:

$$
\frac{\partial L}{\partial x_i} = \sum_{j} \frac{\partial L}{\partial y_j} \cdot \frac{\partial y_j}{\partial x_i}
$$

The first part is given as input to `SoftMax.backward()`. The second part can be calculated as seen above, selecting the correct case depending on whether $i=j$ or not.

By combining everything together the final formula for computing the gradient is:

$$
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
$$

Using `numpy` this can be done efficiently, without having to specifically build the Jacobian matrix. In actual code this translates to:

`grad_input = self.output * (grad_output - np.sum(self.output * grad_output, axis=self.axis, keepdims=True))`

This computes the whole gradient using vectorized operations, without explicitly constructing the Jacobian matrix, making the computation more memory and computationally efficient.

## Linear

## Layer Norm
