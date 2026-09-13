from src.layers import Linear 

import numpy as np

np.random.seed(42)

# --------------------------------------------------
# Setup
# --------------------------------------------------

B = 2
T = 3
input_dim = 4
output_dim = 5

layer = Linear(input_dim, output_dim)

x = np.random.randn(B, T, input_dim)
grad_output = np.random.randn(B, T, output_dim)

# --------------------------------------------------
# 1. Initialization tests
# --------------------------------------------------

assert layer.weights.shape == (input_dim, output_dim)
assert layer.bias.shape == (output_dim,)

assert np.all(layer.bias == 0)

print("Initialization tests passed")


# --------------------------------------------------
# 2. Forward-pass tests
# --------------------------------------------------

y = layer.forward(x)

assert y.shape == (B, T, output_dim)

# Compare with explicit computation for one position
b = 0
t = 0

manual_y = x[b, t] @ layer.weights + layer.bias

assert np.allclose(
    y[b, t],
    manual_y
)

print("Forward tests passed")


# --------------------------------------------------
# 3. Analytical backward pass
# --------------------------------------------------

layer.forward(x)
grad_input = layer.backward(grad_output)

assert grad_input.shape == x.shape
assert layer.weights_grad.shape == layer.weights.shape
assert layer.bias_grad.shape == layer.bias.shape

print("Backward shape tests passed")


# --------------------------------------------------
# Scalar function used for numerical differentiation
#
# L = sum(y * grad_output)
#
# Therefore:
# dL/dy = grad_output
# --------------------------------------------------

def loss():
    y = layer.forward(x)
    return np.sum(y * grad_output)


eps = 1e-5


# --------------------------------------------------
# 4. Numerical gradient with respect to INPUT
# --------------------------------------------------

numerical_input_grad = np.zeros_like(x)

for index in np.ndindex(x.shape):

    original = x[index]

    x[index] = original + eps
    loss_plus = loss()

    x[index] = original - eps
    loss_minus = loss()

    x[index] = original

    numerical_input_grad[index] = (
        loss_plus - loss_minus
    ) / (2 * eps)


print(
    "Input gradient max error:",
    np.max(np.abs(
        grad_input - numerical_input_grad
    ))
)

assert np.allclose(
    grad_input,
    numerical_input_grad,
    atol=1e-5
)

print("Input gradient test passed")


# --------------------------------------------------
# 5. Numerical gradient with respect to WEIGHTS
# --------------------------------------------------

numerical_weights_grad = np.zeros_like(layer.weights)

for index in np.ndindex(layer.weights.shape):

    original = layer.weights[index]

    layer.weights[index] = original + eps
    loss_plus = loss()

    layer.weights[index] = original - eps
    loss_minus = loss()

    layer.weights[index] = original

    numerical_weights_grad[index] = (
        loss_plus - loss_minus
    ) / (2 * eps)


print(
    "Weight gradient max error:",
    np.max(np.abs(
        layer.weights_grad - numerical_weights_grad
    ))
)

assert np.allclose(
    layer.weights_grad,
    numerical_weights_grad,
    atol=1e-5
)

print("Weight gradient test passed")


# --------------------------------------------------
# 6. Numerical gradient with respect to BIAS
# --------------------------------------------------

numerical_bias_grad = np.zeros_like(layer.bias)

for index in np.ndindex(layer.bias.shape):

    original = layer.bias[index]

    layer.bias[index] = original + eps
    loss_plus = loss()

    layer.bias[index] = original - eps
    loss_minus = loss()

    layer.bias[index] = original

    numerical_bias_grad[index] = (
        loss_plus - loss_minus
    ) / (2 * eps)


print(
    "Bias gradient max error:",
    np.max(np.abs(
        layer.bias_grad - numerical_bias_grad
    ))
)

assert np.allclose(
    layer.bias_grad,
    numerical_bias_grad,
    atol=1e-5
)

print("Bias gradient test passed")


# --------------------------------------------------
# Final result
# --------------------------------------------------

print("\nAll Linear layer tests passed!")