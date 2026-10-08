from src.layers import Linear

import numpy as np


B = 2
T = 3
input_dim = 4
output_dim = 5


def make_inputs():
    np.random.seed(42)
    layer = Linear(input_dim, output_dim)
    x = np.random.randn(B, T, input_dim)
    grad_output = np.random.randn(B, T, output_dim)
    return layer, x, grad_output


def numerical_gradient(loss, param, eps=1e-5):
    # Central differences: perturb param in place, then restore it.
    # The scalar loss is L = sum(y * grad_output), so dL/dy = grad_output.
    grad = np.zeros_like(param)

    for index in np.ndindex(param.shape):
        original = param[index]

        param[index] = original + eps
        loss_plus = loss()

        param[index] = original - eps
        loss_minus = loss()

        param[index] = original

        grad[index] = (loss_plus - loss_minus) / (2 * eps)

    return grad


# --------------------------------------------------
# 1. Initialization tests
# --------------------------------------------------

def test_initialization():
    layer, _, _ = make_inputs()

    assert layer.weights.shape == (input_dim, output_dim)
    assert layer.bias.shape == (output_dim,)
    assert np.all(layer.bias == 0)


# --------------------------------------------------
# 2. Forward-pass tests
# --------------------------------------------------

def test_forward():
    layer, x, _ = make_inputs()

    y = layer.forward(x)

    assert y.shape == (B, T, output_dim)

    # Compare with explicit computation for one position
    manual_y = x[0, 0] @ layer.weights + layer.bias

    assert np.allclose(y[0, 0], manual_y)


# --------------------------------------------------
# 3. Analytical backward pass
# --------------------------------------------------

def test_backward_shapes():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == x.shape
    assert layer.weights_grad.shape == layer.weights.shape
    assert layer.bias_grad.shape == layer.bias.shape


# --------------------------------------------------
# 4. Numerical gradient with respect to INPUT
# --------------------------------------------------

def test_input_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    numerical = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), x
    )

    assert np.allclose(grad_input, numerical, atol=1e-5)


# --------------------------------------------------
# 5. Numerical gradient with respect to WEIGHTS
# --------------------------------------------------

def test_weights_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    layer.backward(grad_output)

    numerical = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), layer.weights
    )

    assert np.allclose(layer.weights_grad, numerical, atol=1e-5)


# --------------------------------------------------
# 6. Numerical gradient with respect to BIAS
# --------------------------------------------------

def test_bias_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    layer.backward(grad_output)

    numerical = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), layer.bias
    )

    assert np.allclose(layer.bias_grad, numerical, atol=1e-5)
