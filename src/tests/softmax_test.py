from src.layers import SoftMax
import numpy as np


def make_inputs():
    np.random.seed(0)
    # Example multidimensional input
    x = np.random.randn(2, 3, 4)
    grad_output = np.random.randn(2, 3, 4)
    softmax = SoftMax(axis=-1)
    return softmax, x, grad_output


# -------------------------
# Forward tests
# -------------------------

def test_forward():
    softmax, x, _ = make_inputs()

    y = softmax.forward(x)

    assert y.shape == x.shape
    assert np.allclose(np.sum(y, axis=-1), 1.0)


# -------------------------
# Analytical vs numerical gradient
# -------------------------

def test_backward():
    softmax, x, grad_output = make_inputs()

    softmax.forward(x)
    analytical = softmax.backward(grad_output)

    def f(x):
        y = softmax.forward(x)
        return np.sum(y * grad_output)

    eps = 1e-5
    numerical = np.zeros_like(x)

    for index in np.ndindex(x.shape):
        x_plus = x.copy()
        x_minus = x.copy()

        x_plus[index] += eps
        x_minus[index] -= eps

        numerical[index] = (f(x_plus) - f(x_minus)) / (2 * eps)

    assert np.allclose(analytical, numerical, atol=1e-5)
