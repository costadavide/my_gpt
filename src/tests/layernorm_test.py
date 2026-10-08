from src.layers import LayerNorm
import numpy as np


B, T, D = 2, 3, 4


def make_inputs():
    np.random.seed(42)
    x = np.random.randn(B, T, D)
    grad_output = np.random.randn(B, T, D)

    layer = LayerNorm(D)

    # Use nontrivial parameters
    layer.gamma = np.random.randn(D)
    layer.beta = np.random.randn(D)

    return layer, x, grad_output


def numerical_gradient(loss, param, h=1e-5):
    # Central differences: perturb param in place, then restore it.
    # The scalar loss is L = sum(output * grad_output).
    grad = np.zeros_like(param)

    for index in np.ndindex(param.shape):
        original_value = param[index]

        param[index] = original_value + h
        loss_positive = loss()

        param[index] = original_value - h
        loss_negative = loss()

        param[index] = original_value

        grad[index] = (loss_positive - loss_negative) / (2 * h)

    return grad


# ============================================================
# 1. Forward shape
# ============================================================

def test_forward_shapes():
    layer, x, _ = make_inputs()

    output = layer.forward(x)

    assert output.shape == (B, T, D)
    assert layer.mean.shape == (B, T, 1)
    assert layer.variance.shape == (B, T, 1)
    assert layer.x_hat.shape == (B, T, D)


# ============================================================
# 2. Normalization properties
# ============================================================

def test_normalization():
    layer, x, _ = make_inputs()

    layer.forward(x)

    x_hat_mean = np.mean(layer.x_hat, axis=-1)
    x_hat_variance = np.var(layer.x_hat, axis=-1)

    np.testing.assert_allclose(x_hat_mean, np.zeros((B, T)), atol=1e-10)

    # Because of epsilon, x_hat variance is close to, but not exactly, 1
    expected_variance = layer.variance / (layer.variance + layer.eps)

    np.testing.assert_allclose(
        x_hat_variance,
        expected_variance[..., 0],
        atol=1e-10
    )


# ============================================================
# 3. Compare forward pass with a direct NumPy calculation
# ============================================================

def test_forward_values():
    layer, x, _ = make_inputs()

    output = layer.forward(x)

    mean_reference = np.mean(x, axis=-1, keepdims=True)
    variance_reference = np.var(x, axis=-1, keepdims=True)

    x_hat_reference = (
        (x - mean_reference)
        / np.sqrt(variance_reference + layer.eps)
    )

    output_reference = layer.gamma * x_hat_reference + layer.beta

    np.testing.assert_allclose(output, output_reference, atol=1e-10)


# ============================================================
# 4. Backward shapes
# ============================================================

def test_backward_shapes():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == x.shape
    assert layer.gamma_grad.shape == layer.gamma.shape
    assert layer.beta_grad.shape == layer.beta.shape


# ============================================================
# 5. Numerical input-gradient check
# ============================================================

def test_input_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    analytical_input_grad = layer.backward(grad_output)

    numerical_input_grad = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), x
    )

    np.testing.assert_allclose(
        analytical_input_grad,
        numerical_input_grad,
        rtol=1e-5,
        atol=1e-6
    )


# ============================================================
# 6. Numerical gamma-gradient check
# ============================================================

def test_gamma_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    layer.backward(grad_output)

    # Save analytical gradient before calling forward again
    analytical_gamma_grad = layer.gamma_grad.copy()

    numerical_gamma_grad = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), layer.gamma
    )

    np.testing.assert_allclose(
        analytical_gamma_grad,
        numerical_gamma_grad,
        rtol=1e-5,
        atol=1e-6
    )


# ============================================================
# 7. Numerical beta-gradient check
# ============================================================

def test_beta_gradient():
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    layer.backward(grad_output)

    analytical_beta_grad = layer.beta_grad.copy()

    numerical_beta_grad = numerical_gradient(
        lambda: np.sum(layer.forward(x) * grad_output), layer.beta
    )

    np.testing.assert_allclose(
        analytical_beta_grad,
        numerical_beta_grad,
        rtol=1e-5,
        atol=1e-6
    )


# ============================================================
# 8. Input-gradient invariant
# ============================================================

def test_input_gradient_sums_to_zero():
    # LayerNorm is unaffected by adding the same value to every
    # feature of a token. Therefore, the input gradients should
    # sum to approximately zero along the feature dimension.
    layer, x, grad_output = make_inputs()

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    np.testing.assert_allclose(
        np.sum(grad_input, axis=-1),
        np.zeros((B, T)),
        atol=1e-10
    )
