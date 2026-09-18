
from src.layers import LayerNorm
import numpy as np

np.random.seed(42)

B, T, D = 2, 3, 4
x = np.random.randn(B, T, D)
grad_output = np.random.randn(B, T, D)

layer = LayerNorm(D)

# Use nontrivial parameters
layer.gamma = np.random.randn(D)
layer.beta = np.random.randn(D)

# ============================================================
# 1. Forward shape
# ============================================================

output = layer.forward(x)

assert output.shape == (B, T, D)
assert layer.mean.shape == (B, T, 1)
assert layer.variance.shape == (B, T, 1)
assert layer.x_hat.shape == (B, T, D)

print("Forward shape tests passed")


# ============================================================
# 2. Normalization properties
# ============================================================

x_hat_mean = np.mean(layer.x_hat, axis=-1)
x_hat_variance = np.var(layer.x_hat, axis=-1)

np.testing.assert_allclose(
    x_hat_mean,
    np.zeros((B, T)),
    atol=1e-10
)

# Because of epsilon, x_hat variance is close to, but not exactly, 1
expected_variance = layer.variance / (layer.variance + layer.eps)

np.testing.assert_allclose(
    x_hat_variance,
    expected_variance[..., 0],
    atol=1e-10
)

print("Normalization tests passed")


# ============================================================
# 3. Compare forward pass with a direct NumPy calculation
# ============================================================

mean_reference = np.mean(x, axis=-1, keepdims=True)
variance_reference = np.var(x, axis=-1, keepdims=True)

x_hat_reference = (
    (x - mean_reference)
    / np.sqrt(variance_reference + layer.eps)
)

output_reference = (
    layer.gamma * x_hat_reference + layer.beta
)

np.testing.assert_allclose(output, output_reference, atol=1e-10)

print("Forward-value test passed")


# ============================================================
# 4. Backward shapes
# ============================================================

layer.forward(x)
grad_input = layer.backward(grad_output)

assert grad_input.shape == x.shape
assert layer.gamma_grad.shape == layer.gamma.shape
assert layer.beta_grad.shape == layer.beta.shape

print("Backward shape tests passed")


# Save analytical gradients before calling forward again
analytical_input_grad = grad_input.copy()
analytical_gamma_grad = layer.gamma_grad.copy()
analytical_beta_grad = layer.beta_grad.copy()


# ============================================================
# Helper scalar loss
# ============================================================

def compute_loss(layer, x, grad_output):
    output = layer.forward(x)
    return np.sum(output * grad_output)


# ============================================================
# 5. Numerical input-gradient check
# ============================================================

h = 1e-5
numerical_input_grad = np.zeros_like(x)

for index in np.ndindex(x.shape):
    x_positive = x.copy()
    x_negative = x.copy()

    x_positive[index] += h
    x_negative[index] -= h

    loss_positive = compute_loss(layer, x_positive, grad_output)
    loss_negative = compute_loss(layer, x_negative, grad_output)

    numerical_input_grad[index] = (
        loss_positive - loss_negative
    ) / (2 * h)

input_error = np.max(
    np.abs(analytical_input_grad - numerical_input_grad)
)

print("Input gradient max error:", input_error)

np.testing.assert_allclose(
    analytical_input_grad,
    numerical_input_grad,
    rtol=1e-5,
    atol=1e-6
)

print("Input gradient test passed")


# ============================================================
# 6. Numerical gamma-gradient check
# ============================================================

numerical_gamma_grad = np.zeros_like(layer.gamma)

for i in range(D):
    original_value = layer.gamma[i]

    layer.gamma[i] = original_value + h
    loss_positive = compute_loss(layer, x, grad_output)

    layer.gamma[i] = original_value - h
    loss_negative = compute_loss(layer, x, grad_output)

    numerical_gamma_grad[i] = (
        loss_positive - loss_negative
    ) / (2 * h)

    layer.gamma[i] = original_value

gamma_error = np.max(
    np.abs(analytical_gamma_grad - numerical_gamma_grad)
)

print("Gamma gradient max error:", gamma_error)

np.testing.assert_allclose(
    analytical_gamma_grad,
    numerical_gamma_grad,
    rtol=1e-5,
    atol=1e-6
)

print("Gamma gradient test passed")


# ============================================================
# 7. Numerical beta-gradient check
# ============================================================

numerical_beta_grad = np.zeros_like(layer.beta)

for i in range(D):
    original_value = layer.beta[i]

    layer.beta[i] = original_value + h
    loss_positive = compute_loss(layer, x, grad_output)

    layer.beta[i] = original_value - h
    loss_negative = compute_loss(layer, x, grad_output)

    numerical_beta_grad[i] = (
        loss_positive - loss_negative
    ) / (2 * h)

    layer.beta[i] = original_value

beta_error = np.max(
    np.abs(analytical_beta_grad - numerical_beta_grad)
)

print("Beta gradient max error:", beta_error)

np.testing.assert_allclose(
    analytical_beta_grad,
    numerical_beta_grad,
    rtol=1e-5,
    atol=1e-6
)

print("Beta gradient test passed")


# ============================================================
# 8. Input-gradient invariant
# ============================================================

# LayerNorm is unaffected by adding the same value to every
# feature of a token. Therefore, the input gradients should
# sum to approximately zero along the feature dimension.

np.testing.assert_allclose(
    np.sum(analytical_input_grad, axis=-1),
    np.zeros((B, T)),
    atol=1e-10
)

print("Input-gradient invariant passed")

print("\nAll LayerNorm tests passed!")