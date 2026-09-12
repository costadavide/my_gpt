from src.layers import SoftMax
import numpy as np 

np.random.seed(0)

# Example multidimensional input
x = np.random.randn(2, 3, 4)
grad_output = np.random.randn(2, 3, 4)

softmax = SoftMax(axis=-1)

# -------------------------
# Forward tests
# -------------------------

y = softmax.forward(x)

print("Output shape:", y.shape)
print("Sums along softmax axis:")
print(np.sum(y, axis=-1))

assert y.shape == x.shape
assert np.allclose(np.sum(y, axis=-1), 1.0)

# -------------------------
# Analytical gradient
# -------------------------

softmax.forward(x)
analytical = softmax.backward(grad_output)

# -------------------------
# Numerical gradient
# -------------------------

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

    numerical[index] = (
        f(x_plus) - f(x_minus)
    ) / (2 * eps)

# -------------------------
# Compare
# -------------------------

print("\nAnalytical gradient:")
print(analytical)

print("\nNumerical gradient:")
print(numerical)

print("\nMaximum absolute difference:")
print(np.max(np.abs(analytical - numerical)))

assert np.allclose(
    analytical,
    numerical,
    atol=1e-5
)

print("\nAll tests passed!")