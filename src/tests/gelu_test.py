import numpy as np
from src.layers import GELU


def numerical_gradient(gelu, x, grad_output, eps=1e-5):
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_neg = x.copy()

        x_pos[idx] += eps
        x_neg[idx] -= eps

        y_pos = gelu.forward(x_pos)
        y_neg = gelu.forward(x_neg)

        loss_pos = np.sum(y_pos * grad_output)
        loss_neg = np.sum(y_neg * grad_output)

        grad[idx] = (loss_pos - loss_neg) / (2 * eps)

    return grad


def test_gelu():
    np.random.seed(42)

    gelu = GELU()

    # 1. Forward shape test
    x = np.random.randn(2, 3, 4)
    y = gelu.forward(x)

    assert y.shape == x.shape
    print("Forward shape test passed")


    # 2. Known value test
    x_test = np.array([0.0])
    y_test = gelu.forward(x_test)

    assert np.allclose(y_test, 0.0)
    print("Known value test passed")


    # 3. Forward correctness test
    x = np.random.randn(2, 3, 4)

    expected = 0.5 * x * (
        1 + np.tanh(
            np.sqrt(2 / np.pi)
            * (x + 0.044715 * x**3)
        )
    )

    output = gelu.forward(x)

    assert np.allclose(output, expected)
    print("Forward correctness test passed")


    # 4. GELU behavior test
    x_behavior = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
    y_behavior = gelu.forward(x_behavior)

    # GELU(0) = 0
    assert np.isclose(y_behavior[2], 0.0)

    # For large positive x, GELU(x) ≈ x
    assert np.isclose(y_behavior[-1], 10.0, atol=1e-4)

    # For large negative x, GELU(x) ≈ 0
    assert np.isclose(y_behavior[0], 0.0, atol=1e-4)

    # GELU is not symmetric
    assert not np.isclose(y_behavior[1], -y_behavior[3])

    print("GELU behavior tests passed")


    # 5. Backward shape test
    x = np.random.randn(2, 3, 4)
    grad_output = np.random.randn(*x.shape)

    gelu.forward(x)
    grad_input = gelu.backward(grad_output)

    assert grad_input.shape == x.shape
    print("Backward shape test passed")


    # 6. Numerical gradient test
    gelu.forward(x)
    analytical_grad = gelu.backward(grad_output)

    numerical_grad = numerical_gradient(
        gelu,
        x,
        grad_output
    )

    max_error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print("Gradient max error:", max_error)

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )

    print("Backward gradient test passed")


    # 7. Numerical gradient test with multiple shapes
    shapes = [
        (5,),
        (2, 3),
        (2, 3, 4)
    ]

    for shape in shapes:
        x = np.random.randn(*shape)
        grad_output = np.random.randn(*shape)

        gelu.forward(x)
        analytical_grad = gelu.backward(grad_output)

        numerical_grad = numerical_gradient(
            gelu,
            x,
            grad_output
        )

        max_error = np.max(
            np.abs(analytical_grad - numerical_grad)
        )

        print(
            f"Shape {shape} gradient max error: {max_error}"
        )

        assert analytical_grad.shape == x.shape

        assert np.allclose(
            analytical_grad,
            numerical_grad,
            atol=1e-5
        )

    print("Multiple-shape gradient tests passed")

    print("\nAll GELU tests passed!")


if __name__ == "__main__":
    test_gelu()