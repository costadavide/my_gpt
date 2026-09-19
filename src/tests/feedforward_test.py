import numpy as np
from src.layers import FeedForward


def numerical_gradient_input(layer, x, grad_output, eps=1e-5):
    grad = np.zeros_like(x)

    for idx in np.ndindex(x.shape):
        x_pos = x.copy()
        x_neg = x.copy()

        x_pos[idx] += eps
        x_neg[idx] -= eps

        y_pos = layer.forward(x_pos)
        y_neg = layer.forward(x_neg)

        loss_pos = np.sum(y_pos * grad_output)
        loss_neg = np.sum(y_neg * grad_output)

        grad[idx] = (loss_pos - loss_neg) / (2 * eps)

    return grad


def numerical_gradient_parameter(
    layer,
    x,
    grad_output,
    parameter,
    eps=1e-5
):
    grad = np.zeros_like(parameter)

    for idx in np.ndindex(parameter.shape):
        original_value = parameter[idx]

        parameter[idx] = original_value + eps
        y_pos = layer.forward(x)
        loss_pos = np.sum(y_pos * grad_output)

        parameter[idx] = original_value - eps
        y_neg = layer.forward(x)
        loss_neg = np.sum(y_neg * grad_output)

        parameter[idx] = original_value

        grad[idx] = (loss_pos - loss_neg) / (2 * eps)

    return grad


def test_feedforward():
    np.random.seed(42)

    input_dim = 4
    B = 2
    T = 3

    layer = FeedForward(input_dim)

    x = np.random.randn(B, T, input_dim)

    # 1. Initialization tests
    assert layer.hidden_dim == 4 * input_dim

    assert layer.linear1.weights.shape == (
        input_dim,
        4 * input_dim
    )

    assert layer.linear1.bias.shape == (
        4 * input_dim,
    )

    assert layer.linear2.weights.shape == (
        4 * input_dim,
        input_dim
    )

    assert layer.linear2.bias.shape == (
        input_dim,
    )

    print("Initialization tests passed")


    # 2. Forward shape test
    output = layer.forward(x)

    assert output.shape == (B, T, input_dim)

    print("Forward shape test passed")


    # 3. Backward shape test
    grad_output = np.random.randn(*output.shape)

    layer.forward(x)
    grad_input = layer.backward(grad_output)

    assert grad_input.shape == x.shape

    assert layer.linear1.weights_grad.shape == \
        layer.linear1.weights.shape

    assert layer.linear1.bias_grad.shape == \
        layer.linear1.bias.shape

    assert layer.linear2.weights_grad.shape == \
        layer.linear2.weights.shape

    assert layer.linear2.bias_grad.shape == \
        layer.linear2.bias.shape

    print("Backward shape tests passed")


    # 4. Input gradient check
    layer.forward(x)
    analytical_grad_input = layer.backward(
        grad_output
    )

    numerical_grad_input = numerical_gradient_input(
        layer,
        x,
        grad_output
    )

    input_error = np.max(
        np.abs(
            analytical_grad_input
            - numerical_grad_input
        )
    )

    print("Input gradient max error:", input_error)

    assert np.allclose(
        analytical_grad_input,
        numerical_grad_input,
        atol=1e-5
    )

    print("Input gradient test passed")


    # 5. Linear1 weight gradient check
    layer.forward(x)
    layer.backward(grad_output)

    analytical_grad = layer.linear1.weights_grad.copy()

    numerical_grad = numerical_gradient_parameter(
        layer,
        x,
        grad_output,
        layer.linear1.weights
    )

    error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print(
        "Linear1 weight gradient max error:",
        error
    )

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )

    print("Linear1 weight gradient test passed")


    # 6. Linear1 bias gradient check
    layer.forward(x)
    layer.backward(grad_output)

    analytical_grad = layer.linear1.bias_grad.copy()

    numerical_grad = numerical_gradient_parameter(
        layer,
        x,
        grad_output,
        layer.linear1.bias
    )

    error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print(
        "Linear1 bias gradient max error:",
        error
    )

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )

    print("Linear1 bias gradient test passed")


    # 7. Linear2 weight gradient check
    layer.forward(x)
    layer.backward(grad_output)

    analytical_grad = layer.linear2.weights_grad.copy()

    numerical_grad = numerical_gradient_parameter(
        layer,
        x,
        grad_output,
        layer.linear2.weights
    )

    error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print(
        "Linear2 weight gradient max error:",
        error
    )

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )

    print("Linear2 weight gradient test passed")


    # 8. Linear2 bias gradient check
    layer.forward(x)
    layer.backward(grad_output)

    analytical_grad = layer.linear2.bias_grad.copy()

    numerical_grad = numerical_gradient_parameter(
        layer,
        x,
        grad_output,
        layer.linear2.bias
    )

    error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print(
        "Linear2 bias gradient max error:",
        error
    )

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        atol=1e-5
    )

    print("Linear2 bias gradient test passed")

    print("\nAll FeedForward tests passed!")


if __name__ == "__main__":
    test_feedforward()