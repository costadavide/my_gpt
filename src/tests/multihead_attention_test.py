import numpy as np

from src.layers import MultiHeadAttention


def test_multi_head_attention():
    np.random.seed(42)

    B = 2
    T = 4
    input_dim = 6
    num_heads = 2

    layer = MultiHeadAttention(input_dim, num_heads)

    # --------------------------------------------------
    # 1. Initialization tests
    # --------------------------------------------------

    assert layer.input_dim == input_dim
    assert layer.num_heads == num_heads
    assert layer.head_dim == input_dim // num_heads

    assert layer.Q.weights.shape == (input_dim, input_dim)
    assert layer.K.weights.shape == (input_dim, input_dim)
    assert layer.V.weights.shape == (input_dim, input_dim)

    assert layer.output_projection.weights.shape == (
        input_dim,
        input_dim
    )

    assert layer.Q_out is None
    assert layer.K_out is None
    assert layer.V_out is None
    assert layer.attention_weights is None

    expected_scale = 1 / np.sqrt(layer.head_dim)
    assert np.isclose(layer.scale, expected_scale)

    print("Initialization tests passed")


    # --------------------------------------------------
    # 2. Invalid number of heads test
    # --------------------------------------------------

    try:
        MultiHeadAttention(input_dim=5, num_heads=2)
        assert False, "Expected AssertionError"
    except AssertionError:
        pass

    print("Invalid num_heads test passed")


    # --------------------------------------------------
    # 3. Forward shape test
    # --------------------------------------------------

    x = np.random.randn(B, T, input_dim)

    output = layer.forward(x)

    assert output.shape == (B, T, input_dim)

    print("Forward output shape test passed")


    # --------------------------------------------------
    # 4. Internal tensor shape tests
    # --------------------------------------------------

    assert layer.Q_out.shape == (
        B,
        num_heads,
        T,
        layer.head_dim
    )

    assert layer.K_out.shape == (
        B,
        num_heads,
        T,
        layer.head_dim
    )

    assert layer.V_out.shape == (
        B,
        num_heads,
        T,
        layer.head_dim
    )

    assert layer.attention_weights.shape == (
        B,
        num_heads,
        T,
        T
    )

    print("Internal shape tests passed")


    # --------------------------------------------------
    # 5. Attention weights sum to 1
    # --------------------------------------------------

    attention_sums = np.sum(
        layer.attention_weights,
        axis=-1
    )

    expected = np.ones((B, num_heads, T))

    assert np.allclose(
        attention_sums,
        expected,
        atol=1e-7
    )

    print("Attention weights sum-to-one test passed")


    # --------------------------------------------------
    # 6. Attention weights are valid probabilities
    # --------------------------------------------------

    assert np.all(layer.attention_weights >= 0)
    assert np.all(layer.attention_weights <= 1)

    print("Attention probability range test passed")


    # --------------------------------------------------
    # 7. Causal masking test
    #
    # Every token must assign zero probability to
    # future tokens.
    # --------------------------------------------------

    for i in range(T):
        for j in range(i + 1, T):
            assert np.allclose(
                layer.attention_weights[:, :, i, j],
                0.0
            )

    print("Causal masking test passed")


    # --------------------------------------------------
    # 8. First token attention test
    #
    # Token 0 can only attend to itself.
    # Therefore its attention probability must be 1.
    # --------------------------------------------------

    assert np.allclose(
        layer.attention_weights[:, :, 0, 0],
        1.0
    )

    print("First-token attention test passed")


    # --------------------------------------------------
    # 9. Causality test
    #
    # Changing a future token must not change outputs
    # for previous positions.
    # --------------------------------------------------

    np.random.seed(10)

    layer_causal = MultiHeadAttention(
        input_dim,
        num_heads
    )

    x1 = np.random.randn(B, T, input_dim)
    x2 = x1.copy()

    # Change only the final token
    x2[:, -1, :] += 100.0

    output1 = layer_causal.forward(x1).copy()
    output2 = layer_causal.forward(x2).copy()

    # Positions before the final token must not change
    assert np.allclose(
        output1[:, :-1, :],
        output2[:, :-1, :],
        atol=1e-10
    )

    print("Forward causality test passed")


    # --------------------------------------------------
    # 10. Backward shape test
    # --------------------------------------------------

    np.random.seed(42)

    layer = MultiHeadAttention(
        input_dim,
        num_heads
    )

    x = np.random.randn(B, T, input_dim)

    output = layer.forward(x)

    grad_output = np.random.randn(*output.shape)

    grad_input = layer.backward(grad_output)

    assert grad_input.shape == x.shape

    print("Backward input shape test passed")


    # --------------------------------------------------
    # 11. Parameter gradient shape tests
    # --------------------------------------------------

    assert layer.Q.weights_grad.shape == (
        input_dim,
        input_dim
    )
    assert layer.K.weights_grad.shape == (
        input_dim,
        input_dim
    )
    assert layer.V.weights_grad.shape == (
        input_dim,
        input_dim
    )

    assert layer.output_projection.weights_grad.shape == (
        input_dim,
        input_dim
    )

    assert layer.Q.bias_grad.shape == (input_dim,)
    assert layer.K.bias_grad.shape == (input_dim,)
    assert layer.V.bias_grad.shape == (input_dim,)

    assert layer.output_projection.bias_grad.shape == (
        input_dim,
    )

    print("Parameter gradient shape tests passed")


    # --------------------------------------------------
    # 12. Numerical gradient check for input
    # --------------------------------------------------

    np.random.seed(1)

    B_test = 1
    T_test = 3
    D_test = 4
    H_test = 2

    layer = MultiHeadAttention(
        D_test,
        H_test
    )

    x = np.random.randn(
        B_test,
        T_test,
        D_test
    )

    output = layer.forward(x)

    upstream_grad = np.random.randn(*output.shape)

    analytical_grad = layer.backward(
        upstream_grad
    ).copy()

    epsilon = 1e-5
    numerical_grad = np.zeros_like(x)

    for index in np.ndindex(x.shape):

        original_value = x[index]

        # f(x + epsilon)
        x[index] = original_value + epsilon

        output_plus = layer.forward(x)

        loss_plus = np.sum(
            output_plus * upstream_grad
        )

        # f(x - epsilon)
        x[index] = original_value - epsilon

        output_minus = layer.forward(x)

        loss_minus = np.sum(
            output_minus * upstream_grad
        )

        numerical_grad[index] = (
            loss_plus - loss_minus
        ) / (2 * epsilon)

        x[index] = original_value

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "Input gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("Input numerical gradient test passed")


    # --------------------------------------------------
    # Helper for parameter numerical gradient tests
    # --------------------------------------------------

    def numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        parameter
    ):
        numerical_grad = np.zeros_like(parameter)
        epsilon = 1e-5

        for index in np.ndindex(parameter.shape):

            original_value = parameter[index]

            parameter[index] = (
                original_value + epsilon
            )

            output_plus = layer.forward(x)

            loss_plus = np.sum(
                output_plus * upstream_grad
            )

            parameter[index] = (
                original_value - epsilon
            )

            output_minus = layer.forward(x)

            loss_minus = np.sum(
                output_minus * upstream_grad
            )

            numerical_grad[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            parameter[index] = original_value

        return numerical_grad


    # --------------------------------------------------
    # 13. Q weight gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.Q.weights_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.Q.weights
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "Q weight gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("Q weight gradient test passed")


    # --------------------------------------------------
    # 14. K weight gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.K.weights_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.K.weights
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "K weight gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("K weight gradient test passed")


    # --------------------------------------------------
    # 15. V weight gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.V.weights_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.V.weights
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "V weight gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("V weight gradient test passed")


    # --------------------------------------------------
    # 16. Output projection weight gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = (
        layer.output_projection.weights_grad.copy()
    )

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.output_projection.weights
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "Output projection weight gradient "
        "max error:",
        max_error
    )

    assert max_error < 1e-6

    print(
        "Output projection weight gradient "
        "test passed"
    )


    # --------------------------------------------------
    # 17. Q bias gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.Q.bias_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.Q.bias
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "Q bias gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("Q bias gradient test passed")


    # --------------------------------------------------
    # 18. K bias gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.K.bias_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.K.bias
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "K bias gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("K bias gradient test passed")


    # --------------------------------------------------
    # 19. V bias gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = layer.V.bias_grad.copy()

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.V.bias
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "V bias gradient max error:",
        max_error
    )

    assert max_error < 1e-6

    print("V bias gradient test passed")


    # --------------------------------------------------
    # 20. Output projection bias gradient test
    # --------------------------------------------------

    layer.forward(x)
    layer.backward(upstream_grad)

    analytical_grad = (
        layer.output_projection.bias_grad.copy()
    )

    numerical_grad = numerical_parameter_gradient(
        layer,
        x,
        upstream_grad,
        layer.output_projection.bias
    )

    max_error = np.max(
        np.abs(
            analytical_grad
            - numerical_grad
        )
    )

    print(
        "Output projection bias gradient "
        "max error:",
        max_error
    )

    assert max_error < 1e-6

    print(
        "Output projection bias gradient "
        "test passed"
    )


    print(
        "\nAll MultiHeadAttention tests passed!"
    )


if __name__ == "__main__":
    test_multi_head_attention()