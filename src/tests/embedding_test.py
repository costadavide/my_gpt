import numpy as np

from src.layers import EmbeddingLayer


def test_embedding_layer():
    np.random.seed(42)

    num_embeddings = 6
    embedding_dim = 4

    layer = EmbeddingLayer(num_embeddings, embedding_dim)

    # --------------------------------------------------
    # 1. Initialization tests
    # --------------------------------------------------

    assert layer.embedding_matrix.shape == (num_embeddings, embedding_dim)
    assert layer.embedding_grad is None
    assert layer.input is None

    print("Initialization tests passed")


    # --------------------------------------------------
    # 2. Forward shape test
    # --------------------------------------------------

    x = np.array([
        [0, 2, 4],
        [1, 3, 5]
    ])

    output = layer.forward(x)

    assert output.shape == (2, 3, embedding_dim)

    print("Forward shape test passed")


    # --------------------------------------------------
    # 3. Forward correctness test
    # --------------------------------------------------

    assert np.allclose(output[0, 0], layer.embedding_matrix[0])
    assert np.allclose(output[0, 1], layer.embedding_matrix[2])
    assert np.allclose(output[0, 2], layer.embedding_matrix[4])

    assert np.allclose(output[1, 0], layer.embedding_matrix[1])
    assert np.allclose(output[1, 1], layer.embedding_matrix[3])
    assert np.allclose(output[1, 2], layer.embedding_matrix[5])

    print("Forward correctness test passed")


    # --------------------------------------------------
    # 4. Backward shape test
    # --------------------------------------------------

    grad_output = np.random.randn(*output.shape)

    layer.backward(grad_output)

    assert layer.embedding_grad.shape == layer.embedding_matrix.shape

    print("Backward shape test passed")


    # --------------------------------------------------
    # 5. Simple backward correctness test
    #
    # No repeated indices here, so every used row should
    # receive exactly one grad_output vector.
    # --------------------------------------------------

    expected_grad = np.zeros_like(layer.embedding_matrix)

    for b in range(x.shape[0]):
        for t in range(x.shape[1]):
            token_id = x[b, t]
            expected_grad[token_id] += grad_output[b, t]

    assert np.allclose(layer.embedding_grad, expected_grad)

    print("Backward correctness test passed")


    # --------------------------------------------------
    # 6. Repeated-index test
    #
    # This is especially important for np.add.at.
    # --------------------------------------------------

    x_repeated = np.array([
        [2, 5, 2],
        [2, 1, 5]
    ])

    output = layer.forward(x_repeated)

    grad_output = np.array([
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [2.0, 3.0, 4.0, 5.0]
        ],
        [
            [10.0, 20.0, 30.0, 40.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 4.0, 5.0, 6.0]
        ]
    ])

    layer.backward(grad_output)

    expected_grad = np.zeros_like(layer.embedding_matrix)

    # token 2 appears three times
    expected_grad[2] = (
        grad_output[0, 0]
        + grad_output[0, 2]
        + grad_output[1, 0]
    )

    # token 5 appears twice
    expected_grad[5] = (
        grad_output[0, 1]
        + grad_output[1, 2]
    )

    # token 1 appears once
    expected_grad[1] = grad_output[1, 1]

    assert np.allclose(layer.embedding_grad, expected_grad)

    print("Repeated-index backward test passed")


    # --------------------------------------------------
    # 7. Unused embeddings must have zero gradient
    # --------------------------------------------------

    used_indices = np.unique(x_repeated)

    for i in range(num_embeddings):
        if i not in used_indices:
            assert np.allclose(
                layer.embedding_grad[i],
                np.zeros(embedding_dim)
            )

    print("Unused embeddings gradient test passed")


    # --------------------------------------------------
    # 8. Gradient reset test
    #
    # backward() should not accidentally keep gradients
    # from a previous backward call.
    # --------------------------------------------------

    x = np.array([[0]])

    output = layer.forward(x)

    grad_output = np.ones_like(output)

    layer.backward(grad_output)
    first_grad = layer.embedding_grad.copy()

    layer.backward(grad_output)
    second_grad = layer.embedding_grad.copy()

    assert np.allclose(first_grad, second_grad)

    print("Gradient reset test passed")


    # --------------------------------------------------
    # 9. Numerical gradient check
    # --------------------------------------------------

    np.random.seed(1)

    layer = EmbeddingLayer(5, 3)

    x = np.array([
        [1, 3, 1],
        [2, 4, 3]
    ])

    output = layer.forward(x)

    upstream_grad = np.random.randn(*output.shape)

    layer.backward(upstream_grad)

    analytical_grad = layer.embedding_grad.copy()

    epsilon = 1e-5
    numerical_grad = np.zeros_like(layer.embedding_matrix)

    for i in range(layer.embedding_matrix.shape[0]):
        for j in range(layer.embedding_matrix.shape[1]):

            original_value = layer.embedding_matrix[i, j]

            # f(W + epsilon)
            layer.embedding_matrix[i, j] = original_value + epsilon
            output_plus = layer.forward(x)
            loss_plus = np.sum(output_plus * upstream_grad)

            # f(W - epsilon)
            layer.embedding_matrix[i, j] = original_value - epsilon
            output_minus = layer.forward(x)
            loss_minus = np.sum(output_minus * upstream_grad)

            numerical_grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            # restore parameter
            layer.embedding_matrix[i, j] = original_value

    max_error = np.max(
        np.abs(analytical_grad - numerical_grad)
    )

    print("Embedding gradient max error:", max_error)

    assert max_error < 1e-7

    print("Numerical gradient test passed")

    print("\nAll EmbeddingLayer tests passed!")


if __name__ == "__main__":
    test_embedding_layer()