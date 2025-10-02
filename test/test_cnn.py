from cnn.model import CNN
import numpy as np
import pytest 

@pytest.fixture
def cnn():
    return CNN()


def test_dense_layer(cnn):
    # Input: vector of 2, weights 2x2, bias 2
    x = np.array([1.0, 2.0])
    W = np.array([[1.0, 0.0],
                [0.0, 1.0]])
    b = np.array([0.5, -0.5])
    output = cnn.dense_layer(x, output_size=2, weights=W, bias=b)
    expected = np.array([1.5, 1.5])
    assert np.allclose(output, expected)


def test_convolution(cnn):
    image = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    kernel = np.array([[1, 0],
                    [0, -1]])
    output = cnn.convolution(image, kernel, stride=1)
    assert output.shape == (2, 2)  # should produce a 2x2 feature map


def test_relu(cnn):
    fmap = np.array([[-1, 2], [3, -4]])
    output = cnn.reLu(fmap)
    expected = np.array([[0, 2], [3, 0]])
    assert np.array_equal(output, expected)

def test_max_pool(cnn):
    fmap = np.array([[1, 3, 2, 4],
                    [5, 6, 7, 8],
                    [2, 2, 9, 1],
                    [0, 1, 3, 5]])
    output = cnn.max_pool(fmap, stride=2, pool_size=2)
    # Pools: [[1,3],[5,6]] -> 6 ; [[2,4],[7,8]] -> 8 ; [[2,2],[0,1]] -> 2 ; [[9,1],[3,5]] -> 9
    expected = np.array([[6, 8],
                        [2, 9]])
    assert np.array_equal(output, expected)


def test_softmax(cnn):
    logits = np.array([[1.0, 2.0, 3.0]])
    output = cnn.soft_max(logits)
    exp = np.exp(logits - np.max(logits))
    expected = exp / np.sum(exp)
    assert np.allclose(output, expected)


def test_loss(cnn):
    y_true = np.array([0, 1, 0])  # one-hot
    y_pred = np.array([0.2, 0.7, 0.1])  # already softmaxed
    output = cnn.loss_function(y_true, y_pred)
    expected = -np.sum(y_true * np.log(y_pred))  # should equal -0.7
    assert np.isclose(output, expected)


def test_flatten(cnn):
    arr = np.array([[1,2],[3,4]])
    output = cnn.flatten(arr)
    expected = np.array([1,2,3,4])
    assert np.array_equal(output, expected)


# if __name__ == "__main__":
# test_dense_layer(cnn)
# test_convolution(cnn)
# test_relu(cnn)
# test_max_pool(cnn)
# test_softmax(cnn)
# test_loss(cnn)
# test_flatten(cnn)