from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from NeuralNet import NeuralNet
import matplotlib.pyplot as plt

import numpy as np

def run():
    #mnist = fetch_mldata('MNIST original')
    #X = mnist['data'].astype(float)
    #y = mnist['target'].astype(int)
    ss = StandardScaler()
    #X = ss.fit_transform(X)
    #Y = convertToOneHot(y)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
   #  X = np.array(
   #  [ [ 0.1682942,-0.1922795],
   #     [0.1818595,-0.1501974],
   #     [0.0282240, 0.0299754],
   #    [-0.1513605, 0.1825891],
   #    [-0.1917849, 0.1673311],
   #    [-0.0558831,-0.0017703],
   #     [0.1313973,-0.1692441],
   #     [0.1978716,-0.1811157],
   #     [0.0824237,-0.0264704],
   #    [-0.1088042, 0.1525117],
   #    [-0.1999980, 0.1912752],
   #    [-0.1073146, 0.0541812],
   #     [0.0840334,-0.1327268],
   #     [0.1981215,-0.1976063],
   #     [0.1300576,-0.0808075],
   #    [-0.0575807, 0.1102853]])
   #  y = np.array([
   # [0,   1,   0,   0],
   # [0,   0,   1,   0],
   # [0,   0,   0,   1],
   # [1,   0,   0,   0],
   # [0,   1,   0,   0],
   # [0,   0,   1,   0],
   # [0,   0,   0,   1],
   # [1,   0,   0,   0],
   # [0,   1,   0,   0],
   # [0,   0,   1,   0],
   # [0,   0,   0,   1],
   # [1,   0,   0,   0],
   # [0,   1,   0,   0],
   # [0,   0,   1,   0],
   # [0,   0,   0,   1],
   # [1,   0,   0,   0]])

    #nn = NeuralNet()
    #nn.fit(X_train, Y_train, [25], 0, epochs=15, gradient_method='SGD', learning_rate=10)
    #nn.fit(X_train, Y_train, [25], 0.1, epochs=15, gradient_method='SGD', initial_weights=np.load('weights.npy'), learning_rate=0.1)
    #np.save('weights', nn.weights)

    #index = 67
    #some_digit = X_test[index]
    #some_digit_image = some_digit.reshape(28, 28)
    #print('Y = ')
    #print(Y_test[index])
    #p = nn.predict(X_test[index][np.newaxis])
    #print(p[0])

    X = np.array([[-3],[-2],[3],[2]])
    z = np.zeros((2, 1))
    o = np.ones((2, 1))
    Y = np.concatenate((z, o), axis=0)

    nn = NeuralNet()
    nn.fit(X, Y, None, 1, 20, 'BGD')
    #plt.scatter(X[Y.flatten() == 0][:, 0], X[Y.flatten() == 0][:, 1])
    #plt.scatter(X[Y.flatten() == 1][:, 0], X[Y.flatten() == 1][:, 1])
    print("weights: "+str(nn.weights))
    print("bias: " + str(nn.bias))
    print("pred:")
    print(nn.forward(X.T)[0])
    #plt.show()

def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)
    l = len(vector)
    result = np.zeros(shape=(l, num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)


if __name__ == '__main__':
    run()
