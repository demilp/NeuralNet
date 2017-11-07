import numpy as np

class NeuralNet:
    def __init__(self):
        self.weights = None
        self.k = 0

    def fit(self, X, Y, hidden_layers, c = 0, learning_rate=1, epochs=1000, gradient_method='SGD', initial_weights=None):
        self.weights = []
        self.k = np.shape(Y)[1]

        if initial_weights == None:
            f = np.shape(X)[1]
            for i in hidden_layers:
                self.weights.append(np.random.randn(i, f+1))
                f = i
            self.weights.append(np.random.randn(self.k, f+1))
        else:
            self.weights = initial_weights

        if gradient_method == 'BGD':
            self.BGD(X, Y, learning_rate, epochs, c)
        elif gradient_method == 'MBGD':
            self.MBGD(X, Y, learning_rate, epochs, c)
        else:
            self.SGD(X, Y, learning_rate, epochs, c)

    def predict(self, X):
        activations = []
        zs = []
        activations.append(X)
        for i in np.arange(len(self.weights)):
            m = np.shape(X)[0]
            X = np.c_[np.ones((m, 1)), X]
            Z = X.dot(self.weights[i].T)
            X = NeuralNet.sigma(Z)
            activations.append(X)
            zs.append(Z)
        return X, activations, zs

    def cost(self, X, Y, c):
        h = self.predict(X)[0]
        m = np.shape(X)[0]
        cost = np.sum(np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))) / m
        #reg = 0
        #for w in self.weights:
        #    reg += np.sum(np.sum(w**2))
        #reg *= c/(2*m)
        return cost# + reg

    def gradients(self, X, Y, c):
        m = np.shape(X)[0]
        grads = []
        h, activations, zs = self.predict(X)
        deltas = []
        for d in np.arange(len(self.weights))[::-1]:
            if d == len(self.weights)-1:
                delta = activations[-1] - Y
            else:
                delta = deltas[0].dot(self.weights[d+1][:, 1:]) * self.sigmoidGradient(zs[d])
            deltas.insert(0, delta)
            a = np.c_[np.ones((np.shape(activations[d])[0], 1)), activations[d]]
            gradient = (delta.T.dot(a)) / m
            layer_shape = np.shape(self.weights[d])
            mask = np.c_[np.zeros((layer_shape[0], 1)), np.ones((layer_shape[0], layer_shape[1]-1))]
            reg = mask * (c/m) * self.weights[d]
            gradient += reg
            grads.insert(0, gradient)
        return grads

    @classmethod
    def learning_schedule(cls, t, t0, t1):
        return t0/(t+t1)

    def BGD(self, X, y, learning_rate, epochs, c = 0.02):
        for i in np.arange(epochs):
            gradients = self.gradients(X, y, c)
            for i in np.arange(len(self.weights)):
                self.weights[i] = self.weights[i] - learning_rate * gradients[i]
            if i%150==0:
                print(self.cost(X, y, c))

    def MBGD(self, X, y, learning_rate, epochs, c = 0.02):
        m = X.shape[0]
        size = int(m * 0.1) + 1
        for e in np.arange(epochs):
            print(self.cost(X, y, c))
            for i in np.arange(m):
                index = np.random.randint(m - size)
                gradients = self.gradients(X[index:index + size], y[index:index + size], c)
                for i in np.arange(len(self.weights)):
                    self.weights[i] = self.weights[i] - NeuralNet.learning_schedule(epochs * m + i, epochs / 10, epochs) * learning_rate * gradients[i]


    def SGD(self, X, y, learning_rate, epochs, c = 0.02):
        m = X.shape[0]
        for e in np.arange(epochs):
            if e % 10 == 0:
                print(self.cost(X, y, c))
            for i in np.arange(m):
                if i%10000==0:
                    pass
                    #print(self.cost(X, y, c))
                index = np.random.randint(m)
                gradients = self.gradients(X[index:index+1], y[index:index+1], c)
                for i in np.arange(len(self.weights)):
                    self.weights[i] = self.weights[i] - NeuralNet.learning_schedule(epochs * m + i, epochs / 10, epochs) * learning_rate * gradients[i]



    @classmethod
    def sigmoidGradient(cls, z):
        g = cls.sigma(z)
        g = g * (1 - g)
        return  g

    @classmethod
    def sigma(cls, z):
        res = 1 / (1 + np.exp(-z))
        res = np.clip(res, 0.0000000000001, 0.9999999999999)
        return res
