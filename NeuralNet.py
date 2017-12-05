import numpy as np

class NeuralNet:
    def __init__(self):
        self.weights = None
        self.k = 0

    def fit(self, X, Y, hidden_layers, learning_rate=1, epochs=1000, gradient_method='SGD', initial_weights=None):
        X = X.T.copy()
        Y = Y.T.copy()
        self.weights = []
        self.bias = []
        self.k = np.shape(Y)[0]

        f = np.shape(X)[0]
        self.weights.append(None)
        self.bias.append(None)
        if hidden_layers != None:
            for i in hidden_layers:
                self.weights.append(np.random.randn(i, f) * 0.01)
                self.bias.append(np.zeros((i, 1)))
                f = i
        self.weights.append(np.random.randn(self.k, f) * 0.01)
        self.bias.append(np.zeros((self.k, 1)))

        if gradient_method == 'BGD':
            self.BGD(X, Y, learning_rate, epochs)
        elif gradient_method == 'MBGD':
            self.MBGD(X, Y, learning_rate, epochs)
        else:
            self.SGD(X, Y, learning_rate, epochs)

    def forward(self, X):
        activations = []
        zs = []
        activations.append(X)
        zs.append(None)
        for i in np.arange(1, len(self.weights)):
            Z = self.weights[i].dot(activations[i-1]) + self.bias[i]
            A = NeuralNet.sigma(Z)
            activations.append(A)
            zs.append(Z)
        return A, activations, zs

    def cost(self, X, Y):
        h = self.forward(X)[0]
        m = np.shape(X)[1]
        cost = np.sum(np.sum(-Y * np.log(h) - (1 - Y) * np.log(1 - h))) / m
        return cost

    def gradients(self, X, Y):
        dws = []
        dbs = []
        h, activations, zs = self.forward(X)
        da = - (np.divide(Y, h) - np.divide(1 - Y, 1 - h)) #dJ/dOutput
        for d in np.arange(1, len(self.weights))[::-1]:
            a = activations[d] #output for layer d
            z = zs[d] #linear for layer d
            w = self.weights[d] #weights for layer w

            dz = da * a * (1-a)
            m = w.shape[1]
            dw = 1./m * dz.dot(activations[d].T)
            db = 1./m * np.sum(dz, axis=1, keepdims=True)
            da = w.T.dot(dz)
            dws.insert(0, dw)
            dbs.insert(0, db)
        dws.insert(0, None)
        dbs.insert(0, None)
        return dws, dbs

    @classmethod
    def learning_schedule(cls, t, t0, t1):
        return t0/(t+t1)

    def BGD(self, X, y, learning_rate, epochs):
        for i in np.arange(epochs):
            dw, db = self.gradients(X, y)
            for w in np.arange(1, len(self.weights)):
                self.weights[w] = self.weights[w] - learning_rate * dw[w]
                self.bias[w] = self.bias[w] - learning_rate * db[w]
            #if i%10==0:
            print(self.cost(X, y))

    def MBGD(self, X, y, learning_rate, epochs):
        m = X.shape[0]
        size = int(m * 0.1) + 1
        for e in np.arange(epochs):
            print(self.cost(X, y))
            for i in np.arange(m):
                index = np.random.randint(m - size)
                gradients = self.gradients(X[index:index + size], y[index:index + size])
                for w in np.arange(len(self.weights)):
                    self.weights[w] = self.weights[w] - NeuralNet.learning_schedule(epochs * m + i, epochs / 10, epochs)* learning_rate * gradients[w]


    def SGD(self, X, y, learning_rate, epochs):
        m = X.shape[0]
        for e in np.arange(epochs):
            if e % 1 == 0:
                print(self.cost(X, y))
            np.random.shuffle(X)
            for i in np.arange(m):
                gradients = self.gradients(X[i:i+1], y[i:i+1])
                for w in np.arange(len(self.weights)):
                    self.weights[w] = self.weights[w] - NeuralNet.learning_schedule(epochs * m + i, epochs / 10, epochs) * learning_rate * gradients[w]
                    #self.weights[w] = self.weights[w] - learning_rate * gradients[w]

    def predict(self, X):
        X = X.T.copy()
        output, acts, zs = self.forward(X)


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
