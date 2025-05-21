import numpy as np
import func_util as fu

class NNet(object):
    def __init__(self, layers):
        self.size = len(layers)
        self.layers = layers
        self.biases = [np.random.randn(y, 1) for y in layers[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(layers[:-1], layers[1:])]

    def SGD(self, training_data, epochs, batch_size, learning_rate=0.01, test_data=None):
        """
        Train the neural network using the provided training data.
        :param training_data: List of tuples (input, target)
        :param epochs: Number of epochs to train
        :param batch_size: Size of each training batch
        :param learning_rate: Learning rate for the optimizer
        :param test_data: Optional test data for validation
        """
        if test_data:
            n_test = len(test_data)
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, learning_rate)
            
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def predict(self, x):
        """
        Make predictions using the trained network.
        :param x: Input data
        :return: Predicted output
        """
        # a = self._forward(x)
        # return np.argmax(a)
        self._forward(x)

    def evaluate(self, test_data):
        """
        Evaluate the performance of the network on test data.
        :param test_data: List of tuples (input, target)
        :return: Number of correct predictions
        """
        test_results = [(np.argmax(self._forward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
    
    def accuracy(self, test_data):
        """
        Calculate the accuracy of the network on test data.
        :param test_data: List of tuples (input, target)
        :return: Accuracy as a percentage
        """
        test_results = [(np.argmax(self._forward(x)), y) for (x, y) in test_data]
        accurate = sum(int(x == y) for (x, y) in test_results)
        return accurate, len(test_results), accurate / len(test_results)

    def _forward(self, x, keep_activations=False):
        """
        Forward pass through the network.
        :param x: Input data
        :param keep_activations: Whether to keep activations for backpropagation
        :return: Output of the network
        """
        if keep_activations:
            self.activations = [x]
            self.zs = []
            for b, w in zip(self.biases, self.weights):
                z = np.dot(w, x) + b
                self.zs.append(z)
                x = fu.sigmoid(z)
                self.activations.append(x)
        else:
            for b, w in zip(self.biases, self.weights):
                x = fu.sigmoid(np.dot(w, x) + b)
        
        return x

    def _backprop(self, y):
        """
        Backpropagation algorithm to compute gradients.
        :param y: Target output
        :return: Gradients of weights and biases
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        """
        Calculate the gradient of the cost function with respect to the output
        of the network.
        """

        """
        Apply equation (BP1) from the backpropagation algorithm
        """
        delta = fu.quadratic_loss_prime(y, self.activations[-1]) * fu.sigmoid_prime(self.zs[-1])
        """
        Apply equation (BP3) and (BP4) from the backpropagation algorithm
        """
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, self.activations[-2].T)

        """
        Backpropagate the gradient to the previous layers.
        """
        for l in range(2, self.size):
            z = self.zs[-l]
            sp = fu.sigmoid_prime(z)
            """
            Apply equation (BP2) from the backpropagation algorithm
            """
            delta = np.dot(self.weights[-l + 1].T, delta) * sp

            """
            Apply equation (BP3) and (BP4) from the backpropagation algorithm
            """
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, self.activations[-l - 1].T)

        return nabla_b, nabla_w
    
    def _update_mini_batch(self, mini_batch, learning_rate):
        """
        Update the network weights and biases using backpropagation.
        :param batch: Mini-batch of training data
        :param learning_rate: Learning rate for the optimizer
        """
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            """
            1. feedforward
            """
            self._forward(x, keep_activations=True)
            
            """
            2. backpropagation
            """
            delta_nabla_b, delta_nabla_w = self._backprop(y)

            """
            3. accumulate gradients of weights and biases for each sample
            """
            nabla_b = [nb + nd for nb, nd in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + nd for nw, nd in zip(nabla_w, delta_nabla_w)]

        """
        4. update weights and biases
        """
        self.weights = [w - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
