import numpy as np
import func_util as fu

class NNetOptions:
    """
    Class to hold options for the neural network.
    """
    def __init__(self,
                 hidden_activation = fu.ActivationFunction.SIGMOID,
                 output_activation=fu.ActivationFunction.SIGMOID,
                 cost=fu.CostFunction.QUADRATIC,
                 l2_reg_lambda=0.0):
        self.activation_fns = {
            'hidden': fu.ActivationFunction.get_activation_function(hidden_activation),
            'output': fu.ActivationFunction.get_activation_function(output_activation)
        }
        self.cost_fn = fu.CostFunction.get_cost_function(cost)
        self.l2_reg_lambda = l2_reg_lambda

    def __str__(self):
        return f"Activations: {self.activation_fns}, Cost: {self.cost_fn}"

class NNet(object):
    def __init__(self, layers, options=NNetOptions()):
        self.size = len(layers)
        self.layers = layers
        self.activation_fns = options.activation_fns
        self.cost_fn = options.cost_fn
        self.l2_reg_lambda = options.l2_reg_lambda

        self._weights_initialization()

    def SGD(self, training_data, epochs, batch_size, learning_rate=0.01, validation_data=None):
        """
        Train the neural network using the provided training data.
        :param training_data: List of tuples (input, target)
        :param epochs: Number of epochs to train
        :param batch_size: Size of each training batch
        :param learning_rate: Learning rate for the optimizer
        :param validation_data: Optional test data for validation
        """
        if validation_data:
            n_test = len(validation_data)
        n = len(training_data)

        for j in range(epochs):
            np.random.shuffle(training_data)
            mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, learning_rate, n)

            if validation_data:
                print(f"Epoch {j}: {self.evaluate(validation_data)} / {n_test}")
            else:
                print(f"Epoch {j} complete")

    def predict(self, x):
        """
        Make predictions using the trained network.
        :param x: Input data
        :return: Predicted output
        """
        return self._forward(x)

    def predict_number(self, x):
        """
        Make predictions using the trained network.
        :param x: Input data
        :return: Predicted number
        """
        a = self._forward(x)
        return np.argmax(a)

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

    def _weights_initialization(self):
        """
        Initialize weights and biases for the network.
        :return: None
        """
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]

        h_coefficient = self.activation_fns['hidden'].initialize_weights_coefficient
        o_coefficient = self.activation_fns['output'].initialize_weights_coefficient

        # Initialize weights in hidden layers
        self.weights = [np.random.randn(y, x) * h_coefficient(x) for x, y in zip(self.layers[:-1], self.layers[1:-1])]
        # Initialize weights in output layer
        self.weights.append(np.random.randn(self.layers[-1], self.layers[-2]) * o_coefficient(self.layers[-2]))

    def _forward(self, x, keep_activations=False):
        """
        Forward pass through the network.
        :param x: Input data
        :param keep_activations: Whether to keep activations for backpropagation
        :return: Output of the network
        """
        n = [y+1 for y in np.arange(self.size - 1)]
        if keep_activations:
            self.activations = [x]
            self.zs = []

        for b, w, i in zip(self.biases, self.weights, n):
            z = np.dot(w, x) + b

            if i == self.size - 1:
                x = self.activation_fns['output'].fn(z)
            else:
                x = self.activation_fns['hidden'].fn(z)

            if keep_activations:
                self.zs.append(z)
                self.activations.append(x)

        return x

    def _backprop(self, y):
        """
        Backpropagation algorithm to compute gradients.
        :param y: Target output
        """

        """
        Calculate the gradient of the cost function with respect to
        the output of the network.
        Apply equation (BP1)
        """
        delta = self.cost_fn.prime(y, self.activations[-1]) * \
                self.activation_fns['output'].prime(self.zs[-1])
        """
        Apply equation (BP3) and (BP4)
        """
        self.nabla_b[-1] += delta
        self.nabla_w[-1] += np.dot(delta, self.activations[-2].T)

        """
        Iterate through the layers in reverse order to compute gradients
        """
        for l in range(2, self.size):
            z = self.zs[-l]
            sp = self.activation_fns['hidden'].prime(z)

            """
            Apply equation (BP2)
            """
            delta = np.dot(self.weights[-l + 1].T, delta) * sp

            """
            Apply equation (BP3) and (BP4)
            """
            self.nabla_b[-l] += delta
            self.nabla_w[-l] += np.dot(delta, self.activations[-l - 1].T)

    def _update_mini_batch(self, mini_batch, learning_rate, training_size):
        """
        Update the network weights and biases using backpropagation.
        :param batch: Mini-batch of training data
        :param learning_rate: Learning rate for the optimizer
        """
        self.nabla_b = [np.zeros(b.shape) for b in self.biases]
        self.nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            """
            1. feedforward
            """
            self._forward(x, keep_activations=True)

            """
            2. backpropagation
            """
            self._backprop(y)

        """
        3. update weights and biases
        """
        self.weights = [(1 - learning_rate * (self.l2_reg_lambda / training_size)) * w
                        - (learning_rate / len(mini_batch)) * nw for w, nw in zip(self.weights, self.nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, self.nabla_b)]
