import pickle
import gzip

import numpy as np

def load_data():
    """
    Return the MNIST data as 3 data sets: training, validation, and test data.
    Each data set is a tuple with two entries. The first entry contains
    the actual training images. This is a numpy ndarray with 50,000 entries.
    Each entry is, in turn, a numpy ndarray with 784 values, representing
    the 28 * 28 = 784 pixels in a single MNIST image. The second entry contains
    the digit values (0...9) for the corresponding images contained in the first
    entry of the tuple.
    The validation and test data are similar, except each contains only 10,000 images.
    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the training data a little. That's done in
    the wrapper function load_data_wrapper(), see below.
    """
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(f, encoding='latin1')
    f.close()

    return training_data, validation_data, test_data

def load_data_wrapper():
    """
    Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code.
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)

def load_data_wrapper_convnet():
    """
    Return a tuple containing (training_data, validation_data, test_data) for ConvNet.
    The input x is reshaped to (1, 28, 28) tensor for each image.
    The output y is the same as before: one-hot vector for training, integer for validation/test.
    """
    tr_d, va_d, te_d = load_data()

    training_inputs = [np.reshape(x, (1, 28, 28)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (1, 28, 28)) for x in va_d[0]]
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = [np.reshape(x, (1, 28, 28)) for x in te_d[0]]
    test_data = list(zip(test_inputs, te_d[1]))

    return (training_data, validation_data, test_data)

def vectorized_result(input):
    """
    Return a 10-dimensional unit vector with a 1.0 in the position
    """
    vector = np.zeros((10, 1))
    vector[input] = 1.0
    return vector
