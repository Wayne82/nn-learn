import sys
import nnet
import func_util as fu
import mnist_loader
from convnet import ConvNet, ConvNetConfig
from convnet_layers import Conv2D, MaxPool2D, ReLu, Flatten, FullyConnected

def test_nnet():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    options = nnet.NNetOptions(
        hidden_activation=fu.ActivationFunction.RELU,
        output_activation=fu.ActivationFunction.SOFTMAX,
        cost=fu.CostFunction.CROSS_ENTROPY,
        l2_reg_lambda=0.1
    )
    net = nnet.NNet([784, 30, 10], options)

    print("Network options:", options)
    print("Evaluating an untrained network on test data:")
    print(net.accuracy(test_data))

    print("Training the network:")
    net.SGD(training_data, epochs=30, batch_size=10, learning_rate=0.1, validation_data=validation_data)
    print("Evaluating the trained network on test data:")
    print(net.accuracy(test_data))

def test_convnet():
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper_convnet()
    config = ConvNetConfig(batch_size=10, learning_rate=0.08, epochs=30)
    net = ConvNet(config)

    # Add layers to the ConvNet
    net.add_layer(Conv2D(in_channels=1, out_channels=8, kernel_size=3))\
       .add_layer(ReLu())\
       .add_layer(MaxPool2D(pool_size=2))\
       .add_layer(Flatten())\
       .add_layer(FullyConnected(in_features=8 * 13 * 13, out_features=10))

    print("Network configuration:", config)
    print("Evaluating an untrained ConvNet on test data:")
    print(net.evaluate(test_data))
    print("Training the ConvNet:")
    net.SGD(training_data, validation_data=validation_data)
    print("Evaluating the trained ConvNet on test data:")
    print(net.evaluate(test_data))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py [nnet|convnet]")
        sys.exit(1)
    test_type = sys.argv[1].lower()
    if test_type == "nnet":
        test_nnet()
    elif test_type == "convnet":
        test_convnet()
    else:
        print("Unknown test type. Use 'nnet' or 'convnet'.")