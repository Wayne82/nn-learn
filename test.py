import sys

import torch
import nnet
import func_util as fu
import mnist_loader
from convnet import ConvNet, ConvNetConfig
from convnet_layers import Conv2D, MaxPool2D, ReLu, Flatten, FullyConnected
from gpt_transformer import DataLoader, GPTTransformer, Trainer

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

def test_convnet(architecture='simple'):
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper_convnet()
    config = ConvNetConfig(batch_size=10, learning_rate=0.05, epochs=30)
    net = ConvNet(config)

    if architecture.lower() == 'simple':
        # Simple CNN architecture
        net.add_layer(Conv2D(in_channels=1, out_channels=8, kernel_size=3, padding='same'))\
           .add_layer(ReLu())\
           .add_layer(MaxPool2D(pool_size=2))\
           .add_layer(Flatten())\
           .add_layer(FullyConnected(in_features=8 * 14 * 14, out_features=10))
        print("Using Simple CNN architecture")
    elif architecture.lower() == 'complex':
        # Complex CNN architecture
        net.add_layer(Conv2D(in_channels=1, out_channels=32, kernel_size=3, padding='same'))\
            .add_layer(ReLu())\
            .add_layer(Conv2D(in_channels=32, out_channels=32, kernel_size=3, padding='same'))\
            .add_layer(ReLu())\
            .add_layer(MaxPool2D(pool_size=2, stride=2))\
            .add_layer(Conv2D(in_channels=32, out_channels=64, kernel_size=3, padding='same'))\
            .add_layer(ReLu())\
            .add_layer(Conv2D(in_channels=64, out_channels=64, kernel_size=3, padding='same'))\
            .add_layer(ReLu())\
            .add_layer(MaxPool2D(pool_size=2, stride=2))\
            .add_layer(Flatten())\
            .add_layer(FullyConnected(in_features=64 * 7 * 7, out_features=128))\
            .add_layer(ReLu())\
            .add_layer(FullyConnected(in_features=128, out_features=10))
        print("Using Complex CNN architecture")
    else:
        raise ValueError("Architecture must be 'simple' or 'complex'")

    print("Network configuration:", config)
    print("Evaluating an untrained ConvNet on test data:")
    print(net.evaluate(test_data))
    print("Training the ConvNet:")
    net.SGD(training_data, validation_data=validation_data)
    print("Evaluating the trained ConvNet on test data:")
    print(net.evaluate(test_data))

def test_gpt_transformer():

    # Load data
    data_loader = DataLoader('./data/shijing.txt', batch_size=16)
    data_loader.load_data()
    # data_loader.print_samples(n=200)

    # Initialize model
    model = GPTTransformer(vocab_size=data_loader.get_vocab_size(), n_embd=64, n_head=4, n_layer=2)
    model.print_params()

    # Initialize trainer
    trainer = Trainer(data_loader, model, learning_rate=1e-3)

    # Train the model
    trainer.train(max_iters=5000)

    # Generate text
    context = torch.zeros((1, 1), dtype=torch.long)
    generated = trainer.generate(context, max_new_tokens=100)
    print("Generated text:", data_loader.decode(generated[0].tolist()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py [nnet|convnet|gpt_transformer] [simple|complex]")
        sys.exit(1)
    test_type = sys.argv[1].lower()
    if test_type == "nnet":
        test_nnet()
    elif test_type == "convnet":
        # Check for architecture parameter
        architecture = 'simple'  # default
        if len(sys.argv) > 2:
            architecture = sys.argv[2].lower()
        test_convnet(architecture)
    elif test_type == "gpt_transformer":
        test_gpt_transformer()
    else:
        print("Unknown test type. Use 'nnet', 'convnet', or 'gpt_transformer'.")
        print("For convnet, you can optionally specify 'simple' or 'complex' architecture.")