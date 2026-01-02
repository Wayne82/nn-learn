import sys
import os

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

def test_gpt_transformer(data_path, save_path=None, load_path=None, train_model=True, check_model_params=False, input_text=None):
    # Load data
    data_loader = DataLoader(data_path, batch_size=16)
    data_loader.load_data()
    data_loader.print_data_stats()
    # data_loader.print_samples(n=200)

    # The configuration to train "shijing.txt" and "lunyu.txt".
    model = GPTTransformer(vocab_size=data_loader.get_vocab_size(),
                           block_size=16,
                           n_embd=64,
                           n_head=4,
                           n_layer=4,
                           dropout=0.2)

    # The configuration to train "classical_poetry_simplified.txt".
    # model = GPTTransformer(vocab_size=data_loader.get_vocab_size(),
    #                     block_size=32,
    #                     n_embd=256,
    #                     n_head=6,
    #                     n_layer=6,
    #                     dropout=0.2)
    model.print_params()
    if check_model_params:
        return

    model_loaded = False
    if load_path and os.path.exists(load_path):
        print(f"Loading model parameters from {load_path}")
        model.load_model(load_path)
        model_loaded = True

    if train_model:
        # Initialize trainer
        trainer = Trainer(data_loader, model, learning_rate=1e-4)
        # Train the model
        trainer.train(max_iters=20000)
        if save_path:
            print(f"Saving model parameters to {save_path}")
            model.save_model(save_path)
    elif not model_loaded:
        print("Model not trained or loaded. Exiting test.")
        return

    # Generate text
    if input_text:
        print(f"Starting with input text: {input_text}")
        context = torch.tensor([data_loader.encode(input_text)], dtype=torch.long)
    else:
        context = torch.zeros((1, 1), dtype=torch.long)
    generated = model.generate(context, max_new_tokens=500)
    print("Generated text:\n", data_loader.decode(generated[0].tolist()))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test.py [nnet|convnet|gpt_transformer] [simple|complex] [save_path] [load_path] [train_model] [check_model_params]")
        sys.exit(1)
    model_type = sys.argv[1].lower()
    print(f"Testing model type: {model_type}")
    if model_type == "nnet":
        test_nnet()
    elif model_type == "convnet":
        architecture = 'simple'  # default
        if len(sys.argv) > 2:
            architecture = sys.argv[2].lower()
        test_convnet(architecture)
    elif model_type == "gpt_transformer":
        # Default values
        params = {
            'save_path': None,
            'load_path': None,
            'train_model': True,
            'check_model_params': False,
            'input_text': None
        }
        # Parse key=value pairs
        for arg in sys.argv[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                key = key.strip()
                value = value.strip()
                if key in ['train_model', 'check_model_params']:
                    params[key] = value.lower() == 'true'
                else:
                    params[key] = value
        test_gpt_transformer(
            data_path=params.get('data_path', './data/shijing.txt'),
            save_path=params['save_path'],
            load_path=params['load_path'],
            train_model=params['train_model'],
            check_model_params=params['check_model_params'],
            input_text=params['input_text']
        )
    else:
        print("Unknown test type. Use 'nnet', 'convnet', or 'gpt_transformer'.")
        print("For convnet, you can optionally specify 'simple' or 'complex' architecture.")