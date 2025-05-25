import nnet
import func_util as fu
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = nnet.NNet([784, 30, 10], nnet.NNetOptions(hidden_activation=fu.ActivationFunction.RELU,
                                                    output_activation=fu.ActivationFunction.SIGMOID,
                                                    cost=fu.CostFunction.QUADRATIC,
                                                    l2_reg_lambda=0.1))

print("Evaluating an untrained network on test data:")
print(net.accuracy(test_data))

print("Training the network:")
net.SGD(training_data, epochs=30, batch_size=10, learning_rate=0.1, test_data=test_data)
print("Evaluating the trained network on test data:")
print(net.accuracy(test_data))