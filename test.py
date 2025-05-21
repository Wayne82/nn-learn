import nnet
import mnist_loader

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
net = nnet.NNet([784, 30, 10])

print("Evaluating an untrained network on test data:")
print(net.accuracy(test_data))

print("Training the network:")
net.SGD(training_data, epochs=30, batch_size=10, learning_rate=3.0, test_data=test_data)
print("Evaluating the trained network on test data:")
print(net.accuracy(test_data))