from neuralnetwork import NeuralNetwork
import numpy as np

nn = NeuralNetwork(2,4,1)

print("before training")
print(nn.predict([[1,1]]))


inputs = np.array([[1,0],
              [0,1],
              [0,0],
              [1,1]])

targets = np.array([[1],
              [1],
              [0],
              [0]])

nn.train(inputs, targets, 60000)

print("after training")
print(nn.predict([[0,1]]))