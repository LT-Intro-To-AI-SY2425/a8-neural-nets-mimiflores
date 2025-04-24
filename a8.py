from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

# # xorn = NeuralNet(2, 2, 1)
# xorn = NeuralNet(2, 8, 1)
# xorn.train(xor_training_data)
# print(xorn.test_with_expected(xor_training_data))

voting_data = [([0.9, 0.6, 0.8, 0.3, 0.1], [1]), ([0.8, 0.8, 0.4, 0.6, 0.4], [1]), ([0.8, 0.2, 0.4, 0.6, 0.3], [1]), ([0.5, 0.5, 0.8, 0.4, 0.8], [0]), ([0.3, 0.1, 0.6, 0.8, 0.8], [0]), ([0.6, 0.3, 0.4, 0.3, 0.6], [0])]
nn = NeuralNet(5, 2, 1)
nn.train(voting_data)

print(nn.evaluate([1.0, 1.0, 1.0, 0.1, 0.1]))
print(nn.evaluate([0.5, 0.2, 0.1, 0.7, 0.7]))
print(nn.evaluate([0.8, 0.3, 0.3, 0.3, 0.8]))
print(nn.evaluate([0.8, 0.3, 0.3, 0.8, 0.3]))
print(nn.evaluate([0.9, 0.8, 0.8, 0.3, 0.6]))


