from neural import *

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_training_data = [([1, 1], [0]), ([1, 0], [1]), ([0, 1], [1]), ([0, 0], [0])]

# xorn = NeuralNet(2, 2, 1)
xorn = NeuralNet(2, 8, 1)
xorn.train(xor_training_data)
print(xorn.test_with_expected(xor_training_data))