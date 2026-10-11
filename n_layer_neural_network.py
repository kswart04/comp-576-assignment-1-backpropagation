from three_layer_neural_network_2026 import NeuralNetwork

class DeepNeuralNetwork(num_layers, layer_size):
    def __init__(self, num_layers, layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda