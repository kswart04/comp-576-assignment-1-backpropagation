import numpy as np

from three_layer_neural_network_2026 import NeuralNetwork

class DeepNeuralNetwork(NeuralNetwork):
    def __init__(self, num_layers, layer_sizes, actFun_type='tanh', reg_lambda=0.01, seed=0):
        if num_layers != len(layer_sizes):
            raise ValueError(
                f"num_layers != len(layer_sizes). Invalid construction."
            )
        
        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.actFun_type = actFun_type
        self.reg_lambda = reg_lambda

        self.weights = []
        self.biases = []

        np.random.seed(seed)

        for i in range(len(layer_sizes) - 1):
            input_size = layer_sizes[i]
            output_size = layer_sizes[i + 1]

            W = np.random.randn(input_size, output_size)
            b = np.zeros((1, output_size))

            self.weights.append(W)
            self.biases.append(b)

    def feedforward(self, X):
        self.z_values = []
        self.activations = [X]

        for i in range(len(self.weights)):
            z = self.activations[i] @ self.weights[i] + self.biases[i]
            self.z_values.append(z)

            if i == len(self.weights) - 1:
                shifted_scores = z - np.max(z, axis=1, keepdims=True)
                exp_scores = np.exp(shifted_scores)
                self.probs = exp_scores / np.sum(
                    exp_scores, axis=1, keepdims=True
                )
                self.activations.append(self.probs)
            else:
                a = self.actFun(z, type=self.actFun_type)
                self.activations.append(a)

        return None


    def backprop(self, X, y):
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)

        num_examples = len(X)
        delta = self.probs.copy()
        delta[np.arange(num_examples), y] -= 1
        delta /= num_examples

        for i in range(len(self.weights) -1, -1, -1):
            dW[i] = self.activations[i].T @ delta
            db[i] = np.sum(delta, axis=0, keepdims=True)

            if i > 0:
                delta = (delta @ self.weights[i].T) * self.diff_actFun(
                    self.z_values[i-1], type=self.actFun_type
                )

        return dW, db

    def calculate_loss(self, X, y):
        num_examples = len(X)
        self.feedforward(X)

        scores = self.z_values[-1]
        shifted_scores = scores - np.max(
            scores, axis=1, keepdims=True
        )

        log_probs = shifted_scores - np.log(
            np.sum(np.exp(shifted_scores), axis=1, keepdims=True)
        )
        data_loss = -np.sum(log_probs[np.arange(num_examples), y])
        weight_penalty = (self.reg_lambda / 2) * sum(
            np.sum(W ** 2) for W in self.weights
        )

        return (data_loss + weight_penalty) / num_examples


    def fit_model(self, X, y, epsilon=0.01, num_passes=20000, print_loss=True):
        for i in range(0, num_passes):
            self.feedforward(X)
            dW, db = self.backprop(X, y)

            for j in range(len(self.weights)):
                dW[i] += (self.reg_lambda / len(X)) * self.weights[i]
                self.weights[j] += -epsilon * dW[j]
                self.biases[j] += -epsilon * db[j]

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))