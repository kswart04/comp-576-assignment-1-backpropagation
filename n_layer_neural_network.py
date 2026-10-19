from three_layer_neural_network_2026 import generate_data, plot_decision_boundary
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

    def predict(self, X):
        self.feedforward(X)
        return np.argmax(self.probs, axis=1)


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
                dW[j] += (self.reg_lambda / len(X)) * self.weights[j]
                self.weights[j] += -epsilon * dW[j]
                self.biases[j] += -epsilon * db[j]

            if print_loss and i % 1000 == 0:
                print("Loss after iteration %i: %f" % (i, self.calculate_loss(X, y)))


def run_make_moons_activation_experiment():
    X, y = generate_data()
    layer_sizes = [2, 8, 8, 8, 2]

    for activation in ("tanh", "sigmoid", "relu"):
        model = DeepNeuralNetwork(
            num_layers=len(layer_sizes),
            layer_sizes=layer_sizes,
            actFun_type=activation,
            reg_lambda=0.01,
            seed=0,
        )

        initial_loss = model.calculate_loss(X, y)
        model.fit_model(
            X, y, epsilon=0.01, num_passes=20000, print_loss=False
        )
        final_loss = model.calculate_loss(X, y)
        accuracy = np.mean(model.predict(X) == y)

        print(
            f"{activation.title()}, {len(layer_sizes)} Layers {layer_sizes}: "
            f"Initial Loss={initial_loss:.4f}, "
            f"Final Loss={final_loss:.4f}, "
            f"Training Accuracy={accuracy:.1%}"
        )

        plt.figure()
        plt.title(f"{activation.title()} decision boundary: {layer_sizes}")
        plot_decision_boundary(model.predict, X, y)


def main(num_passes=20000):
    wine = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(
        wine.data, wine.target, test_size=0.2, random_state=0,
        stratify=wine.target,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Compare activations with the same three-layer architecture.
    configurations = [
        ([13, 8, 3], "tanh"),
        ([13, 8, 3], "sigmoid"),
        ([13, 8, 3], "relu"),
    ]

    for layer_sizes, activation in configurations:
        model = DeepNeuralNetwork(
            num_layers=len(layer_sizes),
            layer_sizes=layer_sizes,
            actFun_type=activation,
            reg_lambda=0.01,
            seed=0,
        )
        initial_loss = model.calculate_loss(X_train, y_train)
        model.fit_model(
            X_train, y_train, epsilon=0.01,
            num_passes=num_passes, print_loss=False,
        )
        final_loss = model.calculate_loss(X_train, y_train)
        train_accuracy = np.mean(model.predict(X_train) == y_train)
        test_predictions = model.predict(X_test)
        test_accuracy = np.mean(test_predictions == y_test)

        print(
            f"{activation.title()}, {layer_sizes}: "
            f"Initial Loss={initial_loss:.4f}, "
            f"Final Loss={final_loss:.4f}, "
            f"Training Accuracy={train_accuracy:.1%}, "
            f"Test Accuracy={test_accuracy:.1%}",
            flush=True,
        )

        ConfusionMatrixDisplay.from_predictions(
            y_test, test_predictions,
            display_labels=wine.target_names,
        )
        plt.title(f"Wine test predictions: {activation.title()} {layer_sizes}")
        plt.show()


if __name__ == "__main__":
    main()
