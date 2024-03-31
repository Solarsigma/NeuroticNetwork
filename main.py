import numpy as np
from pprint import pprint
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RELU = 'relu'
LEAKY_RELU = 'lrelu'
TANH = 'tanh'
IDENTITY = 'identity'
STEP = 'step'
SIGMOID = 'sigmoid'
MSE = 'mse'


def activation_fn(fn_name):
    if fn_name == RELU:
        return lambda l: np.maximum(0, l)
    if fn_name == TANH:
        return lambda l: np.tanh(l)
    if fn_name == LEAKY_RELU:
        return lambda l, alpha=0.01: np.maximum(alpha * l, l)
    if fn_name == STEP:
        return lambda l: np.where(l > 0, 1, 0)
    if fn_name == SIGMOID:
        return lambda l: 1 / (1 + np.exp(-l))
    return lambda l: l


def activation_fn_derivative(fn_name):
    if fn_name == RELU:
        return lambda l: np.heaviside(l, 0)
    if fn_name == TANH:
        return lambda l: 1 - np.tanh(l)**2
    if fn_name == LEAKY_RELU:
        return lambda l, alpha=0.01: np.where(l < 0, alpha, 1)
    if fn_name == STEP:
        return lambda l: np.zeros_like(l)
    if fn_name == SIGMOID:
        return lambda l: np.exp(-l) / ((1 + np.exp(-l)) ** 2)
    return lambda l: np.ones_like(l)


def loss_fn(fn_name):
    if fn_name == MSE:
        return lambda y_true, y_pred: (y_pred - y_true)**2
    return lambda y_true, y_pred: (y_pred - y_true)


def loss_fn_gradient(fn_name):
    if fn_name == MSE:
        return lambda y_true, y_pred: 2*(y_pred - y_true)
    return lambda y_true, y_pred: np.ones_like(y_pred)


class NeuroticNetwork:
    def __init__(self, layer_structure=2, inputs=2, outputs=1, loss_fn=MSE, learning_rate=0.05, tolerance=1e-3, max_epochs=10000, no_of_nodes_per_layer=3, activation_fn=RELU):
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = loss_fn
        self.loss_history = []
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        self.__init_layers__(layer_structure, no_of_nodes_per_layer, activation_fn)
        self.__init_weights__()

    
    def __init_layers__(self, layer_structure, default_no_of_nodes_per_layer, default_activation_fn):
        if isinstance(layer_structure, float):
            self.layers = ([{ 'nodes': default_no_of_nodes_per_layer, 'activation_fn': default_activation_fn }] * layer_structure) + [{ 'nodes': self.outputs, 'activation_fn': IDENTITY }]
        elif isinstance(layer_structure, list) and (isinstance(layer_structure[0], float) or isinstance(layer_structure[0], int)):
            self.layers = [{ 'nodes': nodes, 'activation_fn': default_activation_fn } for nodes in layer_structure] + [{ 'nodes': self.outputs, 'activation_fn': IDENTITY }]
        else:
            for layer in layer_structure:
                if 'nodes' not in layer:
                    layer['nodes'] = default_no_of_nodes_per_layer
                if 'activation_fn' not in layer:
                    layer['activation_fn'] = default_activation_fn
            self.layers = layer_structure
            self.outputs = layer_structure[-1]['nodes']
    

    def __forward_propogate__(self, x, w):
        return w @ self.__pad_ones__(x)


    def __init_weights__(self):
        self.weights = []
        prev_layer_nodes = self.inputs
        for layer in self.layers:
            self.weights.append(np.random.normal(0, 1, (layer['nodes'], prev_layer_nodes + 1)))
            prev_layer_nodes = layer['nodes']
    

    def __forward_compute__(self, input):
        l = [input.T]
        pre_activation_values = [input.T]
        post_activation_values = [input.T]
        for i, layer in enumerate(self.layers):
            pre_activation_values.append(self.__forward_propogate__(post_activation_values[-1], self.weights[i]))
            post_activation_values.append(activation_fn(layer['activation_fn'])(pre_activation_values[-1]))
        return pre_activation_values, post_activation_values
    

    def train(self, x, y, verbose=False):
        self.loss_history = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        loss = epoch_num = 1
        while (np.abs(np.sum(loss)) > self.tolerance and epoch_num < self.max_epochs):
            print(f"Epoch Number {epoch_num}")
            if verbose:
                print("Weights:")
                pprint(self.weights)
            pre_activation_computes, post_activation_computes = self.__forward_compute__(x_train)
            loss = loss_fn(self.loss_fn)(y_train.T, post_activation_computes[-1])
            if verbose:
                print(f"Loss: {np.sum(loss)}")
            self.__back_propogate__(y_train.T, pre_activation_computes, post_activation_computes)
            self.loss_history.append(np.sum(loss))
            epoch_num += 1
        
        print(f"Validation Loss = {np.sum(loss_fn(self.loss_fn)(y_test.T, self.predict(x_test)))}")
        print("Final Weights: ")
        pprint(self.weights)
    

    def predict(self, x):
        _, final_computes = self.__forward_compute__(x)
        return final_computes[-1]
    

    def __pad_ones__(self, x):
        return np.row_stack((x, np.ones((1, x.shape[1]))))
    

    def __back_propogate__(self, y_true, pre_activation_computed_values, post_activation_computed_values):
        loss_gradient = activation_fn_derivative(self.layers[-1]['activation_fn'])(pre_activation_computed_values[-1]) * loss_fn_gradient(self.loss_fn)(y_true, post_activation_computed_values[-1])
        for i in range(len(self.layers) - 1, -1, -1):
            prev_layer = self.layers[i-1]
            activated_layer_input = post_activation_computed_values[i]
            layer_input = pre_activation_computed_values[i]
            delta_weight = self.learning_rate * loss_gradient @ self.__pad_ones__(activated_layer_input).T
            if i > 0:
                loss_gradient = activation_fn_derivative(prev_layer['activation_fn'])(layer_input) * (self.weights[i][:, :-1].T @ loss_gradient)
            self.weights[i] -= delta_weight
    

    def get_weights(self):
        return self.weights
    

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.loss_history, color='red')
        plt.title("Loss vs Epoch Num")
        plt.show(block=False)


def run_regression_sanity():
    nn = NeuroticNetwork(layer_structure=[3, 3], inputs=1, learning_rate=1e-6, loss_fn=MSE, activation_fn=LEAKY_RELU, max_epochs=10000)
    x = np.arange(0, 100, 0.1)
    x = x.reshape(x.shape[0], 1)
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)
    y = np.sqrt(x)

    x_test = np.array([44.5, 10.2, 1.5, 94.5, 106.3])
    x_test = x_test.reshape(x_test.shape[0], 1)
    x_test_scaled = scaler.transform(x_test)
    y_test = np.sqrt(x_test)


    nn.train(x_scaled, y, verbose=True)
    nn.plot_loss_history()

    plt.figure()
    plt.title("Test Data vs Predicted Data")
    plt.plot(x_test_scaled, y_test, 'bo')
    plt.plot(x_test_scaled, nn.predict(x_test_scaled).T, 'rs')
    plt.show(block=True)


def run_classification_sanity():
    nn = NeuroticNetwork(layer_structure=[2, 5, 3], inputs=2, learning_rate=1e-3, loss_fn=MSE, activation_fn=SIGMOID, max_epochs=5000)

    x = np.array([[x, y] for y in range(0, 100, 5) for x in range(0, 100, 5)])
    y = np.array([1 if coord[1] >= 0.036*((coord[0] - 50)**2) + 10 else 0 for coord in x])
    scaler = StandardScaler().fit(x)
    x_scaled = scaler.transform(x)

    rand_ints_1 = np.random.default_rng().integers(0, 101, size=25)
    rand_ints_2 = np.random.default_rng().integers(0, 101, size=25)
    x_test = np.array([[i, j] for i in rand_ints_1 for j in rand_ints_2])
    y_test = np.array([1 if coord[1] >= 0.036*((coord[0] - 50)**2) + 10 else 0 for coord in x_test])
    x_test_scaled = scaler.transform(x_test)


    nn.train(x_scaled, y, verbose=True)
    nn.plot_loss_history()

    plt.figure()
    plt.title("Test Data vs Predicted Data")
    colors = ['blue' if y == 0 else 'red' for y in y_test]
    plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=colors)
    plt.show(block=False)
    y_pred = nn.predict(x_test_scaled)
    y_pred_stepped = np.where(y_pred > 0.5, 1, 0).T
    colors = ['green' if y_pred_val == 0 else 'yellow' for y_pred_val in y_pred_stepped]
    plt.figure()
    plt.scatter(x_test_scaled[:, 0], x_test_scaled[:, 1], c=colors)
    plt.show(block=True)

# run_regression_sanity()
run_classification_sanity()