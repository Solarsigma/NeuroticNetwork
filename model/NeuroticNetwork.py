import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from model.math import Loss, Activation
from pprint import pprint


class NeuroticNetwork:
    def __init__(self, layer_structure=2, inputs=2, outputs=1, loss_fn=Loss.MSE.value, loss_params={}, learning_rate=0.05, tolerance=1e-3, max_epochs=10000, no_of_nodes_per_layer=3, activation_fn=Activation.RELU.value, activation_params={}):
        self.inputs = inputs
        self.outputs = outputs
        self.loss_fn = Loss.get_by_value(loss_fn).set_params(**loss_params)
        self.loss_history = []
        self.learning_rate = learning_rate
        self.tolerance = tolerance
        self.max_epochs = max_epochs
        self.__init_layers__(layer_structure, no_of_nodes_per_layer, activation_fn, activation_params)
        self.__init_weights__()

    
    def __init_layers__(self, layer_structure, default_no_of_nodes_per_layer, default_activation_fn, default_activation_params):
        default_activation_enum = Activation.get_by_value(default_activation_fn).set_params(**default_activation_params)
        if isinstance(layer_structure, float):
            self.layers = ([{ 'nodes': default_no_of_nodes_per_layer, 'activation_fn': default_activation_enum }] * layer_structure) + [{ 'nodes': self.outputs, 'activation_fn': Activation.IDENTITY }]
        elif isinstance(layer_structure, list) and (isinstance(layer_structure[0], float) or isinstance(layer_structure[0], int)):
            self.layers = [{ 'nodes': nodes, 'activation_fn': default_activation_enum } for nodes in layer_structure] + [{ 'nodes': self.outputs, 'activation_fn': Activation.IDENTITY }]
        else:
            for layer in layer_structure:
                if 'nodes' not in layer:
                    layer['nodes'] = default_no_of_nodes_per_layer
                if 'activation_fn' not in layer:
                    layer['activation_fn'] = default_activation_enum
                else:
                    layer['activation_fn'] = Activation.get_by_value(layer['activation_fn'])
                if 'activation_params' not in layer:
                    layer['activation_fn'].set_params(**default_activation_params)
                else:
                    layer['activation_fn'].set_params(**layer['activation_params'])
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
            post_activation_values.append(layer['activation_fn'].compute(pre_activation_values[-1]))
        return pre_activation_values, post_activation_values
    

    def train(self, x, y, verbose=False):
        self.loss_history = []
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)
        epoch_num = 1
        abs_loss_val = max(self.tolerance, 1)
        while (np.abs(abs_loss_val) > self.tolerance and epoch_num < self.max_epochs):
            if verbose:
                print(f"Epoch Number {epoch_num}")
                print("Weights:")
                pprint(self.weights)
            pre_activation_computes, post_activation_computes = self.__forward_compute__(x_train)
            loss = self.loss_fn.compute(np.atleast_2d(y_train).T, post_activation_computes[-1])
            abs_loss_val = np.sum(loss)/x.shape[0]
            if verbose:
                print(f"Loss: {abs_loss_val}")
            self.__back_propogate__(y_train.T, pre_activation_computes, post_activation_computes)
            self.loss_history.append(abs_loss_val)
            epoch_num += 1
        val_loss = self.loss_fn.compute(np.atleast_2d(y_test).T, self.predict(x_test))
        print(f"Validation Loss = {np.sum(val_loss)/x_test.shape[0]}")
        print("Final Weights: ")
        pprint(self.weights)
    

    def predict(self, x):
        _, final_computes = self.__forward_compute__(x)
        return np.squeeze(final_computes[-1].T)
    

    def __pad_ones__(self, x):
        return np.row_stack((x, np.ones((1, x.shape[1]))))
    

    def __back_propogate__(self, y_true, pre_activation_computed_values, post_activation_computed_values):
        activation_fn_d = self.layers[-1]['activation_fn'].gradient(pre_activation_computed_values[-1])
        loss_fn_grad = np.atleast_3d(self.loss_fn.gradient(np.atleast_2d(y_true), post_activation_computed_values[-1]).T)
        loss_gradient = np.reshape((activation_fn_d.T @ loss_fn_grad), loss_fn_grad.shape[:-1]).T
        for i in range(len(self.layers) - 1, -1, -1):
            prev_layer = self.layers[i-1]
            activated_layer_input = post_activation_computed_values[i]
            layer_input = pre_activation_computed_values[i]
            delta_weight = self.learning_rate * loss_gradient @ self.__pad_ones__(activated_layer_input).T
            if i > 0:
                activation_fn_d = prev_layer['activation_fn'].gradient(layer_input)
                loss_fn_grad = np.atleast_3d((self.weights[i][:, :-1].T @ loss_gradient).T)
                loss_gradient =  np.reshape((activation_fn_d.T @ loss_fn_grad), loss_fn_grad.shape[:-1]).T
            self.weights[i] -= delta_weight
    

    def get_weights(self):
        return self.weights
    

    def plot_loss_history(self):
        plt.figure()
        plt.plot(self.loss_history, color='red')
        plt.title("Loss vs Epoch Num")
        plt.show(block=False)
