import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from model.NeuroticNetwork import NeuroticNetwork
from model.math import Loss, Activation

def run_regression_sanity():
    nn = NeuroticNetwork(layer_structure=[3, 3], inputs=1, learning_rate=8e-5, loss_fn=Loss.MSE.value, activation_fn=Activation.LEAKY_RELU.value, max_epochs=5000, tolerance=1e-2)
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
    nn = NeuroticNetwork(layer_structure=[2, 5, 3], inputs=2, learning_rate=1e-3, loss_fn=Loss.MSE.value, activation_fn=Activation.SIGMOID.value, max_epochs=5000, tolerance=1e-2)

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

run_regression_sanity()
run_classification_sanity()