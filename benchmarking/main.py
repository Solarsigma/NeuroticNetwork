from benchmarking.regression import preprocess_data, load_datasets
import os
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from tabulate import tabulate
from model.math import Activation
from model.util import HyperparameterTuner

def run_regression_benchmark():

    dataset_dir_path = os.path.abspath("/media/anand/Data/Datasets/NeuroticNetwork/House Prices - Regression Dataset")

    train_df = load_datasets(dataset_dir_path)

    ## Visualize Data
    # visualize_data(train_df)

    x_train, x_test, y_train, y_test = preprocess_data(train_df)

    # visualize_data(x_train)

    mlp_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=1e-8, max_iter=1000, tol=1e-3)

    sklearn_param_grid = {  'alpha': np.logspace(-6, -2, num=5),
                            'learning_rate_init': np.logspace(-10, -6, num=10)}
    
    sklearn_tuner = GridSearchCV(mlp_nn, param_grid=sklearn_param_grid)
    sklearn_tuner.fit(x_train, y_train)
    print("Optimum params for sklearn NN")
    print(sklearn_tuner.best_params_)
    y_pred_sklearn = sklearn_tuner.predict(x_test)

    
    custom_param_grid = { 'activation_fn': [Activation.RELU.value, Activation.LEAKY_RELU.value],
                          'learning_rate': np.logspace(-11, -9, num=5)}
    custom_nn_params = { 'inputs': x_train.shape[1], 'layer_structure': [100], 'outputs': 1, 'max_epochs': 1000, 'tolerance': 1e-3}

    custom_tuner = HyperparameterTuner(model_params=custom_nn_params, param_grid=custom_param_grid)
    custom_tuner.fit(x_train, y_train)
    print("Optimum params for custom NN")
    print(custom_tuner.get_optimum_params())
    best_model = custom_tuner.get_optimized_model()
    best_model.train(x_train, y_train)
    y_pred_custom = best_model.predict(x_test)


    print("\n\nComparison report")
    print("".join(["="]*40))

    headers = ['Metric', 'Scikit-Learn NN', 'NeuroticNetwork']
    metrics_to_eval = {'R2 Score': r2_score, 'Root Mean Squared Error': mean_squared_error, 'Explained Variance Score': explained_variance_score}
    scores = []
    for metric in metrics_to_eval:
        score = [metric]
        score.append(metrics_to_eval[metric](y_pred_sklearn, y_test))
        score.append(metrics_to_eval[metric](y_pred_custom, y_test))
        scores.append(score)

    print(tabulate(scores, headers=headers))