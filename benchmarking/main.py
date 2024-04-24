import pandas as pd
from benchmarking.regression.main import visualize_data, load_datasets
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from model.NeuroticNetwork import NeuroticNetwork
from model.math.score import score
from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
from tabulate import tabulate

def reg_benchmark():

    dataset_dir_path = os.path.abspath("/media/anand/Data/Datasets/NeuroticNetwork/House Prices - Regression Dataset")

    train_df = load_datasets(dataset_dir_path)

    ## Visualize Data
    visualize_data(train_df)

    imputer = IterativeImputer(max_iter=10)
    train_df['LotFrontage'] = pd.DataFrame(imputer.fit_transform(np.array(train_df['LotFrontage']).reshape(-1,1)))
    train_df['GarageYrBlt'] = pd.DataFrame(np.round(imputer.fit_transform(np.array(train_df['GarageYrBlt']).reshape(-1,1))))
    train_df['MasVnrArea'] = pd.DataFrame(imputer.fit_transform(np.array(train_df['MasVnrArea']).reshape(-1,1)))


    pd.set_option('future.no_silent_downcasting', True)
    cols_to_encode = [col for col in train_df if train_df[col].dtype == 'object']
    for col in cols_to_encode:
        uniq_vals = pd.unique(train_df[col])
        train_df[col] = train_df[col].replace(uniq_vals, np.arange(1, len(uniq_vals)+1))

    col_to_predict = 'SalePrice'
    y = train_df.get(col_to_predict).copy()

    cols_to_remove = ['Id']
    cols_to_remove.append(col_to_predict)
    x = train_df.drop(cols_to_remove, axis=1)

    visualize_data(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = pd.DataFrame(scaler.transform(x_train))
    x_test_scaled = pd.DataFrame(scaler.transform(x_test))

    pca = PCA(n_components='mle')
    pca.fit(x_train_scaled)

    x_train_fin = pd.DataFrame(pca.transform(x_train_scaled))
    x_test_fin = pd.DataFrame(pca.transform(x_test_scaled))

    mlp_nn = MLPRegressor(hidden_layer_sizes=(100,), activation='relu', solver='sgd', learning_rate_init=1e-8, max_iter=1000, tol=1e-3)

    mlp_nn.fit(x_train_fin, y_train)


    plt.plot(mlp_nn.loss_curve_)
    plt.show(block=False)
    y_pred_sklearn = mlp_nn.predict(x_test_fin)

    custom_nn = NeuroticNetwork(inputs=x_train_fin.shape[1], layer_structure=[100], outputs=1, learning_rate=1e-10, max_epochs=1000, tolerance=1e-3)

    custom_nn.train(x_train_fin, y_train)
    custom_nn.plot_loss_history()
    plt.show()
    y_pred_custom = custom_nn.predict(x_test_fin)\

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





    ## scikit learn NN w/ hypetuning

    ## custom NN w/ hypetuning

    ## maybe some other NN w/ hypetuning