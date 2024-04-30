from benchmarking.util import load_dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def visualize_data(df):
    print(df)
    print("\nColumn-wise Details: \n")
    non_obj_cols = []
    for col in df:
        print(f"Column {col} has datatype {df.get(col).dtype}")
        uniq_vals = pd.unique(df.get(col))
        print(f"\t No. of unique vals = {len(uniq_vals)}")
        if df.get(col).dtype == 'object':
            if len(uniq_vals) <= 10:
                print(f"\t Unique values: {uniq_vals}")
        else:
            print(f"\t Minimum = {min(uniq_vals)}, Maximum = {max(uniq_vals)}")
            non_obj_cols.append(col)
        no_of_nan_vals = df.get(col).isnull().sum()
        print(f"\t No. of NaN values = {no_of_nan_vals}")
        print()
    print("Correlation matrix between features")
    corr = df[non_obj_cols[1:]].corr(method='spearman')
    # plt.subplots(figsize=(20,20))
    # sns.heatmap(abs(corr))
    # plt.show()


def load_datasets(dataset_dir_path):
    train_dataset_filename = "train.csv"
    return load_dataset(os.path.join(dataset_dir_path, train_dataset_filename))


def preprocess_data(train_df):
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

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_scaled = pd.DataFrame(scaler.transform(x_train))
    x_test_scaled = pd.DataFrame(scaler.transform(x_test))

    pca = PCA(n_components='mle')
    pca.fit(x_train_scaled)

    x_train_fin = pd.DataFrame(pca.transform(x_train_scaled))
    x_test_fin = pd.DataFrame(pca.transform(x_test_scaled))

    return x_train_fin, x_test_fin, y_train, y_test