from benchmarking.util import load_dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

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