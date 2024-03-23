import pandas as pd
from sklearn.neighbors import NearestNeighbors

test_df = pd.read_csv('test.csv')
features_df = pd.read_csv('features.csv')

features_df.reset_index(drop=True, inplace=True)

nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features_df[['lat', 'lon']])

_, test_indices = nn_model.kneighbors(test_df[['lat', 'lon']])

test_merged = pd.concat([test_df, features_df.iloc[test_indices[:, 0], 2:].reset_index(drop=True)], axis=1)

test_merged.to_csv('test_merged.csv', index=False)
