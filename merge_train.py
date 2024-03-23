import pandas as pd
from sklearn.neighbors import NearestNeighbors

train_df = pd.read_csv('train.csv')
features_df = pd.read_csv('features.csv')

features_df.reset_index(drop=True, inplace=True)

nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features_df[['lat', 'lon']])

_, train_indices = nn_model.kneighbors(train_df[['lat', 'lon']])

train_merged = pd.concat([train_df, features_df.iloc[train_indices[:, 0], 2:].reset_index(drop=True)], axis=1)

score_column = train_merged.pop('score')
train_merged.insert(1, 'score', score_column)

train_merged.to_csv('train_merged.csv', index=False)
