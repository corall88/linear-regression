import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Загрузка данных
train_df = pd.read_csv('train.csv')
features_df = pd.read_csv('features.csv')

# Убедимся, что индекс уникален
features_df.reset_index(drop=True, inplace=True)

# Обучение модели ближайших соседей только на обучающих данных
nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features_df[['lat', 'lon']])

# Нахождение ближайших точек для обучающей и тестовой выборок
_, train_indices = nn_model.kneighbors(train_df[['lat', 'lon']])

# Объединение признаков из ближайших точек с обучающей выборкой
train_merged = pd.concat([train_df, features_df.iloc[train_indices[:, 0], 2:].reset_index(drop=True)], axis=1)

# Удаление столбца score из train_merged и сохранение его в переменной
score_column = train_merged.pop('score')
# Вставка столбца score обратно на первую позицию в train_merged
train_merged.insert(1, 'score', score_column)


# Сохранение обработанных данных в файлы
train_merged.to_csv('train_merged.csv', index=False)