import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Загрузка данных
test_df = pd.read_csv('test.csv')
features_df = pd.read_csv('features.csv')

# Убедимся, что индекс уникален
features_df.reset_index(drop=True, inplace=True)

# Обучение модели ближайших соседей
nn_model = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(features_df[['lat', 'lon']])

# Нахождение ближайших точек для обучающей и тестовой выборок
_, test_indices = nn_model.kneighbors(test_df[['lat', 'lon']])

# Объединение признаков из ближайших точек с обучающей и тестовой выборками
test_merged = pd.concat([test_df, features_df.iloc[test_indices[:, 0], 2:].reset_index(drop=True)], axis=1)

# Проверка результатов
print("Тестовая выборка после объединения с признаками:")
print(test_merged.columns)


test_merged.to_csv('test_merged.csv', index=False)
