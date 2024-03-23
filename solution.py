from keras.models import load_model
import pandas as pd

df_solution = pd.read_csv('submission_sample.csv')
df_test = pd.read_csv('test_merged.csv')
X_test = df_test.iloc[:, 1:]

model = load_model('best_model.h5')
predictions = model.predict(X_test)

df_solution.iloc[:, 1] = predictions.flatten()
df_solution.to_csv('submission.csv', index=False)
