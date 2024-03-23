import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.regularizers import l1_l2
from tensorflow.keras.callbacks import ModelCheckpoint

EPOCHS = 180
BATCH_SIZE = 50
LEARNING_RATE = 0.0003
FILENAME = "train_merged.csv"

def prepare_data(filename):
    df_train = pd.read_csv(filename)
    n_features = len(df_train.columns) - 2
    X_train = df_train.iloc[:, 2:]
    y_train = df_train.iloc[:, 1]
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_test, y_test, n_features

def build_model(n_features, learning_rate):
    model = Sequential([
    Dense(512, input_dim=n_features, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_absolute_error')

    return model

data = prepare_data(FILENAME)
X_train = data[0]
y_train = data[1]
X_test = data[2]
y_test = data[3]
n_features = data[4]

model = build_model(n_features, LEARNING_RATE)

checkpoint_callback = ModelCheckpoint(filepath='best_model.h5', 
                                      monitor='val_loss', 
                                      save_best_only=True,
                                      verbose=0)

history = model.fit(x=X_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1, validation_data=(X_test, y_test), callbacks=[checkpoint_callback])
