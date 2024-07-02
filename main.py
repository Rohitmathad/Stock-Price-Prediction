'''import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn import linear_model
from keras. models import Sequential
from keras. layers import Dense
from keras.optimizers import Adam
import keras. backend as K
from keras. callbacks import EarlyStopping
import pydot
from keras. models import load_model
from keras. layers import LSTM
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.utils import plot_model

from sklearn.model_selection import TimeSeriesSplit

dt = pd.read_csv("C:/Users/Rohit/Downloads/stock price prediction/Stock_Data/ADANIPORTS.csv")

dt.head()
scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(dt[features])
feature_transform= pd.DataFrame(
columns=features,data=feature_transform, index=dt.index)
feature_transform.head()
# Assuming trainX and trainY are your training data
# Ensure trainX is 3D for LSTM and trainY is 2D
trainX = np.random.rand(1000, 10, 1)  # Example shape, replace with your data
trainY = np.random.rand(1000, 1)      # Example shape, replace with your data
X_test = np.random.rand(100, 10, 1)   # Example shape, replace with your data
y_test = np.random.rand(100, 1)       # Example shape, replace with your data

# Normalize the data
scaler_X = MinMaxScaler(feature_range=(0, 1))
scaler_Y = MinMaxScaler(feature_range=(0, 1))

# Reshape trainX and X_test for scaler
trainX_reshaped = trainX.reshape(-1, trainX.shape[-1])
X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

trainX_scaled = scaler_X.fit_transform(trainX_reshaped).reshape(trainX.shape)
X_test_scaled = scaler_X.transform(X_test_reshaped).reshape(X_test.shape)

# Reshape trainY and y_test for scaler
trainY_scaled = scaler_Y.fit_transform(trainY)
y_test_scaled = scaler_Y.transform(y_test)
timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
# Define the LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(trainX_scaled.shape[1], trainX_scaled.shape[2]), activation='relu', return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, activation='relu', return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_absolute_error')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

# Train the model
history = model.fit(trainX_scaled, trainY_scaled, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping, reduce_lr])

# Predict using the model
y_pred_scaled = model.predict(X_test_scaled)

# Inverse transform the predictions and true values
y_pred = scaler_Y.inverse_transform(y_pred_scaled)
y_test_original = scaler_Y.inverse_transform(y_test_scaled)

# Calculate MAE
mae = mean_absolute_error(y_test_original, y_pred)
print(f"MAE: {mae}")

# Plot the results
import matplotlib.pyplot as plt

plt.plot(y_test_original, label='True Value')
plt.plot(y_pred, label='LSTM Prediction')
plt.title("LSTM Prediction vs True Value")
plt.xlabel('Time Scale')
plt.ylabel('Value')
plt.legend()
plt.show()
'''