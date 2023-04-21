import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error


data = pd.read_csv("product_data.csv")

data = data.dropna()

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)


train_data = pd.get_dummies(train_data, columns=["PRODUCT_TYPE_ID"])

scaler = StandardScaler()
train_data[["PRODUCT_LENGTH"]] = scaler.fit_transform(train_data[["PRODUCT_LENGTH"]])
test_data[["PRODUCT_LENGTH"]] = scaler.transform(test_data[["PRODUCT_LENGTH"]])

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(train_data.shape[1]-1,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse', metrics=['mape'])

istory = model.fit(train_data.iloc[:,:-1], train_data.iloc[:,-1], epochs=10, validation_split=0.2)


test_loss, test_mape = model.evaluate(test_data.iloc[:,:-1], test_data.iloc[:,-1], verbose=0)
print("Test loss:", test_loss)
print("Test MAPE:", test_mape)


new_data = pd.read_csv("new_product_data.csv")
new_data = pd.get_dummies(new_data, columns=["PRODUCT_TYPE_ID"])
new_data[["PRODUCT_LENGTH"]] = scaler.transform(new_data[["PRODUCT_LENGTH"]])
predictions = model.predict(new_data)

np.savetxt("predictions.csv", predictions, delimiter=",")
