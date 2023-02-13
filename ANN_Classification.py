import numpy as np
import pandas as pd
import tensorflow as tf

# Check TF verion installed
# print(tf.__version__)

dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:, 3:-1].values # Get data from column 3 to the end
y = dataset.iloc[:, -1].values # Get data from the first to last row

# Encoding
from sklearn.preprocessing import LabelEncoder
le =LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2]) # Encode column 2 - gender and replace it in the dataset

# Hot Encode
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# print(x)
# France = 1.0 0.0 0.0
# Germany = 0.0 1.0 0.0
# Spain = 0.0 0.0 1.0

# Create train & test datasets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=0) # test_size = 0.2 means 20% of data will be tests and 80% train sets

# Feature scaling
from sklearn.preprocessing import StandardScaler 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Build the layers
ann = tf.keras.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))
ann.add(tf.keras.layers.Dense(units=6, activation="relu"))

ann.add(tf.keras.layers.Dense(units=1, activation="sigmoid"))

# Train the network
ann.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
ann.fit(x_train, y_train, batch_size=32, epochs=100)

# Predict results
predPercent = ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) # Predict a random entry
print(predPercent > 0.5)

x_test = np.asarray(x_test)
y_prediction = ann.predict(x_test)
y_prediction = y_prediction > 0.5
formattedPrediction = np.concatenate((y_prediction.reshape(len(y_prediction),1), y_test.reshape(len(y_test),1)),1)
print(formattedPrediction)
