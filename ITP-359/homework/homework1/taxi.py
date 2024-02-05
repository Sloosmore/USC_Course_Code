import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from datetime import datetime
from sklearn.model_selection import train_test_split

#steps done slighly different order

#Import CSV
df = pd.read_csv('homework/homework1/Taxi Fares.csv')

#drop key
df = df.drop(columns=['key'])

#Replace 0's with nan for droping
space_array = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude']
for i in space_array:
    df[i] = df[i].replace(0, np.nan)
    
#drop bad rows 
df = df.drop_duplicates()
df = df.dropna(axis = 0)

#split data
y = df['fare_amount']
X = df.drop(columns=['fare_amount'])

#datetime fomrating
format_data = '%Y-%m-%d %H:%M:%S UTC'
Dt = pd.to_datetime(X['pickup_datetime'], format=format_data)
hours = Dt.dt.hour
min = Dt.dt.minute

#add other cols in
X['min'] = hours * 60 + min
X['day_of_week'] = Dt.dt.dayofweek
X['distance'] = np.sqrt(((X['pickup_longitude'] - X['dropoff_longitude'])*54.6)**2 + ((X['pickup_latitude'] - X['dropoff_latitude'])*69)**2)

#drop everything
X = X.drop(columns=['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude'])

#check 
print(X.shape)
print(X.columns)
print(X.info())

#define tf model 
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(4,)))
model.add(tf.keras.layers.Dense(1024, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(1))
model.compile(optimizer='Adam', loss='mse')
#split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

#fit
trained_model = model.fit(X_train, y_train, batch_size=50, epochs=20, validation_data=(X_test, y_test) )

#plot
array = np.arange(20)
fig, ax = plt.subplots(1)
ax.plot(array, trained_model.history['loss'], label='Train Loss')
ax.plot(array, trained_model.history['val_loss'], label='Val Loss')
ax.set_title('Training and val accuarcy')
ax.set_xlabel('epoch')
ax.set_ylabel('mse loss')
ax.legend()
fig.savefig('mse-loss')

#transform y test for pred
y_test = y_test.to_numpy()[:, np.newaxis]

#compute r2
y_pred = model.predict(X_test)
metric = tf.keras.metrics.R2Score()
metric.update_state(y_test, y_pred)
result = metric.result()
score = result.numpy()
print('R2 Score: ', score)
#0.7261578

#pred final val
pred_array = np.array([2, 15*60+20,5, 3.2])[np.newaxis, :]
final_pred = model.predict(pred_array)

print(final_pred)
#[[14.548513]]
