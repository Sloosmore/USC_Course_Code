import pandas as pd
import sklearn as skt
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

#read in file 
df = pd.read_csv('homework/homework2/Temperature data.csv', delimiter=',')
"""      
Year  Anomaly
0  1850-01    -0.36
1  1850-02    -0.12
2  1850-03    -0.23
3  1850-04    -0.19
"""
#to DateTime
df['Time'] = pd.to_datetime(df['Year'], format='%Y-%m')

df = df.drop('Year', axis=1)

#Plot the data
fig, ax = plt.subplots()
ax.plot(df['Time'], df['Anomaly'])
ax.set_xlabel('Year')
ax.set_ylabel('Temp Anomaly')
ax.set_title('Temperature Anomaly Over Time')
fig.savefig('temp_anomaly.png')

#Numpy array of anon
np_Anom = np.array(df['Anomaly']).reshape(-1,1)
np_Anom = MinMaxScaler().fit_transform(np_Anom)

window_list = []
window_size = 28
#print(np_Anom.shape[0])
max_length = np_Anom.shape[0]
y = []
#more effiecent for loop  
#need to cut last 28 rows
for i in range(window_size + 1):
    content = np_Anom[i::].flatten().tolist()
    zeros = [0] * i
    addList = content + zeros
    if (i != window_size + 1):
        window_list.append(addList)
    else:
        y.append(addList)
    
#print(window_array.shape)
time_array = np.array(window_list).reshape(np_Anom.shape[0], -1)

time_array = np.delete(time_array, slice(np_Anom.shape[0]-window_size-1, np_Anom.shape[0]), axis=0)
print(time_array.shape)


nn_model = tf.keras.Sequential()
nn_model.add(tf.keras.Input(shape=(window_size,)))
nn_model.add(tf.keras.layers.Dense(100))
nn_model.add(tf.keras.layers.Dense(50))
nn_model.add(tf.keras.layers.Dense(10))
nn_model.add(tf.keras.layers.Dense(1))
nn_model.compile(
    optimizer='adam',
    loss="mse",
)



