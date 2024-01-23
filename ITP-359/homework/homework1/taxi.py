import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

df = pd.read_csv('ITP-359/homework/homework1/Taxi Fares.csv')
print(df.columns)


df = df.drop(columns=['passenger_count', 'key'])
space_array = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude']
for i in space_array:
    df[i] = df[i].replace(0, np.nan)
    

df = df.drop_duplicates()
df = df.dropna(axis = 0)

print(df.info())

# we are guessing that the taxi charge is a flat rate so we can drop the passanger_count