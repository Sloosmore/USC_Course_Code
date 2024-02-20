import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
   


(train_images, train_labels), (test_images, test_labels)  = keras.datasets.cifar10.load_data()

fig, axis = plt.subplots(10, 10, figsize=(10, 10))
axis = axis.ravel()

for i in range(100):
    axis[i].imshow(train_images[i])
    axis[i].axis('off')
    
plt.savefig('normal_img')

#scale the data
X_train = train_images/255
X_test = test_images/255

#convert to grayscale
X_train_grey_img = color.rgb2gray(X_train)
X_test_grey_img = color.rgb2gray(X_test) 


fig, axis = plt.subplots(10, 10, figsize=(10, 10))
axis = axis.ravel()

#plot noisy images
for i in range(100):
    axis[i].imshow(np.clip(X_train_grey_img[i], 0.0, 1.0))    
    axis[i].axis('off')

plt.savefig('100_noisy_img')

X_train_grey_img= X_train_grey_img.reshape(-1, 32, 32, 1)
X_test_grey_img = X_test_grey_img.reshape(-1, 32, 32, 1)

#create convolutional autoencoder
#images are 32*32*1
autoencoder = tf.keras.Sequential()
autoencoder.add(tf.keras.Input(shape=(32,32,1)))
#encoding
autoencoder.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32, 1)))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
autoencoder.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
autoencoder.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

# Decoding
autoencoder.add(tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same'))

autoencoder.compile(optimizer='adam', loss='mse')


autoencoder.fit(X_train_grey_img, X_train, epochs=5, batch_size=200, shuffle=True, validation_data=(X_test_grey_img, X_test))


grey_pred_images = autoencoder.predict(X_test_grey_img[:10])
img2_array = [X_test[:10], X_test_grey_img[:10], grey_pred_images]
fig, axes = plt.subplots(3, 10, figsize=(10, 3))
for i, val in enumerate(img2_array):
    for j in range(10):
        axes[i, j].imshow(val[j])
        axes[i, j].axis('off')

plt.savefig('comparison3_img')
    

    
