import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


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

#add noise to the data
noise_factor = 0.1
X_train_noisy_img = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)
X_test_noisy_img = X_test + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test.shape)


fig, axis = plt.subplots(10, 10, figsize=(10, 10))
axis = axis.ravel()

#plot noisy images
for i in range(100):
    axis[i].imshow(np.clip(X_train_noisy_img[i], 0.0, 1.0))    
    axis[i].axis('off')

plt.savefig('100_noisy_img')



#create convolutional autoencoder
#images are 32*32*3
autoencoder = tf.keras.Sequential()
autoencoder.add(tf.keras.Input(shape=(32,32,3)))
#encoding
#need to learn more about padding and network architecture
autoencoder.add(tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=(32,32,3)))
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


autoencoder.fit(X_train_noisy_img, X_train, epochs=5, batch_size=200, shuffle=True, validation_data=(X_test_noisy_img, X_test))

noisy_pred_images = autoencoder.predict(X_test_noisy_img[:50])
img_array = [X_test[:50], X_test_noisy_img[:50], noisy_pred_images]

for i, img_set in enumerate(img_array):
    fig, axes = plt.subplots(5, 10, figsize=(10, 5))
    axes = axes.ravel()
    for j, img in enumerate(img_set):
        axes[j].imshow(img)
        axes[j].axis('off')
    plt.savefig(f'comparison_img{i}')

noisy_pred_images2 = autoencoder.predict(X_test_noisy_img[:3])
img2_array = [X_test[:3], X_test_noisy_img[:3], noisy_pred_images2[:3]]
fig, axes = plt.subplots(3, 3, figsize=(3, 3))
for i, val in enumerate(img2_array):
    for j in range(3):
        axes[i, j].imshow(val[j])
        axes[i, j].axis('off')

plt.savefig('comparison3_img')

#need to learn about imshow

    
