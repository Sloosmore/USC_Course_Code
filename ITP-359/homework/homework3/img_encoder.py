import keras
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


(train_images, train_labels), (test_images, test_labels)  = keras.datasets.cifar10.load_data()

full_images_compare = Image.new('RGB', (train_images.shape[1]*10, train_images.shape[2]*10))

for i in range(100):
    
    im = Image.fromarray(train_images[i])
    full_images_compare.paste(im, (i%10*32, i//10*32))
    
plt.figure(figsize=(10,10))
plt.imshow(full_images_compare)
plt.savefig('normal_img')

#scale the data
X_train_img = train_images.reshape(train_images.shape[0], -1)/255
X_test_img = test_images.reshape(test_images.shape[0], -1)/255

#add noise to the data
noise_factor = 0.05
X_train_noisy_img = X_train_img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train_img.shape)
X_test_noisy_img = X_test_img + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_test_img.shape)

noisy_full_images = Image.new('RGB', (train_images.shape[1]*10, train_images.shape[2]*10))


#plot noisy images
for i in range(100):
    im = Image.fromarray((X_train_noisy_img[i]*255).reshape(32,32,3).astype('uint8'))
    noisy_full_images.paste(im, (i%10*32, i//10*32))

plt.figure(figsize=(10,10))
plt.imshow(noisy_full_images)
plt.savefig('100_noisy_img')
#plt.show()

X_train = train_images.reshape(-1, 32, 32, 3)
X_test = test_images.reshape(-1, 32, 32, 3)


#create convolutional autoencoder
#images are 32*32*3
autoencoder = tf.keras.Sequential()
autoencoder.add(tf.keras.Input(shape=(32,32,3)))
#encoding
autoencoder.add(tf.keras.layers.Conv2D((24), (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))
autoencoder.add(tf.keras.layers.Conv2D((12), (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))

#decoding
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2D((12), (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.UpSampling2D((2, 2)))
autoencoder.add(tf.keras.layers.Conv2D((24), (3,3), activation='relu', padding='same'))
autoencoder.add(tf.keras.layers.Conv2D(3, (3,3), activation='sigmoid', padding='same'))

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=10, batch_size=128, shuffle=True, validation_data=(X_test, X_test))

encoder = tf.keras.Model(inputs=autoencoder.input, outputs=autoencoder.layers[4].output)
for i in range(50):
    encoded_imgs = encoder.predict(X_test_noisy_img[i].reshape(1, 32, 32, 3))

full_images_compare = Image.new('RGB', (train_images.shape[1]*10, train_images.shape[2]*10))    
#side by side comparison of original and noisy images
for i in range(3):
    encoded_imgs = encoder.predict(X_test_noisy_img[i].reshape(1, 32, 32, 3))
    im = Image.fromarray((X_test_noisy_img[i]*255).reshape(32,32,3).astype('uint8'))
    im2 = Image.fromarray((encoded_imgs.reshape(32,32,3)*255).astype('uint8'))
    full_images_compare.paste(im, (i*32, 32))
    full_images_compare.paste(im2, (i*64, train_images.shape[2]*10))

plt.figure(figsize=(10,10))
plt.savefig('noisy_img_encoded')