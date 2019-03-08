from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.models import Model, Sequential
from keras import backend as K
import numpy as np
# using VGG16 architecture
encoder = Sequential(name="encoder")
# 500 * 500 * 1
encoder.add(ZeroPadding2D((6, 6)))
# 512 * 512 * 1
encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 250 * 250 * 64
encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 125 * 125 * 64
encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 63 * 63 * 128
encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 32 * 32 * 128
encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 16 * 16 * 256
encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 8 * 8 * 256

decoder = Sequential(name="decoder")
decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
decoder.add(Cropping2D((6, 6)))

autoencoder = Sequential(name="autoencoder")
autoencoder.add(encoder)
autoencoder.add(decoder)
autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

# data = np.zeros([4, 500, 500, 1])
# tmp = autoencoder.predict(data)
# autoencoder.summary()
# encoder.summary()
# decoder.summary()
# exit(0)

from load_data import load_Hurricane_data
import numpy as np

x_train, x_test = load_Hurricane_data("Uf.dat")
value_range_train = np.max(x_train) - np.min(x_train)
min_train = np.min(x_train)
value_range_test = np.max(x_test) - np.min(x_test)
min_test = np.min(x_test)
x_train = (x_train.astype('float32') - min_train) / value_range_train
x_test = (x_test.astype('float32') - min_test) / value_range_test
x_train = np.reshape(x_train, (len(x_train), 500, 500, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 500, 500, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = (decoded_imgs + min_test) * value_range_test
encoded_imgs.tofile("encoded_cnn.dat")
decoded_imgs.tofile("decoded_cnn.dat")