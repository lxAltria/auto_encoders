from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import backend as K
import numpy as np
# using VGG16 architecture
encoder = Sequential(name="encoder")
# 500 * 500 * 1
encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 250 * 250 * 64
encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((2, 2), padding='same'))
# 125 * 125 * 64
encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((5, 5), padding='same'))
# 25 * 25 * 128
encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
encoder.add(MaxPooling2D((5, 5), padding='same'))
# 5 * 5 * 256

decoder = Sequential(name="decoder")
decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((5, 5)))
decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((5, 5)))
decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
decoder.add(UpSampling2D((2, 2)))
decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))

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

x_train, x_test = load_Hurricane_data("Uf48.dat")
value_range_train = np.max(x_train) - np.min(x_train)
min_train = np.min(x_train)
value_range_test = np.max(x_test) - np.min(x_test)
min_test = np.min(x_test)
x_train = (x_train.astype('float32') - min_train) / value_range_train
x_test = (x_test.astype('float32') - min_test) / value_range_test
x_train = np.reshape(x_train, (len(x_train), 500, 500, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 500, 500, 1))  # adapt this if using `channels_first` image data format

autoencoder.fit(x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test))

# save model
autoencoder.save('autoencoder_{:.2f}.h5'.format(ratio))
# save output
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = decoded_imgs * value_range_test + min_test
encoded_imgs.tofile("/tmp/xin/Hurricane/encoded_cnn_test.dat")
decoded_imgs.tofile("/tmp/xin/Hurricane/decoded_cnn_test.dat")
encoded_imgs = encoder.predict(x_train)
decoded_imgs = decoder.predict(encoded_imgs)
decoded_imgs = decoded_imgs * value_range_train + min_train
encoded_imgs.tofile("/tmp/xin/Hurricane/encoded_cnn_train.dat")
decoded_imgs.tofile("/tmp/xin/Hurricane/decoded_cnn_train.dat")
