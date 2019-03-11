from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
import numpy as np

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']

def build_encoder():
	# using VGG16 architecture
	encoder = Sequential(name="encoder")
	# 500 * 500 * 1
	encoder.add(ZeroPadding2D((6, 6), input_shape=(500, 500, 1,)))
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
	return encoder, (500*500)*1.0 / (8*8*256)

def build_encoder_simple():
# using VGG16 architecture
	encoder = Sequential(name="encoder")
	# 500 * 500 * 1
	encoder.add(Conv2D(64, (3, 3), input_shape=(500, 500, 1,), activation='relu', padding='same'))
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
	return encoder, (500*500)*1.0 / (5*5*256)

def build_decoder():
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
	return decoder

def build_decoder_simple():
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
	return decoder

num_gpus = len(get_available_gpus())
encoder, ratio = build_encoder_simple()
decoder = build_decoder_simple()
autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder)
autoencoder.add(decoder)

print("\n---------- using {} gpus ----------\n".format(num_gpus))
parallel_autoencoder = multi_gpu_model(autoencoder, gpus=num_gpus)
parallel_autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

from load_data import load_Hurricane_data
import numpy as np

x_train, x_test = load_Hurricane_data("Uf.dat")
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
min_train = np.min(x_train)
max_train = np.max(x_train)
value_range_train = max_train - min_train
min_test = np.min(x_test)
max_test = np.max(x_test)
value_range_test = max_test - min_test
x_train = (x_train - min_train) / value_range_train
x_test = (x_test - min_test) / value_range_test
x_train = np.reshape(x_train, (len(x_train), 500, 500, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 500, 500, 1))  # adapt this if using `channels_first` image data format
print("\n---------- Training data value range: {} ({} ~ {}) ----------".format(value_range_train, min_train, max_train))
print("---------- Testing data value range: {} ({} ~ {}) ----------\n".format(value_range_test, min_test, max_test))

parallel_autoencoder.fit(x_train, x_train,
    epochs=50,
    batch_size=32,
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
