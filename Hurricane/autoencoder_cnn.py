from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.keras import backend as K
from tensorflow.python.client import device_lib
from tensorflow.keras import optimizers
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
	# encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 256 * 256 * 64
	encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 128 * 128 * 128
	encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 64 * 64 * 256
	encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 32 * 32 * 256
	encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 16 * 16 * 512
	encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 8 * 8 * 512
	encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 4 * 4 * 512
	last_layer = encoder.layers[-1]
	shape = last_layer.output_shape
	size = 1
	for i in range(len(shape) - 1):
		size = size * shape[i+1]
	ratio = (500*500)*1.0 / size
	return encoder, ratio

def build_decoder():
	decoder = Sequential(name="decoder")
	decoder.add(Conv2D(512, (3, 3), input_shape=(4, 4, 512,), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
	decoder.add(Cropping2D((6, 6)))
	return decoder

def build_encoder_simple():
# using VGG16 architecture
	encoder = Sequential(name="encoder")
	# 500 * 500 * 1
	encoder.add(Conv2D(64, (3, 3), input_shape=(500, 500, 1,), activation='relu', padding='same'))
	# encoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 250 * 250 * 64
	encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# 125 * 125 * 64
	encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((5, 5), padding='same'))
	# 25 * 25 * 128
	encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# encoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((5, 5), padding='same'))
	# 5 * 5 * 512
	last_layer = encoder.layers[-1]
	shape = last_layer.output_shape
	size = 1
	for i in range(len(shape) - 1):
		size = size * shape[i+1]
	ratio = (500*500)*1.0 / size
	return encoder, ratio

def build_decoder_simple():
	decoder = Sequential(name="decoder")
	decoder.add(Conv2D(512, (3, 3), input_shape=(5, 5, 512,), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((5, 5), interpolation='bilinear'))
	decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((5, 5), interpolation='bilinear'))
	decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	# decoder.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
	return decoder

num_gpus = len(get_available_gpus())
# encoder, ratio = build_encoder_simple()
# decoder = build_decoder_simple()
encoder, ratio = build_encoder()
decoder = build_decoder()
autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder)
autoencoder.add(decoder)

print("\n---------- using {} gpus ----------\n".format(num_gpus))
parallel_autoencoder = multi_gpu_model(autoencoder, gpus=num_gpus)
# opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = optimizers.Adam(lr=0.001)
parallel_autoencoder.compile(optimizer=opt, loss='mean_squared_error')

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
print("\n")
print("---------- Training data value range: {} ({} ~ {}) ----------".format(value_range_train, min_train, max_train))
print("---------- Testing data value range: {} ({} ~ {}) ----------".format(value_range_test, min_test, max_test))
print("\n")

parallel_autoencoder.fit(x_train, x_train,
    epochs=30,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test))

# save model
autoencoder.save('autoencoder_{:.2f}.h5'.format(ratio))
encoder.save('encoder_{:.2f}.h5'.format(ratio))
decoder.save('decoder_{:.2f}.h5'.format(ratio))

# evaluate output
encoded_train = encoder.predict(x_train)
decoded_train = decoder.predict(encoded_train)
decoded_train = decoded_train * value_range_train + min_train
decoded_train = decoded_train.reshape([-1, 100, 500, 500])
encoded_test = encoder.predict(x_test)
decoded_test = decoder.predict(encoded_test)
decoded_test = decoded_test * value_range_test + min_test
decoded_test = decoded_test.reshape([-1, 100, 500, 500])

x_train, x_test = load_Hurricane_data("Uf.dat")
x_train = x_train.reshape([-1, 100, 500, 500])
x_test = x_test.reshape([-1, 100, 500, 500])
from assess import PSNR
print("---------- Statistics for training data ----------")
for i in range(len(x_train)):
	psnr, rmse = PSNR(x_train[i], decoded_train[i])
	print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))
print("\n\n")

print("---------- Statistics for testing data ----------")
for i in range(len(x_test)):
	psnr, rmse = PSNR(x_test[i], decoded_test[i])
	print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))

# encoded_train.tofile("/tmp/xin/Hurricane/encoded_cnn_train.dat")
# decoded_train.tofile("/tmp/xin/Hurricane/decoded_cnn_train.dat")
# encoded_test.tofile("/tmp/xin/Hurricane/encoded_cnn_test.dat")
# decoded_test.tofile("/tmp/xin/Hurricane/decoded_cnn_test.dat")


