from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential

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