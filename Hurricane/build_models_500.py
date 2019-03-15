from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
import numpy as np

def build_encoder(input_shape, num_filter1, num_layers):
	# using VGG16-like architecture
	encoder = Sequential(name='encoder')
	for i in range(num_layers):
		encoder.add(Conv2D(num_filter1, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
		encoder.add(MaxPooling2D((2, 2), padding='same'))
		if i % 2 == 1:
			num_filter1 = num_filter1 * 2
	last_layer = encoder.layers[-1]
	shape = last_layer.output_shape
	return encoder, np.delete(shape, 0), num_filter1

def build_decoder(input_shape, num_filter1, num_layers):
	decoder = Sequential(name='decoder')
	for i in reversed(range(num_layers)):
		if i % 2 == 1:
			num_filter1 = num_filter1 // 2
		decoder.add(Conv2D(num_filter1, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
		decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(1, (3, 3), activation='sigmoid', padding='same'))
	return decoder
