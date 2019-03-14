from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D, Cropping2D
from tensorflow.keras.models import Model, Sequential
import numpy as np

def build_encoder(input_shape, num_filter1):
	# using VGG16 architecture
	encoder = Sequential(name="encoder")
	# n * n * 1
	encoder.add(Conv2D(num_filter1, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
	encoder.add(Conv2D(num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/2 * n/2 * num_filter1
	encoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/4 * n/4 * 2*num_filter1
	encoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/8 * n/8 * 2*num_filter1
	encoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/16 * n/16 * 4*num_filter1
	encoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/32 * n/32 * 4*num_filter1
	encoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	# n/64 * n/64 * 8*num_filter1
	encoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	encoder.add(MaxPooling2D((2, 2), padding='same'))
	last_layer = encoder.layers[-1]
	shape = last_layer.output_shape
	return encoder, np.delete(shape, 0)

def build_decoder(input_shape, num_filter1):
	decoder = Sequential(name="decoder")
	decoder.add(Conv2D(8*num_filter1, (3, 3), input_shape=input_shape, activation='relu', padding='same'))
	decoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(8*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(4*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(2*num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(Conv2D(num_filter1, (3, 3), activation='relu', padding='same'))
	decoder.add(UpSampling2D((2, 2), interpolation='bilinear'))
	decoder.add(Conv2D(num_filter1, (3, 3), activation='relu', padding='same'))
	# crop
	decoder.add(Cropping2D((6, 6)))
	return decoder
