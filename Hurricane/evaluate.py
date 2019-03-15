from utils import load_Hurricane_data, normalize, denormalize
import numpy as np
from tensorflow.keras.models import load_model
import sys

def PSNR(data, dec_data):
	data_range = np.max(data) - np.min(data)
	diff = data - dec_data
	rmse = np.sqrt(np.mean(diff**2))
	psnr = 20 * np.log10(data_range / rmse)
	return psnr, rmse

def evaluate_data(data, dec_data, tag):
	data = data.reshape([-1, 100, 500, 500])
	dec_data = dec_data.reshape([-1, 100, 500, 500])
	print("---------- Statistics for {} data ----------".format(tag))
	for i in range(len(data)):
		psnr, rmse = PSNR(data[i], dec_data[i])
		print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))
	print("\n\n")

def predict_and_evaluate(ratio):
	block_size_info = np.loadtxt('block_size.txt')
	block_size = block_size_info[0]
	x_train, x_test = load_Hurricane_data('Uf.dat', resize=True, block_size=block_size)
	x_train, min_train, value_range_train = normalize(x_train)
	x_test, min_test, value_range_test = normalize(x_test)
	x_train = x_train.reshape([len(x_train), block_size, block_size, 1])  # adapt this if using `channels_first` image data format
	x_test = x_test.reshape([len(x_test), block_size, block_size, 1])  # adapt this if using `channels_first` image data format
	
	parallel_autoencoder = load_model('parallel_autoencoder_{}.h5'.format(ratio))
	autoencoder = parallel_autoencoder.get_layer('autoencoder')
	encoder = autoencoder.get_layer('encoder')
	decoder = autoencoder.get_layer('decoder')

	decoded_train = parallel_autoencoder.predict(x_train)
	decoded_train = denormalize(decoded_train, min_train, value_range_train)
	decoded_train = synthesize_data(decoded_train, block_size)
	decoded_test = parallel_autoencoder.predict(x_test)
	decoded_test = denormalize(decoded_test, min_test, value_range_test)
	decoded_test = synthesize_data(decoded_train, block_size)
	decoded_test = decoded_test.reshape([-1, 100, 500, 500])

	x_train, x_test = load_Hurricane_data('Uf.dat')
	evaluate_data(x_train, decoded_train, 'training')
	evaluate_data(x_test, decoded_test, 'testing')

def evaluate_by_file(dec_train_file, dec_test_file):
	x_train, x_test = load_Hurricane_data('Uf.dat')
	decoded_train = np.fromfile(dec_train_file, dtype=np.float32)
	decoded_test = np.fromfile(dec_test_file, dtype=np.float32)
	evaluate_data(x_train, decoded_train, 'training')
	evaluate_data(x_test, decoded_test, 'testing')

