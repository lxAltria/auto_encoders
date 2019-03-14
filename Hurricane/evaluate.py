from utils import load_Hurricane_data, normalize, denormalize
from assess import PSNR
import numpy as np
from tensorflow.keras.models import load_model
import sys

def predict_and_evaluate(ratio):
	x_train, x_test = load_Hurricane_data("Uf.dat")
	x_train, min_train, value_range_train = normalize(x_train)
	x_test, min_test, value_range_test = normalize(x_test)
	x_train = np.reshape(x_train, (len(x_train), 500, 500, 1))  # adapt this if using `channels_first` image data format
	x_test = np.reshape(x_test, (len(x_test), 500, 500, 1))  # adapt this if using `channels_first` image data format

	parallel_autoencoder = load_model("parallel_autoencoder_{}.h5".format(ratio))
	autoencoder = parallel_autoencoder.get_layer('autoencoder')
	encoder = autoencoder.get_layer('encoder')
	decoder = autoencoder.get_layer('decoder')

	encoded_train = encoder.predict(x_train)
	decoded_train = decoder.predict(encoded_train)
	encoded_test = encoder.predict(x_test)
	decoded_test = decoder.predict(encoded_test)
	decoded_train = denormalize(decoded_train, min_train, value_range_train)
	decoded_train = decoded_train.reshape([-1, 100, 500, 500])
	decoded_test = denormalize(decoded_test, min_test, value_range_test)
	decoded_test = decoded_test.reshape([-1, 100, 500, 500])

	x_train, x_test = load_Hurricane_data("Uf.dat")
	x_train = x_train.reshape([-1, 100, 500, 500])
	x_test = x_test.reshape([-1, 100, 500, 500])

	print("---------- Statistics for training data ----------")
	for i in range(len(x_train)):
		psnr, rmse = PSNR(x_train[i], decoded_train[i])
		print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))
	print("\n\n")

	print("---------- Statistics for testing data ----------")
	for i in range(len(x_test)):
		psnr, rmse = PSNR(x_test[i], decoded_test[i])
		print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))

def evaluate(dec_train_file, dec_test_file):
	x_train, x_test = load_Hurricane_data("Uf.dat")
	x_train = x_train.reshape([-1, 100, 500, 500])
	x_test = x_test.reshape([-1, 100, 500, 500])
	decoded_train = np.fromfile(dec_train_file, dtype=np.float32)
	decoded_test = np.fromfile(dec_test_file, dtype=np.float32)
	decoded_train = decoded_train.reshape([-1, 100, 500, 500])
	decoded_test = decoded_test.reshape([-1, 100, 500, 500])
	print("---------- Statistics for training data ----------")
	for i in range(len(x_train)):
		psnr, rmse = PSNR(x_train[i], decoded_train[i])
		print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))
	print("\n\n")
	print("---------- Statistics for testing data ----------")
	for i in range(len(x_test)):
		psnr, rmse = PSNR(x_test[i], decoded_test[i])
		print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))

if(len(sys.argv) > 2):
	evaluate(sys.argv[1], sys.argv[2])
else:
	predict_and_evaluate(sys.argv[1])
