from load_data import load_Hurricane_data
from assess import PSNR
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

from tensorflow.keras.models import load_model
import sys
encoder = load_model("encoder_{}.h5".format(sys.argv[1]))
decoder = load_model("decoder_{}.h5".format(sys.argv[1]))

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

