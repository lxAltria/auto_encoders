from load_data import load_Hurricane_data
from assess import PSNR
import numpy as np

x_train, x_test = load_Hurricane_data("Uf.dat")
x_train = x_train.reshape([-1, 100, 500, 500])
x_test = x_test.reshape([-1, 100, 500, 500])
dec_train = np.fromfile("/tmp/xin/Hurricane/decoded_cnn_train.dat", dtype=np.float32)
dec_train = dec_train.reshape([-1, 100, 500, 500])
dec_test = np.fromfile("/tmp/xin/Hurricane/decoded_cnn_test.dat", dtype=np.float32)
dec_test = dec_test.reshape([-1, 100, 500, 500])

print("---------- Statistics for training data ----------")
for i in range(len(x_train)):
	psnr, rmse = PSNR(x_train[i], dec_train[i])
	print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))

print("\n\n---------- Statistics for testing data ----------")
for i in range(len(x_test)):
	psnr, rmse = PSNR(x_test[i], dec_test[i])
	print("RMSE = {:.4g}, PSNR = {:.2f}".format(rmse, psnr))

