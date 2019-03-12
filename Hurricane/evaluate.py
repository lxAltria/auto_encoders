from load_data import load_Hurricane_data
from assess import psnr
import numpy as np

x_train, x_test = load_Hurricane_data("Uf.dat")
x_train = x_train.reshape([-1, 100, 500, 500])
x_test = x_test.reshape([-1, 500, 500])
dec_train = np.fromfile("/tmp/xin/Hurricane/decoded_cnn_train.dat", dtype=np.float32)
dec_train = dec_train.reshape([-1, 100, 500, 500])
dec_test = np.fromfile("/tmp/xin/Hurricane/decoded_cnn_test.dat", dtype=np.float32)
dec_test = dec_test.reshape([-1, 500, 500])

print("PSNR for training data")
for i in range(len(x_train)):
	print(psnr(x_train[i], dec_train[i]))

print("PSNR for testing data")
for i in range(len(x_test)):
	print(psnr(x_test[i], dec_test[i]))

