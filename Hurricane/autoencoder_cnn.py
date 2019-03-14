from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from tensorflow.keras import optimizers
from build_models import build_encoder, build_decoder
from utils import load_Hurricane_data, get_ratio, normalize, denormalize
import numpy as np

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']

num_gpus = len(get_available_gpus())
input_shape = (500, 500, 1)
num_filters = 32
encoder, compressed_shape = build_encoder(input_shape, num_filters)
ratio = get_ratio(input_shape, compressed_shape)
decoder = build_decoder(compressed_shape, num_filters)
autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder)
autoencoder.add(decoder)

print("\n---------- using {} gpus ----------\n".format(num_gpus))
parallel_autoencoder = multi_gpu_model(autoencoder, gpus=num_gpus, cpu_relocation=True)
opt = optimizers.Adam()
parallel_autoencoder.compile(optimizer=opt, loss='mean_squared_error')

x_train, x_test = load_Hurricane_data("Uf.dat")
x_train, min_train, value_range_train = normalize(x_train)
x_test, min_test, value_range_test = normalize(x_test)
x_train = np.reshape(x_train, (len(x_train), 500, 500, 1))  # adapt this if using `channels_first` image data format
x_test = np.reshape(x_test, (len(x_test), 500, 500, 1))  # adapt this if using `channels_first` image data format

print("\n")
print("---------- Training data value range: {} ({} ~ {}) ----------".format(value_range_train, min_train, np.max(x_train)))
print("---------- Testing data value range: {} ({} ~ {}) ----------".format(value_range_test, min_test, np.max(x_test)))
print("\n")

parallel_autoencoder.fit(x_train, x_train,
    epochs=10,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test))

# save model
parallel_autoencoder.save('parallel_autoencoder_{:.2f}.h5'.format(ratio))

# evaluate output
decoded_train = parallel_autoencoder.predict(x_train)
decoded_train = denormalize(decoded_train, min_train, value_range_train)
decoded_train = decoded_train.reshape([-1, 100, 500, 500])
decoded_test = parallel_autoencoder.predict(x_test)
decoded_test = denormalize(decoded_test, min_test, value_range_test)
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

#encoded_train.tofile("/tmp/xin/Hurricane/encoded_cnn_train.dat")
#decoded_train.tofile("/tmp/xin/Hurricane/decoded_cnn_train.dat")
#encoded_test.tofile("/tmp/xin/Hurricane/encoded_cnn_test.dat")
#decoded_test.tofile("/tmp/xin/Hurricane/decoded_cnn_test.dat")


