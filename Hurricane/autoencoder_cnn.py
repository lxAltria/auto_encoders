from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from tensorflow.keras import optimizers
from build_models import build_encoder, build_decoder, build_encoder_simple, build_decoder_simple
import numpy as np

def get_available_gpus():
   local_device_protos = device_lib.list_local_devices()
   return [x.name for x in local_device_protos if x.device_type == 'GPU']

num_gpus = len(get_available_gpus())
# encoder, ratio = build_encoder_simple()
# decoder = build_decoder_simple()
encoder, ratio = build_encoder()
decoder = build_decoder()
autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder)
autoencoder.add(decoder)

print("\n---------- using {} gpus ----------\n".format(num_gpus))
parallel_autoencoder = multi_gpu_model(autoencoder, gpus=num_gpus, cpu_relocation=True)
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
    epochs=40,
    batch_size=32,
    shuffle=True,
    validation_data=(x_test, x_test))

# save model
autoencoder.set_weights(parallel_autoencoder.get_weights())
parallel_autoencoder.save_weights('parallel_autoencoder_weights_{:.2f}.h5'.format(ratio))
encoder.save_weights('encoder_weights_{:.2f}.h5'.format(ratio))
decoder.save_weights('decoder_weights_{:.2f}.h5'.format(ratio))

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

encoded_train.tofile("/tmp/xin/Hurricane/encoded_cnn_train.dat")
decoded_train.tofile("/tmp/xin/Hurricane/decoded_cnn_train.dat")
encoded_test.tofile("/tmp/xin/Hurricane/encoded_cnn_test.dat")
decoded_test.tofile("/tmp/xin/Hurricane/decoded_cnn_test.dat")


