from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import multi_gpu_model
from tensorflow.python.client import device_lib
from tensorflow.keras import optimizers
from build_models_500 import build_encoder, build_decoder
from utils import load_Hurricane_data, get_ratio, normalize, denormalize, synthesize_data
from evaluate import evaluate_data
import numpy as np
import sys

def get_available_gpus():
  local_device_protos = device_lib.list_local_devices()
  return [x.name for x in local_device_protos if x.device_type == 'GPU']

print("\n---------- Usage: python autoencoder_cnn.py [block_size num_filters num_layers] ----------\n")

block_size = 256
num_filters = 32
num_layers = 8
if len(sys.argv) >= 2:
  block_size = int(sys.argv[1])
if len(sys.argv) >= 3:
  num_filters = int(sys.argv[2])
if len(sys.argv) >= 4:
  num_layers = int(sys.argv[3])

input_shape = (block_size, block_size, 1)
encoder, compressed_shape, num_filters = build_encoder(input_shape, num_filters, num_layers)
ratio = get_ratio(input_shape, compressed_shape)
decoder = build_decoder(compressed_shape, num_filters, num_layers)
autoencoder = Sequential(name='autoencoder')
autoencoder.add(encoder)
autoencoder.add(decoder)

num_gpus = len(get_available_gpus())
print("\n---------- using {} gpus ----------\n".format(num_gpus))
parallel_autoencoder = multi_gpu_model(autoencoder, gpus=num_gpus, cpu_relocation=True)
opt = optimizers.Adam(lr=0.001)
# parallel_autoencoder.compile(optimizer=opt, loss='binary_crossentropy')
parallel_autoencoder.compile(optimizer=opt, loss='mean_squared_error')

x_train, x_test = load_Hurricane_data('Uf.dat', resize=True, block_size=block_size)
x_train, min_train, value_range_train = normalize(x_train)
x_test, min_test, value_range_test = normalize(x_test)
x_train = x_train.reshape([len(x_train), block_size, block_size, 1])  # adapt this if using `channels_first` image data format
x_test = x_test.reshape([len(x_test), block_size, block_size, 1])  # adapt this if using `channels_first` image data format

print("\n")
print("---------- Training data value range: {} ({} ~ {}) ----------".format(value_range_train, min_train, min_train + value_range_train))
print("---------- Testing data value range: {} ({} ~ {}) ----------".format(value_range_test, min_test, min_test + value_range_test))
print("\n")

parallel_autoencoder.fit(x_train, x_train,
    epochs=10,
    batch_size=128,
    shuffle=True,
    validation_data=(x_test, x_test))

# save model
parallel_autoencoder.save('parallel_autoencoder_{:.2f}.h5'.format(ratio))
block_size_info = np.array([1], dtype=np.int32)
block_size_info[0] = block_size
np.savetxt('block_size.txt', block_size_info, fmt='%d')
# evaluate output
decoded_train = parallel_autoencoder.predict(x_train)
decoded_train = denormalize(decoded_train, min_train, value_range_train)
decoded_train = synthesize_data(decoded_train, block_size)
decoded_test = parallel_autoencoder.predict(x_test)
decoded_test = denormalize(decoded_test, min_test, value_range_test)
decoded_test = synthesize_data(decoded_test, block_size)

x_train, x_test = load_Hurricane_data('Uf.dat')
evaluate_data(x_train, decoded_train, 'training')
evaluate_data(x_test, decoded_test, 'testing')

#encoded_train.tofile("/tmp/xin/Hurricane/encoded_cnn_train.dat")
#decoded_train.tofile("/tmp/xin/Hurricane/decoded_cnn_train.dat")
#encoded_test.tofile("/tmp/xin/Hurricane/encoded_cnn_test.dat")
#decoded_test.tofile("/tmp/xin/Hurricane/decoded_cnn_test.dat")


