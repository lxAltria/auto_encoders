import numpy as np
import os

def load_Hurricane_data(filename, folder="/tmp/xin/Hurricane", resize=False, block_size=128):
	if not resize:
		data = np.fromfile("{}/{}".format(folder, filename), dtype=np.float32)
		data = data.reshape([-1, 500, 500, 1])
	else:
		fpath = "/tmp/xin/Hurricane/resized_{}_{}".format(block_size, filename)
		if os.path.exists(fpath):
			data = np.fromfile(fpath, dtype=np.float32)
		else:
			data = np.fromfile("{}/{}".format(folder, filename), dtype=np.float32)
			data = data.reshape([-1, 500, 500])
			data = resize_data(data, block_size)
			data.tofile(fpath)
	num = len(data)
	split_ind = num * 40 // 48
	return data[:split_ind], data[split_ind:]

def get_ratio(input_shape, output_shape):
	input_size = 1
	output_size = 1
	for i in range(len(input_shape)):
		input_size = input_size * input_shape[i]
		output_size = output_size * output_shape[i]
	ratio = input_size*1.0 / output_size
	return ratio

def normalize(data):
	min_data = np.min(data)
	max_data = np.max(data)
	value_range = max_data - min_data
	data = (data - min_data) / value_range
	return data, min_data, value_range

def denormalize(data, min_data, value_range):
	data = data * value_range + min_data
	return data

def get_split_positions(block_size, block_num):
	overlapped = block_num * block_size - 500
	prev_overlapped = overlapped // (block_num - 1)
	split_ind = (block_num - 1) - overlapped % (block_num - 1)
	pos = np.ndarray([block_num], dtype=np.int32)
	pos[0] = 0
	for i in range(block_num - 1):
		pos[i+1] = pos[i] + (block_size - prev_overlapped)
		if i+1 > split_ind:
			pos[i+1] = pos[i+1] - 1
	return pos

def resize_data(data, block_size):
	n = 500 // block_size
	if (n % block_size) != 0:
		n = n + 1
	num = len(data)
	pos = get_split_positions(block_size, n)
	resized_data = np.ndarray([num, n, n, block_size, block_size])
	for i in range(num):
		for j in range(n):
			for k in range(n):
				resized_data[i, j, k] = data[i, pos[j]:pos[j] + block_size, pos[k]:pos[k] + block_size]
	return resized_data.reshape([num * n * n, block_size, block_size, 1])

def synthesize_data(data, block_size):
	n = 500 // block_size
	if (n % block_size) != 0:
		n = n + 1
	num = len(data) // n // n
	pos = get_split_positions(block_size, n)
	data_syn = np.zeros([num, 500, 500], dtype=np.float32)
	tmp_data = data.reshape([num, n, n, block_size, block_size])
	for i in range(num):
		for j in range(n):
			for k in range(n):
				data_syn[i, pos[j]:pos[j] + block_size, pos[k]:pos[k] + block_size] = data_syn[i, pos[j]:pos[j] + block_size, pos[k]:pos[k] + block_size] + tmp_data[i, j, k]
	count_matrix = np.ones([500, 500], dtype=np.int8)
	for j in range(n):
		for k in range(n):
			count_matrix[pos[j]:pos[j] + block_size, pos[k]:pos[k] + block_size] = count_matrix[pos[j]:pos[j] + block_size, pos[k]:pos[k] + block_size] + 1
	data_syn = data_syn / count_matrix
	return data_syn

