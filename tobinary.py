import _pickle as pickle
import os
from PIL import Image as im
import numpy as np
from scipy.misc import imsave
import topng

IMAGE_PATH = 'output/'
BATCH_FILE_PATH = 'cifar-10-batches-py/'
BIN_FILE_PATH = 'binfile/'



def toCifar10(batches):

	for item in batches:

		batch_list = topng.unpickle(BATCH_FILE_PATH + '/' + item)

		if not os.path.isdir(BIN_FILE_PATH):
			os.makedirs(BIN_FILE_PATH)

		output_file = open(BIN_FILE_PATH + item + '.bin', 'wb')
		print(len(batch_list))
		for i in range(len(batch_list['filenames'])):

			get_image = im.open(IMAGE_PATH + '/' + item + '/' + batch_list['filenames'][i])

			data = np.array(get_image).transpose(2, 0, 1).flatten()

			
			label = map(int, str(batch_list['labels'][i]))

			out = np.array(list(label) + list(data), np.uint8)

			out.tofile(output_file)

		output_file.close()


if __name__ == '__main__':
	batches = {'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch'}
	toCifar10(batches)