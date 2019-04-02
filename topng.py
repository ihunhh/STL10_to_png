import _pickle as pickle
import os
from PIL import Image as im
import numpy as np
from scipy.misc import imsave

def unpickle(file):

    with open(file, 'rb',) as fo:
        dict = pickle.load(fo, encoding = 'latin1')
    return dict

def topng(files, path, image_size):
    
    target = 'output/'
    if not os.path.isdir(target):
        os.makedirs(target)
        

    for item in files:
        
        file_list = unpickle(path + '/' + item)
        if not os.path.isdir(target + item):
            os.makedirs(target + item)
        print('Processing ' + item + '...')
        for i in range(len(file_list['data'])):
            image = np.reshape(file_list['data'][i], (3, image_size, image_size))
            image = image.transpose(1, 2, 0)
            store_path =  target + item + '/' + file_list['filenames'][i]
            imsave(store_path, image)


if __name__ == '__main__':
    
    batch_list = {'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', 'test_batch'}
    file_path = 'cifar-10-batches-py'
    image_size = 32
    if os.path.isdir(file_path):
        topng(batch_list, file_path, image_size)




 