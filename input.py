import numpy as np
import pickle

IMG_WIDTH = 32
IMG_HEIGHT = 32
IMG_DEPTH = 3
#NUM_CLASS = 10
num_class = 43

train_data_path = 'traffic_signs_data/train.p'
vali_path = 'traffic_signs_data/test.p'
#full_data_path = 'cifar10_data/data_batch_'
#vali_path = 'cifar10_data/test_batch'



def read_training_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''
    with open(train_data_path, mode='rb') as f:
        train = pickle.load(f)

    data, label = train['features'], train['labels']

    print('Reading images from ' + train_data_path)


    return data, label



def read_validation_data():
    '''
    Read in validation data. Whitening at the same time
    :return: Validation image data as 4D numpy array. Validation labels as 1D numpy array
    '''

    with open(vali_path, mode='rb') as f:
        train = pickle.load(f)

    data, label = train['features'], train['labels']

    print('Reading images from ' + train_data_path)


    return data, label
