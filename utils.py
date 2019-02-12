import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from input import *

BN_EPSILON = 0.001

#Gray scale
def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    #return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Definition of gray function
def F_gray(X_Arr):
    X_out = np.zeros((len(X_Arr),32,32,1))
    #X_out = np.zeros_like(X_Arr)
    for i in range(len(X_Arr)):
        img = X_Arr[i].squeeze()
        X_out[i,:,:,0] = grayscale(img)
    return X_out


# Distortion function
def F_distort(X_Arr):
    X_out = np.zeros_like(X_Arr)
    for i in range(len(X_Arr)):
        # Parameters
        angle = (15 * 2 * (np.random.rand() - .5))  # From -15 to 15 degrees
        Transl_x = 2 * 2 * (np.random.rand() - .5)  # From -2 to 2 pixels
        Transl_y = 2 * 2 * (np.random.rand() - .5)  # From -2 to 2 pixels

        img = X_Arr[i]
        M = cv2.getRotationMatrix2D((16, 16), angle, 1)
        M[0, 2] = Transl_x
        M[1, 2] = Transl_y

        # X_out[i,:,:,:] = cv2.warpAffine(img,M,(32,32))

        imgswap = cv2.warpAffine(img, M, (32, 32))

        # Perspective Transformation
        d1 = 2 * (np.random.rand() - .5)
        d2 = 2 * (np.random.rand() - .5)

        pts1 = np.float32([[2, 2], [30, 2], [0, 30], [30, 30]])
        pts2 = np.float32([[d1, d2], [30 + d1, d2], [d1, 30 + d2], [30 + d1, 30 + d2]])
        # pts2 = np.float32([[0,0],[32,0],[0,32],[32,32]])
        M = cv2.getPerspectiveTransform(pts1, pts2)

        X_out[i, :, :, :] = cv2.warpPerspective(imgswap, M, (32, 32))

    return X_out



def preProcessingData(X_train, y_train):

    n_train = len(X_train)
    image_shape = X_train[0].squeeze().shape
    n_classes = max(y_train) - min(y_train) + 1

    # Count samples in each set
    count_train = np.zeros(n_classes)
    count_labels = np.zeros(n_classes)

    for i in range(n_classes):
        count_labels[i] = i

    for i in range(n_train):
        idx = int(y_train[i])
        count_train[idx] += 1

    # Count upper and lower bounds of classes
    limbounds = np.zeros([n_classes, 2])

    limbounds[0, 0] = 0
    limbounds[0, 1] = count_train[0]

    for i in range(1, n_classes):
        limbounds[i, 0] = limbounds[i - 1, 1] + 1
        limbounds[i, 1] = limbounds[i, 0] + count_train[i] - 1

    # Inclusion of warped images
    print(
        "The number of warped images to be included in the train data is inversily proportional to the number of samples:")
    print("  Sets with less than 1000 samples will have 200% increase")
    print("  Sets with less than 2000 samples will have 100% increase")
    print("  Sets with greater than 2000 samples will have 50% increase")

    # Distort
    X_out = X_train
    y_out = y_train

    print("X_train.shape original: ", X_train.shape)
    print("y_train.shape original: ", y_train.shape)

    for n in range(n_classes):  # For each class
        if count_train[n] < 1000:
            # Add twice
            tini = int(limbounds[n, 0])
            tend = int(limbounds[n, 1])
            X_swap = X_train[tini:tend, :, :, :]
            y_d1 = y_train[tini:tend]
            y_d2 = y_train[tini:tend]

            # Transformation
            X_d1 = F_distort(X_swap)
            X_d2 = F_distort(X_swap)

            X_out = np.concatenate([X_out, X_d1, X_d2])
            y_out = np.concatenate([y_out, y_d1, y_d2])

        elif count_train[n] < 2000:
            tini = int(limbounds[n, 0])
            tend = int(limbounds[n, 1])
            X_swap = X_train[tini:tend, :, :, :]
            y_d1 = y_train[tini:tend]

            # Transformation
            X_d1 = F_distort(X_swap)
            X_out = np.concatenate([X_out, X_d1])
            y_out = np.concatenate([y_out, y_d1])

        else:
            tini = int(limbounds[n, 0])
            tend = int((limbounds[n, 1] + limbounds[n, 0]) / 2)
            X_swap = X_train[tini:tend, :, :, :]
            y_d1 = y_train[tini:tend]

            # Transformation
            X_d1 = F_distort(X_swap)
            X_out = np.concatenate([X_out, X_d1])
            y_out = np.concatenate([y_out, y_d1])

    # Transform in gray
    X_train_gray = F_gray(X_out)

    # Normalize images to [0,1]
    X_train_norm = X_train_gray / 255

    # Shuffle
    from sklearn.utils import shuffle

    X_train_shuff, y_train_shuff = shuffle(X_train_norm, y_out)

    print("X_train.shape agumented: ", X_train_shuff.shape)
    print("y_train.shape agumented: ", y_train_shuff.shape)

    return  X_train_shuff, y_train_shuff



def preProcessingImageData(Xdata):

    # Transform in gray
    X_train_gray = F_gray(Xdata)

    # Normalize images to [0,1]
    X_train_norm = X_train_gray / 255


    print("X_train.shape : ", X_train_norm.shape)

    return X_train_norm

def resizeImageData(Xdata, resize):
    if resize == 32:
        resizedData = Xdata
    else:
        temp_x = tf.image.resize_images(Xdata, [resize, resize])
        resizedData = tf.image.resize_images(temp_x, [32, 32])

    return resizedData

def display_random_image(images, predics, labels) :
    import matplotlib.pyplot as plt
    n = len(images)
    rows, cols = 5, 5
    fig, ax_array = plt.subplots(rows, cols)
    #plt.figure(1)
    plt.suptitle('wrong inference Images')
    for idx, ax in enumerate(ax_array.ravel()):
        # show a random image of the current class
        ax.imshow(images[idx])  # , cmap='gray')
        ax.set_title('label{:02d} pred{:02d}'.format(predics[idx], labels[idx]))

    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()
    plt.show()
#################################################################################3





def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def batch_normalization_layer(name, input_layer, dimension, is_training=True):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        bn_layer = tf.identity(input_layer)

    return bn_layer

def residual_block(input_layer, output_channel, is_training, first_block=False):
    '''
    Defines a residual block in ResNet
    :param input_layer: 4D tensor
    :param output_channel: int. return_tensor.get_shape().as_list()[-1] = output_channel
    :param first_block: if this is the first residual block of the whole network
    :return: 4D tensor.
    '''
    input_channel = input_layer.get_shape().as_list()[-1]

    # When it's time to "shrink" the image size, we use stride = 2
    if input_channel * 2 == output_channel:
        increase_dim = True
        stride = 2
    elif input_channel == output_channel:
        increase_dim = False
        stride = 1
    else:
        raise ValueError('Output and input channel does not match in residual blocks!!!')

    # About : https://timodenk.com/blog/tensorflow-batch-normalization/
    # The first conv layer of the first residual block does not need to be normalized and relu-ed.
    with tf.variable_scope('conv1_in_block'):
        if first_block:
            conv1 = tf.layers.conv2d(input_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME')

        else:
            bn_layer= tf.layers.batch_normalization(input_layer, training= is_training)
            #bn_layer = batch_normalization_layer(input_layer, input_layer.get_shape().as_list()[-1])
            conv1 = tf.layers.conv2d(bn_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME', strides=(stride, stride))


    with tf.variable_scope('conv2_in_block'):
        bn_layer= tf.layers.batch_normalization(conv1, training= is_training)
        #bn_layer = batch_normalization_layer(conv1, conv1.get_shape().as_list()[-1])
        conv2 = tf.layers.conv2d(bn_layer, output_channel, [3, 3], activation=tf.nn.relu, padding='SAME')


    # When the channels of input layer and conv2 does not match, we add zero pads to increase the
    #  depth of input layers
    if increase_dim is True:

        pooled_input = tf.nn.avg_pool(input_layer, ksize=[1, 2, 2, 1],
                                      strides=[1, 2, 2, 1], padding='VALID')
        padded_input = tf.pad(pooled_input, [[0, 0], [0, 0], [0, 0], [input_channel // 2,
                                                                      input_channel // 2]])

    else:
        padded_input = input_layer

    output = conv2 + padded_input
    return output

def whitening_image(image_np):
    '''
    Performs per_image_whitening
    :param image_np: a 4D numpy array representing a batch of images
    :return: the image numpy array after whitened
    '''
    tmp = np.empty(image_np.shape)
    for i in range(len(image_np)):
        mean = np.mean(image_np[i, ...])
        # Use adjusted standard deviation here, in case the std == 0.
        std = np.max([np.std(image_np[i, ...]), 1.0/np.sqrt(IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH)])
        tmp[i,...] = (image_np[i, ...] - mean) / std
    return tmp


def random_crop_and_flip(batch_data, padding_size):
    '''
    Helper to random crop and random flip a batch of images
    :param padding_size: int. how many layers of 0 padding was added to each side
    :param batch_data: a 4D batch array
    :return: randomly cropped and flipped image
    '''
    cropped_batch = np.zeros(len(batch_data) * IMG_HEIGHT * IMG_WIDTH * IMG_DEPTH).reshape(
        len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_DEPTH)

    pad_width = ((0, 0), (padding_size, padding_size), (padding_size, padding_size), (0, 0))
    batch_data = np.pad(batch_data, pad_width=pad_width, mode='constant', constant_values=0)

    for i in range(len(batch_data)):
        x_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        y_offset = np.random.randint(low=0, high=2 * padding_size, size=1)[0]
        cropped_batch[i, ...] = batch_data[i, ...][x_offset:x_offset+IMG_HEIGHT,
                      y_offset:y_offset+IMG_WIDTH, :]

        flip_prop = np.random.randint(low=0, high=2)
        if flip_prop == 0:
            cropped_batch[i, ...] = cv2.flip(cropped_batch[i, ...], 1) #axis=1

    return cropped_batch

def generate_augment_train_batch(train_data, train_labels, train_batch_size):
    '''
    This function helps generate a batch of train data, and random crop, horizontally flip
    and whiten them at the same time
    :param train_data: 4D numpy array
    :param train_labels: 1D numpy array
    :param train_batch_size: int
    :return: augmented train batch data and labels. 4D numpy array and 1D numpy array
    '''
    EPOCH_SIZE = 10000 * 5
    padding_size= 2

    offset = np.random.choice(EPOCH_SIZE - train_batch_size, 1)[0]
    batch_data = train_data[offset:offset+train_batch_size, ...]
    batch_data = random_crop_and_flip(batch_data, padding_size=padding_size)

    batch_data = whitening_image(batch_data)
    batch_label = train_labels[offset:offset+train_batch_size, ...]

    return batch_data, batch_label

def generate_vali_batch(vali_data, vali_label, vali_batch_size):
    '''
    If you want to use a random batch of validation data to validate instead of using the
    whole validation data, this function helps you generate that batch
    :param vali_data: 4D numpy array
    :param vali_label: 1D numpy array
    :param vali_batch_size: int
    :return: 4D numpy array and 1D numpy array
    '''
    offset = np.random.choice(10000 - vali_batch_size, 1)[0]
    vali_data_batch = vali_data[offset:offset+vali_batch_size, ...]
    vali_label_batch = vali_label[offset:offset+vali_batch_size]
    return vali_data_batch, vali_label_batch

