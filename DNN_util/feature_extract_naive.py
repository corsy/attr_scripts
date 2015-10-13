import caffe
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2 as cv

# Configuration
model_path = '/home/luwei/Desktop/test/refine.caffemodel'
model_define = '/home/luwei/Desktop/test/deploy.prototxt'
#image_mean = '/opt/caffe/examples/CNN_EXPERIMENTS/image_mean.npy'     # need to be npy file
output_dir_path = '/home/luwei/Desktop/test/'

fliter_extract_list = ['conv1','conv1_p']


def show_image(im):
    if im.ndim == 3:
        im = im[:, :, ::-1]

    return im


def vis_square(data, padsize=1, padval=0):
    data -= data.min()
    data /= data.max()

    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant', constant_values=(padval, padval))
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    im = show_image(data)
    return im


def load_net():
    caffe.set_mode_gpu()
    net = caffe.Net(model_define, model_path,
                           caffe.TEST)

    # input pre-processing: 'data' is the name of the input blob == net.inputs[0]
    # transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    # transformer.set_transpose('data', (2, 0, 1))
    # transformer.set_mean('data', np.load(image_mean).mean(1).mean(1)) # mean pixel
    # transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
    # transformer.set_channel_swap('data', (2, 1, 0))  # the reference model has channels in BGR order instead of RGB
    # net.blobs['data'].reshape(1, 2, 227, 227)

    return net

if __name__ == '__main__':
    # Setting up the parameter
    plt.rcParams['figure.figsize'] = (10, 10)
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'

    net = load_net()

    blobs = [(k, v.data.shape) for k, v in net.blobs.items()]


    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)


    for filter_name in fliter_extract_list:
        filters = net.params[filter_name][0].data
        im = vis_square(filters.transpose(0, 2, 3, 1))
        im = im * 255
        cv.imwrite(output_dir_path + filter_name + '_kernel.jpg', im)