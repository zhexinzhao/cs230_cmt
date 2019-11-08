"""Load data from '/data' folder
"""

import argparse
import random
import numpy as np
import os
from const_params import *


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data', help="Directory with the Input and Output dataset")


def load_data(validation_size_p, dirname, data_size_x, data_size_y):
    """
    Params:
    validation_size_p: percentage of data to be used for validation
    dirname (str): directory containing raw data
    """
    data_in_dir = os.path.join(dirname, 'InputRaw')
    data_out_dir = os.path.join(dirname, 'OutputRaw')

    # Get the filenames in each directory (train and test)
    filenames = os.listdir(data_in_dir)
    filenames_out = os.listdir(data_out_dir)
    # Shuffle the files
    random.seed(230)
    filename_zip = list(zip(filenames, filenames_out))
    random.shuffle(filename_zip)
    filenames, filenames_out = zip(*filename_zip)
    # filenames_out = [f[1] for f in filename_zip]
    print(filenames[0])
    print(filenames_out[0])

    filenames = [os.path.join(data_in_dir, f) for f in filenames if f.endswith('.txt')]
    filenames_out = [os.path.join(data_out_dir, f) for f in filenames_out if f.endswith('.txt')]

    # Load data from txt files into nparray
    Ndata = len(filenames)
    data_in = np.zeros((Ndata, data_size_x, 6))
    data_out = np.zeros((Ndata, data_size_y, 1))
    for ii in range(Ndata):
        data_in_ii = np.loadtxt(fname=filenames[ii], delimiter='\t')
        data_out_ii = np.loadtxt(fname=filenames_out[ii], delimiter='\t')
        # data_in = np.append(data_in, data_in_ii, axis=0)
        # data_out = np.append(data_out, data_out_ii, axis=0)
        data_in[ii, :, :] = data_in_ii
        data_out[ii, :, :] = data_out_ii[:data_size_y].reshape((data_size_y, 1))

        if ii == -1:
            print(data_in[ii, 0:5, :])
            print(data_out[ii, 0:5, :])

    print(data_in.shape)
    print(data_out.shape)
    # data_out is further split into 3 parts
    data_out_c = data_out[:, 0:NC, :]       # about the background channel
    data_out_w = data_out[:, NC::NPERMODE, :]
    data_out_d = np.zeros((Ndata, NPERMODE-1, MMODE))
    for jj in range(MMODE):
        data_out_d[:, :, jj] = data_out[:, NC+jj*NPERMODE+1:NC+(jj+1)*NPERMODE, 0]

    # split the data into train/dev sets
    train_dev_sep = int(Ndata * (1 - validation_size_p/100.0))
    train_x = data_in[:train_dev_sep, :, :]
    train_y_c = data_out_c[:train_dev_sep, :, :]
    train_y_w = data_out_w[:train_dev_sep, :, :]
    train_y_d = data_out_d[:train_dev_sep, :, :]
    dev_x = data_in[train_dev_sep:, :, :]
    dev_y_c = data_out_c[train_dev_sep:, :, :]
    dev_y_w = data_out_w[train_dev_sep:, :, :]
    dev_y_d = data_out_d[train_dev_sep:, :, :]

    # print(data_out_w[0:5, :, :])

    return (train_x, train_y_c, train_y_w, train_y_d, dev_x, dev_y_c, dev_y_w, dev_y_d)


if __name__ == '__main__':
    args = parser.parse_args()
    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # Define the data directories
    # dirname = 'C:\\Users\\joann\\PycharmProjects\\TensorFlow_Python37\\cs230_cmt\\data'
    dirname = os.path.realpath(args.data_dir)
    data_size_y = NC + MMODE * NPERMODE
    data_set = load_data(validation_size_p=VAL_SIZE_P, dirname=dirname, data_size_x=DATA_SIZE, data_size_y=data_size_y)

    print("Done Loading data")
