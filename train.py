"""Train the model"""

import argparse
import logging
import os

from data_loader import *

import tensorflow as tf

from model.input_fn import input_fn
from model.utils import Params
from model.utils import set_logger
from model.model_fn import model_fn
from model.training import train_and_evaluate


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments\\base_model_4',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data',
                    help="Directory containing the dataset")
parser.add_argument('--restore_from', default=None,
                    help="Optional, directory or file containing weights to reload before training")

if __name__ == '__main__':
    # Set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(230)

    # Load the parameters from json file
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path)

    # Check that we are not overwriting some previous experiment
    # Comment these lines if you are developing your model and don't care about overwritting
    model_dir_has_best_weights = os.path.isdir(
        os.path.join(args.model_dir, "best_weights"))
    overwritting = model_dir_has_best_weights and args.restore_from is None
    assert not overwritting, "Weights found in model_dir, aborting to avoid overwrite"

    # Set the logger
    set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Creating the datasets...")
    dirname = os.path.realpath(args.data_dir)
    size_y = NC + NPERMODE * MMODE
    data_set = load_data(validation_size_p=VAL_SIZE_P, dirname=dirname, data_size_x=DATA_SIZE, data_size_y=size_y)
    (train_x, train_y_c, train_y_w, train_y_d, dev_x, dev_y_c, dev_y_w, dev_y_d) = data_set
    # print(train_y[0:3, 0:3, -1])
    (m_train, n_x, n_channel_x) = train_x.shape
    m_dev = dev_x.shape[0]

    print(train_x.shape)
    print(m_dev)

    params.train_size = m_train
    params.eval_size = m_dev

    # Create the two iterators over the two datasets
    train_inputs = input_fn(True, train_x, train_y_c, train_y_w, train_y_d, params)
    eval_inputs = input_fn(False, dev_x, dev_y_c, dev_y_w, dev_y_d, params)

    # Define the model
    logging.info("Creating the model...")
    train_model_spec = model_fn('train', train_inputs, params)
    eval_model_spec = model_fn('eval', eval_inputs, params, reuse=True)

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(train_model_spec, eval_model_spec,
                       args.model_dir, params, args.restore_from)