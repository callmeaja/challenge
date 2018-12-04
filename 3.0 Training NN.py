import tensorflow as tf
import numpy as np
import os
import time
import datetime
from importlib.machinery import SourceFileLoader
os.chdir('Z:\Desktop\Competitions\AI Challenge')

textCNN = SourceFileLoader('textCNN', 'Codes\2.0 Setting up the Pipeline.py').load_module()

# Hyperparameters
filter_size = 3
num_filters = 128
dropout_keep_prob = 0.5

# Training parameters
batch_size = 64
num_epochs = 200
evaluate_every = 100
checkpoint_every = 100
num_checkpoints = 5

def train(x_train, y_train, x_test, y_test):
    train_p = x_train[:, 1]
    val_p = x_test[:, 1]
    train_q = x_train[:, 0]
    test_q = x_test[:, 0]

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():
            cnn = textCNN(
                sequence_length=50,
                filter_size=filter_size,
                num_filters=num_filters
            )

            # Defining training procedure
            global_step = tf.Variable(0, name='gloabal_step', trainable=False)
            optimiser = tf.train.AdamOptiizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)