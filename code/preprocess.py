import csv
import numpy as np
from numpy.core.defchararray import split
import pandas as pd
import random
import hyperparameters as hp
import tensorflow as tf

'''
Function to preprocess data
train_file - path to the training data
test_file - path to testing data
return:
    training - list of (sentence->string, sentiment->int) tuples where setence is a list of integers
    validation - ditto
    testing - ditto
    vocab - list of strings of length VOCAB_SIZE, starts with padding and UNK
    vocab - a dictionary mapping words to unique IDs
'''
def get_data(train_path, test_path):
    n = 1600000
    # Number of rows from the training data to take
    size = hp.TRAINING_SIZE
    skip = sorted(random.sample(range(n),n-size))

    # Reading the csv data, taking only columns 0 and 5
    train_df = pd.read_csv(train_path, header=None, usecols=[0, 5], encoding='latin-1', skiprows=skip)
    test_df = pd.read_csv(test_path, header=None, usecols=[0, 5], encoding='latin-1')

    # Text vectorization - abstracts away having to create a vocabulary and turning text into indices
    encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder.adapt(train_df[5])

    train_data = np.array(train_df[5])
    train_labels = tf.one_hot(train_df[0], hp.NUM_CLASSES).numpy()
    test_data = np.array(test_df[5])
    test_labels = tf.one_hot(test_df[0], hp.NUM_CLASSES).numpy()

    return (train_data, train_labels, test_data, test_labels, encoder)

if __name__ == "__main__":
    get_data("../training.1600000.processed.noemoticon.csv", "../testdata.manual.2009.06.14.csv")