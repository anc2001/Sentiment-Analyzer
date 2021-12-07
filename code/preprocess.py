import csv
import numpy as np
from numpy.core.defchararray import split
import pandas as pd
import random
import hyperparameters as hp
import tensorflow as tf
import tensorflow_datasets as tfds

'''
Function to preprocess data
train_file - path to the training data
test_file - path to testing data
return:
    train_data - (TRAINING_SIZE, ) size array where each index is a string 
    train_labels - (TRAINING_SIZE, NUM_CLASSES) of one hot vectors representing the sentiment 
    test_data - (TESTING_SIZE, ) size array where each index is a string 
    test_labels -  (TRAINING_SIZE, NUM_CLASSES) of one hot vectors representing the sentiment 
    encoder - a tf.keras.layers.TextVectorization layer that contains the vocabulary
'''
def get_data(train_path, test_path, model_encoder=None):
    n = 1600000
    # Number of rows from the training data to take
    size = hp.TRAINING_SIZE
    skip = sorted(random.sample(range(n),n-size))

    # Reading the csv data, taking only columns 0 and 5
    # train_df = pd.read_csv(train_path, header=None, usecols=[hp.LABEL_IDX, hp.INPUT_IDX], encoding='latin-1', skiprows=skip)
    # test_df = pd.read_csv(test_path, header=None, usecols=[hp.LABEL_IDX, hp.INPUT_IDX], encoding='latin-1')

    train_df, test_df, validation_df = split_data(train_path, skip)

    # Text vectorization - abstracts away having to create a vocabulary and turning text into indices
    encoder = None
    if model_encoder:
        encoder = model_encoder
    else:
        encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder.adapt(train_df[5])

    train_data = np.array(train_df[hp.INPUT_IDX])
    train_labels = np.array(train_df[hp.LABEL_IDX]) / 2
    train_labels = tf.one_hot(train_labels, hp.NUM_CLASSES).numpy()
    test_data = np.array(test_df[hp.INPUT_IDX])
    test_labels = np.array(test_df[hp.LABEL_IDX]) / 2
    test_labels = tf.one_hot(test_labels, hp.NUM_CLASSES).numpy()

    return (train_data, train_labels, test_data, test_labels, encoder)

# Splits large training dataset into testing and validation sets
def split_data(data, skip):
    train = pd.read_csv(data, header=None, usecols=[hp.LABEL_IDX, hp.INPUT_IDX], encoding='latin-1', skiprows=skip)
    test = train.sample(frac=0.05)
    train = train.drop(test.index)
    validation = train.sample(frac=0.05)
    train = train.drop(validation.index)
    return train, test, validation

def load_tfds_imdb(model_encoder=None):
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.shuffle(hp.BUFFER_SIZE).batch(hp.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    encoder = None
    if model_encoder:
        encoder = model_encoder
    else:
        encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    return (train_dataset, test_dataset, encoder)

if __name__ == "__main__":
    get_data("../training.1600000.processed.noemoticon.csv", "../testdata.manual.2009.06.14.csv")