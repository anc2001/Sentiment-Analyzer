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
def get_data(train_path, n):
    # Number of rows from the training data to take
    size = hp.TRAINING_SIZE
    skip = sorted(random.sample(range(n),n-size))
    train_df, test_df, validation_df = split_data(train_path, skip)

    # Text vectorization - abstracts away having to create a vocabulary and turning text into indices
    encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder.adapt(np.asarray(train_df[hp.INPUT_IDX]).astype('<U3'))

    train_data = np.asarray(train_df[hp.INPUT_IDX]).astype('<U3')
    train_label = np.asarray(train_df[hp.LABEL_IDX]).astype(int) / hp.DENOM
    test_data = np.asarray(test_df[hp.INPUT_IDX]).astype('<U3')
    test_label = np.asarray(test_df[hp.LABEL_IDX]).astype(int) / hp.DENOM
    validation = (np.asarray(validation_df[hp.INPUT_IDX]).astype('<U3'), np.asarray(validation_df[hp.LABEL_IDX]).astype(int) / hp.DENOM)

    return (train_data, train_label, test_data, test_label, validation, encoder)

# Splits large training dataset into testing and validation sets
def split_data(data, skip):
    train = pd.read_csv(data, header=None, usecols=[hp.LABEL_IDX, hp.INPUT_IDX], encoding='latin-1', skiprows=skip)
    train = train.drop(0)
    test = train.sample(frac=0.05)
    train = train.drop(test.index)
    validation = train.sample(frac=0.05)
    train = train.drop(validation.index)
    return train, test, validation

def get_imdb_data():
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.shuffle(hp.BUFFER_SIZE).batch(hp.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(hp.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder = tf.keras.layers.TextVectorization(max_tokens=hp.VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    return (train_dataset, test_dataset, encoder)

def get_sarcasm_data():
    hp.LABEL_IDX = 0
    hp.INPUT_IDX = 1
    hp.DENOM = 1
    return get_data("../train-balanced-sarcasm.csv", 1010825)

def get_twitter_data():
    hp.LABEL_IDX = 0
    hp.INPUT_IDX = 5
    hp.DENOM = 4
    return get_data("../training.1600000.processed.noemoticon.csv", 1600000)

if __name__ == "__main__":
    get_sarcasm_data()
    get_twitter_data()