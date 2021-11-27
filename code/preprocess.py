import csv
import numpy as np
from numpy.core.defchararray import split
import pandas as pd
import random

'''
Function to preprocess data
train_file - path to the training data
test_file - path to testing data
return:
    train_sentences - a list of lists of strings, represents all of the input sentences
    train_labels - a numpy array of integers, represents the labels for each input
    train_sentences - 
    train_labels - 
    vocab - a dictionary mapping words to unique IDs
'''
def get_data(train_path, test_path):
    n = 1600000
    # Number of rows from the training data to take
    size = 100000
    skip = sorted(random.sample(range(n),n-size))

    # Reading the csv data, taking only columns 0 and 5
    train_df = pd.read_csv(train_path, header=None, usecols=[0, 5], encoding='latin-1', skiprows=skip)
    test_df = pd.read_csv(test_path, header=None, usecols=[0, 5], encoding='latin-1')

    # Preprocess all sentences
    train_sentences = [preprocess(sentence) for sentence in train_df[5]]
    test_sentences = [preprocess(sentence) for sentence in test_df[5]]
    vocab = create_vocab(train_sentences, test_sentences)

    train_sentences = [convert_to_id(sentence, vocab) for sentence in train_sentences]
    test_sentences = [convert_to_id(sentence, vocab) for sentence in test_sentences]

    train_labels = np.array(train_df[0])
    test_labels = np.array(test_df[0])
    return train_sentences, train_labels, test_sentences, test_labels, vocab

'''
Preproceessing function, converts a sentence to a standard format
'''
def preprocess(sentence):
    temp = sentence.strip()
    temp = temp.lower()
    temp = temp.split()
    temp = [word for word in filter(wordFilter, temp)]
    return temp

'''
Given all of the sentences in the training and testing data, creates a dictionary
mapping each word to a unique id
'''
def create_vocab(train, test):
    words = set()
    for sentence in train:
        words.update(sentence)
    for sentence in test:
        words.update(sentence)
    vocab = {word:count for count, word in enumerate(words)}
    return vocab

'''
Maps each word in the given sentence to its id in vocab
'''
def convert_to_id(sentence, vocab):
    return [vocab[word] for word in sentence]

'''
Filter to determine which words are important enough to keep
'''
def wordFilter(word):
    return len(word) >= 2

if __name__ == "__main__":
    get_data("training.1600000.processed.noemoticon.csv", "testdata.manual.2009.06.14.csv")

        