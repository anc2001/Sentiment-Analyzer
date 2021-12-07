import nltk
import pandas as pd
import random
import tensorflow as tf
import numpy as np
import time
import csv
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

'''
Determines which tokens will remain in the sentence
return: True if we wish to keep the token, False otherwise
'''
def gate(token):
    return token.isalnum() and token not in stop_words

'''
Function to filter out non-utf-8 characters, stem the words, and make them all lowercase
return: a string
'''
def strip_and_stem(token):
    temp = bytes(token, "utf-8").decode("utf-8", "ignore")
    return stemmer.stem(temp)

'''
Creates a row to eventually write into the .csv file
return: a list containing the label and the preprocessed sentence
'''
def create_row(label, sentence):
    tokens = word_tokenize(sentence)
    line = [strip_and_stem(token) for token in tokens if gate(token)]
    return [str(label), " ".join(line)]

def create_sentence(sentence):
    tokens = word_tokenize(sentence)
    line = [strip_and_stem(token) for token in tokens if gate(token)]
    return " ".join(line)

'''
Function to preprocess the Sentiment 140 dataset
'''
def preprocess(train_path, test_path, train_csv, test_csv):
    start = time.time()
    print("Starting preprocessing...")

    train_df = pd.read_csv(train_path, header=None, usecols=[0, 5], encoding='latin-1')
    test_df = pd.read_csv(test_path, header=None, usecols=[0, 5], encoding='latin-1')

    with open(train_csv, "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for _, row in train_df.iterrows():
            sentence = create_sentence(row[5])
            if len(sentence) != 0:
                writer.writerow([row[0], sentence])

    with open(test_csv, "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for _, row in test_df.iterrows():
            sentence = create_sentence(row[5])
            if len(sentence) != 0:
                writer.writerow([row[0], sentence])

    end = time.time()
    print(end - start)
    print("Done!")

# Only works with Sentiment 140 dataset
if __name__ == "__main__":
    preprocess("training.1600000.processed.noemoticon.csv", "testdata.manual.2009.06.14.csv", "train_preprocessed.csv", "test_preprocessed.csv")
