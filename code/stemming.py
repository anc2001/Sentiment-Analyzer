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
Function to preprocess the dataset
'''
def preprocess(source, target, label, text, denom):
    start = time.time()
    print("Starting preprocessing...")

    df = pd.read_csv(source, header=None, usecols=[label, text], encoding='latin-1')
    df.dropna(subset=[text])

    with open(target, "w", newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        for _, row in df.iterrows():
            sentence = create_sentence(row[text])
            if len(sentence) != 0:
                writer.writerow([row[label] // denom, sentence])

    end = time.time()
    print(end - start)
    print("Done!")

if __name__ == "__main__":
    pass
    # preprocess(<data>, <name of new csv>, label_column, text_column, denominator)
