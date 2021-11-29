import tensorflow as tf
import numpy as np
from preprocess import get_data
from lstm_model import LSTM_Model

def train(model, train_inputs, train_labels):
    pass

def test(model, test_inputs, test_labels):
    pass

def main():
    print("Running preprocessing...")
    (training, testing, vocab) = get_data("../data/training.1600000.processed.noemoticon.csv", "../data/testdata.manual.2009.06.14.csv")
    print("Preprocessing complete.")
    model = LSTM_Model()

if __name__ == '__main__':
	main()