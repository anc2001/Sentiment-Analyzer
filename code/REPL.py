from main import load_weights, get_rating
from lstm_model import LSTM_Model
import tensorflow as tf
import hyperparameters as hp
from preprocess import get_data
import numpy as np
import stemming

def main():
    try:
        (train_data, train_labels, test_data, test_labels, encoder) = get_data("train_preprocessed.csv", "test_preprocessed.csv")
        model = LSTM_Model(encoder)
        model = load_weights(model, "checkpoint")
        while True:
            try:
                _in = input(">> ")
                if _in == "exit":
                    exit()
                try:
                    sentence = stemming.create_sentence(_in)
                    if (len(sentence)) != 0:
                        rating = get_rating(model,sentence)
                        text = '\n Rating: '
                        print('{}{}'.format(text, rating.numpy()[0]))
                    else:
                        print("Not enough info!")
                except:
                    out = exec(_in)
                    if out != None:
                        print(out)
            except Exception as e:
                print(f"Error: {e}")
    except KeyboardInterrupt as e:
        print("\nExiting...")

if __name__ == '__main__':
	main()       