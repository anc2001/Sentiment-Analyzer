from main import load_weights, get_rating
from lstm_model import LSTM_Model
import tensorflow as tf
import hyperparameters as hp
from preprocess import get_data
import numpy as np

def main():
    try:
        # (train_data, train_labels, test_data, test_labels, encoder) = get_data("../training.1600000.processed.noemoticon.csv", "../testdata.manual.2009.06.14.csv")
        # model = LSTM_Model(encoder)
        # temp = np.array(["huh"])
        # _ = model(temp)
        # model = load_weights(model, "checkpoint")
        model = tf.keras.models.load_model("lstm_model")
        while True:
            try:
                _in = input(">> ")
                if _in == "exit":
                    exit()
                try:
                    # rating = get_rating(model,_in)
                    rating = model.predict(np.array([(_in)]))
                    # If the prediction is >= 0.0, it is positive else it is negative.
                    text = '\n Rating: '
                    print('{}{}'.format(text, rating.numpy()[0]))
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