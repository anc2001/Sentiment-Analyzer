from main import load_weights, get_rating
from lstm_model import LSTM_Model
import tensorflow as tf
import hyperparameters as hp
from preprocess import get_data
from main import load_encoder
import numpy as np
import argparse

def parseArguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sarcasm", action="store_true")
    parser.add_argument("--type", type=str, choices=["imdb", "twitter"], default="twitter") 
    args = parser.parse_args()
    return args 

def main(args):
    try:
        sentiment_model = None
        sarcasm_model = None
        if args.type == "imdb":
            encoder = load_encoder("imdb_encoder.pkl")
            sentiment_model = LSTM_Model(encoder)
            sentiment_model = load_weights(sentiment_model, "imdb")
        elif args.type == "twitter":
            encoder = load_encoder("twitter_encoder.pkl")
            sarcasm_model = LSTM_Model(encoder)
            sarcasm_model = load_weights(sarcasm_model, "twitter")
        
        if args.use_sarcasm:
            encoder = load_encoder("sarcasm_encoder.pkl")
            sarcasm_model = LSTM_Model(encoder)
            sarcasm_model = load_weights(sarcasm_model, "sarcasm")
        while True:
            try:
                _in = input(">> ")
                if _in == "exit":
                    exit()
                try:
                    # If the prediction is >= 0.0, it is positive else it is negative.
                    rating = sentiment_model(np.array([_in]))
                    if args.use_sarcasm:
                        sarcasm_rating = sarcasm_model(np.array([_in]))
                        vibe_check = sarcasm_rating.numpy()[0]
                        print("Sarcasm rating (for debugging):")
                        print(vibe_check)
                        if vibe_check >= 0:
                            rating = -rating
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
    args = parseArguments()
    main(args)