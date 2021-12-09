import argparse
import tensorflow as tf
import numpy as np
from preprocess import get_twitter_data, get_imdb_data, get_sarcasm_data
from lstm_model import LSTM_Model
import hyperparameters as hp
from matplotlib import pyplot as plt
import os.path
import pickle

def parseArguments():
    parser = argparse.ArgumentParser()
    #Options are imdb, sentiment140
    parser.add_argument("--type", type=str, choices=["imdb", "twitter", "sarcasm", "twitter_sarcasm"], default="imdb") 
    parser.add_argument("--model_type", type=str, default="lstm")
    args = parser.parse_args()
    return args 

def visualize_analysis(model,text):
    rating = get_rating(model, text)
    text += '\n \n Rating: ' + rating
    plt.text(.5,.5,text, bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},ha='center', va='center') 
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    plt.grid(True)
    plt.show()
    pass

def get_rating(model, text):
    data = np.array([text])
    logits = model.call(data)
    rating = tf.argmax(logits, 1)
    return rating

def save_weights(model, name):
    output_dir = os.path.join("model_ckpts", name)
    output_path = os.path.join(output_dir, name)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)

def load_weights(model, name):
    weights_path = os.path.join("model_ckpts", name, name)
    sample_text = ('The movie was cool. The animation and the graphics '
               'were out of this world. I would recommend this movie.')
    _ = model(np.array([sample_text]))
    model.load_weights(weights_path)
    return model

def save_encoder(encoder, name):
    pickle.dump({'config': encoder.get_config(),
             'weights': encoder.get_weights()}
            , open(name, "wb"))
    return None

def load_encoder(name):
    from_disk = pickle.load(open(name, "rb"))
    new_encoder = tf.keras.layers.TextVectorization.from_config(from_disk['config'])
    new_encoder.adapt(tf.data.Dataset.from_tensor_slices(["xyz"]))
    new_encoder.set_weights(from_disk['weights'])
    return new_encoder

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def visualize_training(history, name):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.savefig(name)

def train_imdb():
    (train_dataset, test_dataset, encoder) = get_imdb_data()
    validation_dataset = test_dataset
    model = LSTM_Model(encoder)
    model.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy'])
    history = model.model.fit(train_dataset, epochs=hp.EPOCHS,
                    validation_data=validation_dataset,
                    validation_steps=30)
    test_loss, test_acc = model.model.evaluate(test_dataset)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    visualize_training(history, "imdb_dataset.png")
    save_encoder(encoder, "imdb_encoder.pkl")
    save_weights(model, "imdb")

def train_twitter():
    (train_data, train_label, test_data, test_label, validation, encoder) = get_twitter_data()
    model = LSTM_Model(encoder)
    model.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy'])  
    history = model.model.fit(train_data, train_label, epochs=hp.EPOCHS, batch_size=hp.BATCH_SIZE,
            validation_data=validation,
            validation_steps=30)    
    test_loss, test_acc = model.model.evaluate(test_data, test_label)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    visualize_training(history, "twitter_dataset.png")
    save_encoder(encoder, "twitter_encoder.pkl")
    save_weights(model, "twitter")

def train_sarcasm():
    (train_data, train_label, test_data, test_label, validation, encoder) = get_twitter_data()
    model = LSTM_Model(encoder)
    model.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(1e-4),
            metrics=['accuracy'])  
    history = model.model.fit(train_data, train_label, epochs=hp.EPOCHS, batch_size=hp.BATCH_SIZE,
            validation_data=validation,
            validation_steps=30)    
    test_loss, test_acc = model.model.evaluate(test_data, test_label)
    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    visualize_training(history, "sarcasm_dataset.png")
    save_encoder(encoder, "sarcasm_encoder.pkl")
    save_weights(model, "sarcasm")

def main(args):
    if args.type == "imdb":
        train_imdb()
    elif args.type == "twitter":
        train_twitter()
    elif args.type == "sarcasm":
        train_sarcasm()

if __name__ == '__main__':
    args = parseArguments()
    main(args)
