import tensorflow as tf
import numpy as np
from preprocess import get_data
from lstm_model import LSTM_Model
import hyperparameters as hp

'''
Trains the model, shuffles and batches the data feeding to network and backpropagates
    train_data - (TRAINING_SIZE, ) size array where each index is a string 
    train_labels - (TRAINING_SIZE, NUM_CLASSES) of one hot vectors representing the sentiment 
'''
def train(model, train_inputs, train_labels):
    optimizer = tf.keras.optimizers.Adam(hp.LEARNING_RATE)

    # Shuffle arrays
    indices = range(train_labels.shape[0])
    indices = tf.random.shuffle(indices)
    inputs = tf.gather(train_inputs, indices)
    labels = tf.gather(train_labels, indices)

    start = 0
    end = hp.BATCH_SIZE
    while end <= train_labels.shape[0]:
        batch_inputs = inputs[start:end]
        batch_labels = labels[start:end]
        with tf.GradientTape() as tape:
            probs = model(batch_inputs)
            loss = model.loss_function(probs, batch_labels)
            accuracy = model.accuracy_function(probs, batch_labels)
            if start % 10000 == 0:
                print("Accuracy: ", accuracy.numpy())
                print("Loss: ", loss.numpy())
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        start = end
        end += hp.BATCH_SIZE
    return None

'''
Tests the model
    test_data - (TESTING_SIZE, ) size array where each index is a string 
    test_labels -  (TRAINING_SIZE, NUM_CLASSES) of one hot vectors representing the sentiment 
'''
def test(model, test_inputs, test_labels):
    #For batching, but as is 

    # start = 0
    # end = hp.BATCH_SIZE
    # accum = []
    # print(test_labels.shape[0])
    # while end <= test_labels.shape[0]:
    #     batch_inputs = test_inputs[start:end]
    #     batch_labels = test_labels[start:end]
    #     probs = model(batch_inputs)
    #     accuracy = model.accuracy_function(probs, batch_labels).numpy()
    #     accum.append(accuracy)
    #     start = end
    #     end += hp.BATCH_SIZE
    
    probs = model(test_inputs)
    accuracy = model.accuracy_function(probs, test_labels).numpy()
    return accuracy

def main():
    (train_data, train_labels, test_data, test_labels, encoder) = get_data("../training.1600000.processed.noemoticon.csv", "../testdata.manual.2009.06.14.csv")
    model = LSTM_Model(encoder)
    for i in range(hp.EPOCHS):
        train(model, train_data, train_labels)
        accuracy = test(model, test_data, test_labels)
        print("Epoch accuracy:", accuracy)

if __name__ == '__main__':
	main()