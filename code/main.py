import tensorflow as tf
import numpy as np
from preprocess import get_data
from lstm_model import LSTM_Model
import hyperparameters as hp
from matplotlib import pyplot as plt
import os.path

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
            model.loss_visualization.append(loss)
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

def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    NOTE: DO NOT EDIT

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  

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

def load_weights(model, name):
    weights_path = os.path.join("model_ckpts", name, name)
    model.load_weights(weights_path)
    return model

def save_weights(model, name):
    output_dir = os.path.join("model_ckpts", name)
    output_path = os.path.join(output_dir, name)
    os.makedirs("model_ckpts", exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    model.save_weights(output_path)

def main():
    (train_data, train_labels, test_data, test_labels, encoder) = get_data("../training.1600000.processed.noemoticon.csv", "../testdata.manual.2009.06.14.csv")
    model = LSTM_Model(encoder)
    for i in range(hp.EPOCHS):
        train(model, train_data, train_labels)
        accuracy = test(model, test_data, test_labels)
        print("Epoch accuracy:", accuracy)
    visualize_loss(model.loss_visualization)

    #This should be abstracted later 
    save_weights(model, "checkpoint")

if __name__ == '__main__':
	main()