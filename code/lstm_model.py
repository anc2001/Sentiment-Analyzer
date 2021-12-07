import tensorflow as tf
import numpy as np
import hyperparameters as hp
from tensorflow.python.ops.embedding_ops import embedding_lookup_sparse

class LSTM_Model(tf.keras.Model):
    def __init__(self, encoder):
        super(LSTM_Model, self).__init__()
        self.model = tf.keras.Sequential([
            encoder,
            tf.keras.layers.Embedding(hp.VOCAB_SIZE, hp.EMBEDDING_SIZE, mask_zero=False),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(hp.NUM_CLASSES),
        ])
        self.loss_visualization = []

    '''
    Does one forward pass
        sentences - (BATCH_SIZE, ) of strings
    '''
    def call(self, sentences):
        return self.model(sentences)

    '''
    Calculates the accuracy of the given logits and labels
        logits - (BATCH_SIZE, NUM_CLASSES) of logits 
        labels - (BATCH_SIZE, NUM_CLASSES) of one hot vectors for ground truth
    '''
    def accuracy_function(self, logits, labels):
        correct_predictions = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        return tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    '''
    Calculates the model cross-entropy loss for one forward pass 
        logits - (BATCH_SIZE, NUM_CLASSES) of logits 
        labels - (BATCH_SIZE, NUM_CLASSES) of one hot vectors for ground truth
    '''
    def loss_function(self, logits, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))