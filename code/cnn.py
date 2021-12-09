import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import hyperparameters as hp
from tensorflow.python.ops.embedding_ops import embedding_lookup_sparse

class LSTM_Model(tf.keras.Model):
    def __init__(self, encoder):
        super(LSTM_Model, self).__init__()
        self.model = tf.Sequential([
            layers.Embedding(hp.VOCAB_SIZE, 40),
            layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)),
            layers.MaxPooling1D(5),
            layers.Conv1D(20, 6, activation='relu',kernel_regularizer=regularizers.l1_l2(l1=2e-3, l2=2e-3),bias_regularizer=regularizers.l2(2e-3)),
            layers.GlobalMaxPooling1D(),
            layers.Dense(hp.NUM_CLASSES,activation='softmax')
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