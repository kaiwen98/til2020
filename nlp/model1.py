import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import pickle
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tensorflow.keras.models import Model, model_from_json, load_model
from tensorflow.keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from tensorflow.keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import optimizers
from tensorflow.keras import initializers, regularizers, constraints, callbacks
from sklearn.utils import shuffle
import tensorflow as tf

class TEXT_MODEL(Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=50,
                 dnn_units=512,
                 model_output_classes=2,
                 dropout_rate=0.1,
                 training = True,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = GlobalMaxPooling1D()
        self.trainable = True
        self.dense_1 = Dense(units=dnn_units, activation="relu")
        self.dropout = Dropout(rate=dropout_rate)

        self.last_dense = Dense(units=5,input_shape=(5,),
                                        activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        print("Attention: ", type(self))
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated = self.dropout(concatenated, training = training)
        model_output = self.last_dense(concatenated)
        
        return model_output