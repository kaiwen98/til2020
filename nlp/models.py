from __future__ import absolute_import, division

import tensorflow as tf
from keras import optimizers
from keras import initializers
from keras.layers import InputSpec, Layer, LSTM, GRU

import os
import gc
import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
import pickle
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from keras.models import Model, model_from_json, load_model
from keras.layers import Input, Dense, Embedding, MaxPooling1D, Conv1D, SpatialDropout1D
from keras.layers import add, Dropout, PReLU, BatchNormalization, GlobalMaxPooling1D
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras.layers import BatchNormalization, Concatenate
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from sklearn.utils import shuffle
from tensorflow.compat.v1.keras.optimizers import Adam, SGD
from keras.backend import sigmoid
from keras.utils import get_custom_objects
from keras.layers import concatenate

import tensorflow as tf

class AttentionWeightedAverage(Layer):
    """
    Computes a weighted average of the different channels across timesteps.
    Uses 1 parameter pr. channel to compute the attention value for a single timestep.
    """

    def __init__(self, return_attention=False, **kwargs):
        self.init = initializers.get('uniform')
        self.supports_masking = True
        self.return_attention = return_attention
        super(AttentionWeightedAverage, self).__init__(** kwargs)
        print("init")

    def build(self, input_shape):
        self.input_spec = [InputSpec(ndim=3)]
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[2], 1),
                                 name='{}_W'.format(self.name),
                                 initializer=self.init)
        self.trainable_weights = [self.W]
        super(AttentionWeightedAverage, self).build(input_shape)
        print("build")

    def call(self, x, mask=None):
        # computes a probability distribution over the timesteps
        # uses 'max trick' for numerical stability
        # reshape is done to avoid issue with Tensorflow
        # and 1-dimensional weights
        logits = K.dot(x, self.W)
        x_shape = K.shape(x)
        logits = K.reshape(logits, (x_shape[0], x_shape[1]))
        ai = K.exp(logits - K.max(logits, axis=-1, keepdims=True))

        # masked timesteps have zero weight
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            ai = ai * mask
        att_weights = ai / (K.sum(ai, axis=1, keepdims=True) + K.epsilon())
        print(K.int_shape(att_weights))
        weighted_input = x * K.expand_dims(att_weights)
        print(K.int_shape(weighted_input))
        
        return weighted_input
     

def DPCNN(maxlen,max_features, embed_size, embedding_matrix):
    print("DPCNN used")
    filter_nr = 64
    filter_size = 3
    max_pool_size = 3
    max_pool_strides = 2
    dense_nr = 256
    spatial_dropout = 0.2
    dense_dropout = 0.5
    train_embed = False
    conv_kern_reg = regularizers.l2(0.00001)
    conv_bias_reg = regularizers.l2(0.00001)
    
    comment = Input(shape=(maxlen,))
    emb_comment = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=train_embed)(comment)
    emb_comment = SpatialDropout1D(spatial_dropout)(emb_comment)

    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)
    block1 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1)
    block1 = BatchNormalization()(block1)
    block1 = PReLU()(block1)

    #we pass embedded comment through conv1d with filter size 1 because it needs to have the same shape as block output
    #if you choose filter_nr = embed_size (300 in this case) you don't have to do this part and can add emb_comment directly to block1_output
    resize_emb = Conv1D(filter_nr, kernel_size=1, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(emb_comment)
    resize_emb = PReLU()(resize_emb)
        
    block1_output = add([block1, resize_emb])
    block1_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block1_output)

    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block1_output)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
    block2 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2)
    block2 = BatchNormalization()(block2)
    block2 = PReLU()(block2)
        
    block2_output = add([block2, block1_output])
    block2_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block2_output)

    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block2_output)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
    block3 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3)
    block3 = BatchNormalization()(block3)
    block3 = PReLU()(block3)
        
    block3_output = add([block3, block2_output])
    block3_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block3_output)

    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block3_output)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)
    block4 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4)
    block4 = BatchNormalization()(block4)
    block4 = PReLU()(block4)

    block4_output = add([block4, block3_output])
    block4_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block4_output)

    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block4_output)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)
    block5 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5)
    block5 = BatchNormalization()(block5)
    block5 = PReLU()(block5)

    block5_output = add([block5, block4_output])
    block5_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block5_output)

    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block5_output)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)
    block6 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6)
    block6 = BatchNormalization()(block6)
    block6 = PReLU()(block6)

    block6_output = add([block6, block5_output])
    block6_output = MaxPooling1D(pool_size=max_pool_size, strides=max_pool_strides)(block6_output)

    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block6_output)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)
    block7 = Conv1D(filter_nr, kernel_size=filter_size, padding='same', activation='linear', 
                kernel_regularizer=conv_kern_reg, bias_regularizer=conv_bias_reg)(block7)
    block7 = BatchNormalization()(block7)
    block7 = PReLU()(block7)

    block7_output = add([block7, block6_output])
    output = GlobalMaxPooling1D()(block7_output)

    output = Dense(dense_nr, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(dense_dropout)(output)
    output = Dense(5, activation='sigmoid')(output)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = Model(comment, output)
    model.compile(loss='binary_crossentropy', 
        optimizer=optimizers.Adam(),
        metrics=['accuracy'])
    return model


def rcnn(maxlen,max_features, embed_size, embedding_matrix):
    train_embed = False

    comment = Input(shape = (maxlen,))
    embedding_comment = Embedding(max_features, embed_size, input_length = maxlen, weights=[embedding_matrix], trainable=train_embed)(comment)
    x_context = Bidirectional(GRU(128, return_sequences=True))(embedding_comment)
    x = Concatenate()([embedding_comment, x_context])
    convs = []
    for kernel_size in [1,2,3,4,5]:
        conv = Conv1D(64, kernel_size = 3, activation = 'relu')(x)
        convs.append(conv)
    poolings = [GlobalAveragePooling1D()(conv) for conv in convs] + [GlobalMaxPooling1D()(conv) for conv in convs]
    x = Concatenate()(poolings)
    output = Dense(5, activation = "softmax")(x)
    mirrored_strategy = tf.distribute.MirroredStrategy()

    with mirrored_strategy.scope():
        model = Model(comment, output)
    model.compile(loss='binary_crossentropy', 
    optimizer=optimizers.Adam(),
    metrics=['accuracy'])
    print(model.summary())
    return model


def rcnn1(maxlen,max_features, embed_size, embedding_matrix):
    train_embed = False

    comment = Input(shape = (maxlen,))
    embedding_comment = Embedding(max_features, embed_size, input_length = maxlen, weights=[embedding_matrix], trainable=train_embed)(comment)
    embedding_comment = SpatialDropout1D(0.2)(embedding_comment)
    rnn_1 = Bidirectional(LSTM(64, return_sequences=True))(embedding_comment)

    conv_2 = Conv1D(128, 2, kernel_initializer="normal", padding="valid", activation="relu", strides=1)(rnn_1)
    maxpool = GlobalMaxPooling1D()(conv_2)
    attn = Lambda(lambda t: AttentionWeightedAverage()(t))(conv_2)
    attn = Lambda(lambda t: K.sum(t, axis=1))(attn)
    average = GlobalAveragePooling1D()(conv_2)

    concatenated = Concatenate(axis = 1)([maxpool, attn, average])
    """
    x = Dropout(0.5)(concatenated)
    x = Dense(120, activation="relu")(x)
    output_layer = Dense(5, activation="sigmoid")(x)
    """
    output = Dropout(0.2)(concatenated)
    output = Dense(256, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(0.2)(output)
    output = Dense(5, activation='sigmoid')(output)

    model = Model(inputs=comment, outputs=output)
    adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=5, decay=1e-5)
    model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    model.summary()
    return model
  

