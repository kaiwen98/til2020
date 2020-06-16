#dpcnn http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
#dpcnn with conv1d, model architecture and all parameters copied from neptune-ml since it's publicly available
#https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/best_configs/fasttext_dpcnn.yaml
#Got it to PLB 0.984 with 10fold cv on local computer after playing with parameters
#Try to improve score on your own local pc or throw it in the blender with the rest of them :)
# %% [code] {"scrolled:true"}
from __future__ import absolute_import, division
import math
import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras import optimizers
from keras import initializers
from keras.layers import InputSpec, Layer
from keras import backend as K
import keras
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
from keras.preprocessing import text, sequence
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from keras import optimizers
from keras import initializers, regularizers, constraints, callbacks
from sklearn.utils import shuffle
from tensorflow.compat.v1.keras.optimizers import Adam, SGD
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.utils import get_custom_objects
import models 

import tensorflow as tf



EMBEDDING_FILE = './input/word_embeddings.pkl'

def f1(y_pred0, y_test0):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    y_pred = [item for sublist in y_pred0 for item in sublist]
    y_test = [item for sublist in y_test0 for item in sublist]
    print(y_pred)
    print(y_test)
    for i in range(len(y_pred)):
        if y_pred[i] == 0: 
            if y_test[i] == 0: 
                TN += 1
            else:
                FN += 1
        else:
            if y_test[i] == 0: 
                FP += 1
            else: 
                TP += 1
    
    print("FP: ", FP,"  TP: ", TP,"  FN: ", FN, " TN: ", TN)
    precision = TP /(TP + FP)
    recall = TP/(TP+FN)
    return float(2 * precision * recall / (precision + recall))


def swish(x, beta = 1): return (x * sigmoid(beta*x))

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))

def extract_embed(fd):
    with open(fd, 'rb') as f:
        embeddings_index = pickle.load(f)
    all_embs = np.stack(embeddings_index.values())
    return all_embs.mean(), all_embs.std(), embeddings_index

def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001, 0.00005, 0.00005]
    return a[ind] 

#straightfoward preprocess

# %% [code] {"scrolled:true"}
def classifier(model_name, emb_mean, emb_std, embeddings_index):
    train = pd.read_csv('./input/TIL_NLP_train1_dataset.csv')
    test = pd.read_csv('./input/TIL_NLP_unseen_dataset.csv')
    print('running classifier')

    max_features = 4248
    print(max_features)
    maxlen = 200
    embed_size = 100
    train = shuffle(train)
    test = shuffle(test)
    X_train = train["word_representation"].fillna("fillna").values
    y_train = train[["outwear", "top", "trousers", "women dresses", "women skirts"]].values
    X_test = test["word_representation"].fillna("fillna").values    
    y_test = test[["outwear", "top", "trousers", "women dresses", "women skirts"]].values
    y_test = y_test.tolist()
    print('preprocessing start')
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)

    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    del X_train, X_test, train, test
    gc.collect()

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector
        
    print('preprocessing done')

    model = models.DPCNN(maxlen, max_features, embed_size, embedding_matrix)

    num_folds = 8
    num = 0
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    for train, test in kfold.split(x_train, y_train):
    
        print("Training Fold number: ", num)
        batch_size = 128
        epochs = 6
        lr = callbacks.LearningRateScheduler(schedule)
        ra_val = RocAucEvaluation(validation_data=(x_train[test], y_train[test]), interval = 1)
        es = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 2, restore_best_weights = True, mode = 'min')
        mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only= True, save_weights_only=True)
        model.fit(x_train[train], y_train[train], batch_size=batch_size, epochs=epochs, validation_data=(x_train[test], y_train[test]), callbacks = [lr, ra_val, es, mc] ,verbose = 1)
        num += 1
        
        y_pred = model.predict(x_test)
        y_pred = [[1 if i > 0.5 else 0 for i in r] for r in y_pred]
        
        accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_pred))])/len(y_pred) * 100
        print([y_pred[i] == y_test[i] for i in range(len(y_pred))])
        print(accuracy, "%")
        print(f1(y_pred, y_test))
    
    return model

# %% [code] {"scrolled:true"}
if __name__ == "__main__":
    global embedding_index
    start = 0
    emb_mean, emb_std, embeddings_index = extract_embed(EMBEDDING_FILE)
    while(start<5):
        model = 'ensemble_dpcnn_' + str(start) 
        model = classifier(model,emb_mean, emb_std, embeddings_index)
        #_save_model(model)
        start = start + 1




# %%
