
##############################################################################################################
# EconsRox NLP Model Submission                                                                              #
# Integrated Stack Ensemble with 5 Deep pyramid CNN and 5 RCNN pre-trained deep learning models              #
# DPCNN: http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf                                       #
# RCNN: https://github.com/nielintos/Tox-RCNN <Inspired>                                                     #
# Stacked Ensemble: https://machinelearningmastery.com/stacking-ensemble-for-deep-learning-neural-networks/  #
##############################################################################################################

from __future__ import absolute_import, division
import math
import tensorflow as tf
from keras.layers import Dense, Input, Embedding, Lambda, Dropout, Activation, SpatialDropout1D, Reshape, GlobalAveragePooling1D, Flatten, Bidirectional, CuDNNGRU, add, Conv1D, GlobalMaxPooling1D
from keras import optimizers
from keras import initializers
from keras.layers import InputSpec, Layer, Concatenate
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
from numpy import dstack, argmax, array
from keras.utils import to_categorical
from sklearn.linear_model import LogisticRegression

import tensorflow as tf



EMBEDDING_FILE = './input/word_embeddings.pkl'
s_model_file = 'model_cv1.json'
model_file = 'model_cv.json'
s_weights_file = 'model_cv1.h5'
weights_file = 'model_cv.h5'

def f1(y_pred0, y_test0):
    TP = 0.0
    FP = 0.0
    FN = 0.0
    TN = 0.0
    y_pred = [item for sublist in y_pred0 for item in sublist]
    y_test = [item for sublist in y_test0 for item in sublist]
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
"""
def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind]
"""
def schedule(epochs):
    ans = 0.001 * math.exp(-0.3*epochs)
    print("epochs: ", ans)
    return ans

def datagen(train, test, gen_test = False):
    max_features = 4248
    maxlen = 200
    train = shuffle(train)
    X_train = train["word_representation"].fillna("fillna").values
    y_train = train[["outwear", "top", "trousers", "women dresses", "women skirts"]].values
    X_test = test["word_representation"].fillna("fillna").values    
    #y_test = test[["outwear", "top", "trousers", "women dresses", "women skirts"]].values
    #y_test = y_test.tolist()
    print('preprocessing start')
    tokenizer = text.Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(X_train) + list(X_test))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    
    x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

    del X_train, X_test, train, test
    gc.collect()
    if gen_test is False:
        return x_train, x_test, y_train, tokenizer
    else:
        return x_test
   

def gauge_acc(y_pred, y_test):
    accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_pred))])/len(y_pred) * 100
    print(accuracy, "%")
    print(f1(y_pred, y_test))
    return f1(y_pred,y_test)

def classifier(model_name, emb_mean, emb_std, embeddings_index):
    max_features = 4248
    maxlen = 200
    embed_size = 100

    d_train = pd.read_csv('./input/TIL_NLP_train_dataset.csv')
    d_test = pd.read_csv('./input/TIL_NLP_test_dataset.csv')
    print('running classifier')

    tokenizer = text.Tokenizer(num_words=max_features)
    x_train, x_test, y_train, tokenizer = datagen(d_train, d_test)
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))

    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i-1] = embedding_vector
    
    num_models_rcnn = 5
    num_models_dpcnn = 5
    num_models = num_models_rcnn + num_models_dpcnn

    print('preprocessing done')
    members = []
    for i in range(num_models_rcnn): 
        model = models.rcnn1(maxlen, max_features, embed_size, embedding_matrix)
        model.load_weights('final_ensemble_rcnn_'+str(i)) 
        members.append(model)

    for i in range(int(num_models_dpcnn)): 
        model = models.DPCNN(maxlen, max_features, embed_size, embedding_matrix)
        model.load_weights('final_ensemble_dpcnn_'+str(i)) 
        members.append(model)
    
    for i in range(len(members)):
	    model = members[i]
	    for layer in model.layers:
		    # make not trainable
		    layer.trainable = False
		    # rename to avoid 'unique layer name' issue
		    layer.name = 'ensemble_' + str(i+1) + '_' + layer.name
    
    # Metalearner
    supermodel_input = [model.input for model in members]
    supermodel_output = [model.output for model in members]
    output = Concatenate()(supermodel_output)
    output = Dense(25, activation='linear')(output)
    output = BatchNormalization()(output)
    output = PReLU()(output)
    output = Dropout(0.3)(output)
    output = Dense(5, activation='sigmoid')(output)
    
    
    supermodel = Model(input = supermodel_input, output = output)
    #supermodel.load_weights('final_ensemble_complete1.h5')
    adam_optimizer = optimizers.Adam(lr=1e-3, clipvalue=5, decay=1e-5)
    supermodel.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])
    

    
    num_folds = 5
    num = 0
    numod = 1
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    for train, test in kfold.split(x_train, y_train):
        #x_test, y_test = datagen(d_train, d_test, gen_test = True)
        print("Training Fold number: ", num)
        num += 1
        """
        ypred_arr = []
        for model in members:
            pred = model.predict(x_test)
            pred = [[1 if i > 0.5 else 0 for i in r] for r in pred] 
            if float(gauge_acc(pred, y_test)) > 0.92:
                ypred_arr.append(pred)

        #ypred_arr = [model.predict(x_test) for model in members]
        for x in ypred_arr:
            x = [[1 if i > 0.5 else 0 for i in r] for r in x] 
            print("model "+ str(numod) +" used")
            gauge_acc(x, y_test)
            numod += 1

        numod = 0
        
        yhat = array(ypred_arr)
        y_pred = np.mean(yhat, axis = 0)
        """
        inputX = [x_train[train] for i in range(num_models)]
        testX = [x_train[test] for i in range(num_models)]
        unseenX = [x_test for i in range(num_models)]
        inputY = y_train[train]
 
        batch_size = 128
        epochs = 25
        lr = callbacks.LearningRateScheduler(schedule)
        ra_val = RocAucEvaluation(validation_data=(x_train[train], y_train[train]), interval = 1)
        es = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = 2, restore_best_weights = True, mode = 'min')
        mc = ModelCheckpoint(model_name, monitor='val_loss', mode='min', verbose=1, save_best_only= True, save_weights_only = True)
    
        supermodel.fit(inputX, inputY, epochs=epochs, batch_size = batch_size, validation_data=(testX, y_train[test]), callbacks = [lr, ra_val, es, mc] ,verbose = 1)
        
        y_pred = supermodel.predict(unseenX)
        print("Ensemble prediction: ")

        
        #accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_pred))])/len(y_pred) * 100
        #print([y_pred[i] == y_test[i] for i in range(len(y_pred))])
        #print(accuracy, "%")
        #print(f1(y_pred, y_test))
    y_pred = supermodel.predict(unseenX)
    y_pred = [[1 if i > 0.5 else 0 for i in r] for r in y_pred]
    submission = pd.read_csv('./input/NLP_submission_example')
    submission[["outwear", "top", "trousers", "women dresses", "women skirts"]] = y_pred
    submission.to_csv('final_prediction.csv', index=False)

    return model



# %% [code] {"scrolled:true"}
if __name__ == "__main__":
    global embedding_index

    modelcat = []
    emb_mean, emb_std, embeddings_index = extract_embed(EMBEDDING_FILE)

    model_name = "final_ensemble_assembled.h5"
    model = classifier(model_name,emb_mean, emb_std, embeddings_index)
       
    
    

# %%
