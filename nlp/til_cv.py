#dpcnn http://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf
#dpcnn with conv1d, model architecture and all parameters copied from neptune-ml since it's publicly available
#https://github.com/neptune-ml/kaggle-toxic-starter/blob/master/best_configs/fasttext_dpcnn.yaml
#Got it to PLB 0.984 with 10fold cv on local computer after playing with parameters
#Try to improve score on your own local pc or throw it in the blender with the rest of them :)
# %% [code] {"scrolled:true"}
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
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects

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

def _load_model():
    # load json and create model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weights_file)
    print("Old model: ", type(model))
    
    model.compile(loss='binary_crossentropy', 
            optimizer=optimizers.Adam(),
            metrics=['accuracy'])
            
    print("Loaded model from disk")
    return model

def _save_model(model):
    model_json = model.to_json()
    with open('model_cv1.json', 'w') as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(s_weights_file)
    print("Saved model to disk")
"""
def schedule(ind):
    a = [0.001, 0.0005, 0.0001, 0.0001]
    return a[ind] 
"""

def schedule(epoch):
    if epoch < 10:
        return float(0.001)
    else:
        return float(0.001*tf.math.exp(-0.1))



#straightfoward preprocess



# %% [code] {"scrolled:true"}
def classifier(model, emb_mean, emb_std, embeddings_index):
    train = pd.read_csv('./input/TIL_NLP_train1_dataset.csv')
    test = pd.read_csv('./input/TIL_NLP_unseen_dataset.csv')
    print('running classifier')

    max_features = 4248
    print(max_features)
    maxlen = 200
    embed_size = 100
    train = shuffle(train)
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

    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=4)
    # K.set_session(tf.Session(graph=tf.get_default_graph(), config=session_conf))

    #model
    #wrote out all the blocks instead of looping for simplicity


    
    
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
    
    #model = Model(comment, output)
    # print("Correct model: ", type(model))
    
    model.compile(loss='binary_crossentropy', 
            optimizer=optimizers.Adam(),
            metrics=['accuracy'])
    
    num_folds = 5
    num = 0
    kfold = KFold(n_splits=num_folds, shuffle=True)
    
    for train, test in kfold.split(x_train, y_train):
    
        print("Training Fold number: ", num)
        batch_size = 128
        epochs = 20
        lr = callbacks.LearningRateScheduler(schedule)
        ra_val = RocAucEvaluation(validation_data=(x_train[test], y_train[test]), interval = 1)
        es = EarlyStopping(monitor = 'val_loss', verbose = 1, patience = int(15), restore_best_weights = True, mode = 'min')
        mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1)
        model.fit(x_train[train], y_train[train], batch_size=batch_size, epochs=epochs, validation_data=(x_train[test], y_train[test]), callbacks = [lr, ra_val, es, mc] ,verbose = 1)
        num += 1
        
        
        y_pred = model.predict(x_test)
        y_pred = [[1 if i > 0.5 else 0 for i in r] for r in y_pred]
        
        accuracy = sum([y_pred[i] == y_test[i] for i in range(len(y_pred))])/len(y_pred) * 100
        print([y_pred[i] == y_test[i] for i in range(len(y_pred))])
        print(accuracy, "%")
        print(f1(y_pred, y_test))
        
        """
        submission = pd.read_csv('../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv')
        submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
        submission.to_csv('dpcnn_test_preds.csv', index=False)
        """
    
    return model

# %% [code] {"scrolled:true"}
if __name__ == "__main__":
    global embedding_index
    start = 0
    emb_mean, emb_std, embeddings_index = extract_embed(EMBEDDING_FILE)
    while(start<1):
        model = load_model('best_model.h5')
        model = classifier(model,emb_mean, emb_std, embeddings_index)
        _save_model(model)
        start = start + 1




# %%
