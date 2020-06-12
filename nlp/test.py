# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

# %% [code]


# %% [code] {"scrolled:true"}
import pickle
from random import shuffle
import pandas as pd
import tensorflow as tf
import numpy as np
import random
import keras

from sklearn.model_selection import train_test_split
from tensorflow.compat.v1.keras.models import Model, Sequential
from tensorflow.compat.v1.keras.layers import *
from tensorflow.compat.v1.keras.optimizers import Adam, SGD


# %% [code] {"scrolled:true"}
def extract_label(fd):
    train_labels = []
    with open(fd, 'rb') as f:
        data = pickle.load(f)
        train_labels.extend(data)
    return data


train_labels = extract_label('./input/word_embeddings.pkl')
print("lol")
# print(train_labels)

# %% [code] {"scrolled:true"}
train = 0
from sklearn.utils import shuffle
while(train < 10):
    train_df = pd.read_csv("./input/TIL_NLP_train_dataset.csv").fillna("blank")
    train_df = shuffle(train_df)
    train_df["word_representation"] = train_df["word_representation"]
    train_samples = train_df["word_representation"].values
    train_labels = train_df[["outwear", "top", "trousers", "women dresses", "women skirts"]].values


    # %% [code] {"scrolled:true"}
    x_train, x_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size=0.2)

    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical

    # %% [code] {"scrolled:true"}
    from keras.preprocessing.text import Tokenizer
    from keras.preprocessing.sequence import pad_sequences
    from keras.utils import to_categorical

    BATCH_SIZE = 512
    MAX_SEQUENCE_LENGTH = 100
    MAX_NUM_WORDS = 10000
    EMBEDDING_DIM = 100
    VALIDATION_SPLIT = 0.2
    print("Xtrain:")
    print(len(x_train))
    print("Ytrain:")
    print(len(y_train))

    # second, prepare text samples and their labels
    print('Processing text dataset')

    # finally, vectorize the text samples into a 2D integer tensor
    tokenizer = Tokenizer(num_words=MAX_NUM_WORDS)
    tokenizer.fit_on_texts(list(x_train))
    sequences = tokenizer.texts_to_sequences(x_train)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    data_train, val_test, data_label, val_label = train_test_split(data, y_train, test_size=0.2)

    tokens_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64', name="SentencesInput")
    embedded_sequences = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM)(tokens_input)

    """
    # %% [code] {"scrolled:true"}
    x = Conv1D(512, 5, activation='relu')(embedded_sequences)
    x = Dropout(0.5)(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(256, 2, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = MaxPooling1D(4)(x)
    x = Conv1D(128, 2, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    final = Dense(5, activation='sigmoid')(x)

    model = Model(inputs=[tokens_input], outputs=[final])
    model.summary()

    # %% [code] {"scrolled:true"}
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    """

    import tensorflow as tf

    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights("model.h5")
    print("Loaded model from disk")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(data_train, data_label, epochs=5, steps_per_epoch = 3, batch_size=512, validation_data=(val_test, val_label), verbose=1)

    tokenizer.fit_on_texts(list(x_test))
    sequences = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # %%

    tokenizer.fit_on_texts(list(x_test))
    sequences = tokenizer.texts_to_sequences(x_test)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data_test = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


    # %%
    yhat = model.predict(data_test, verbose=0)
    print(yhat)
    preds_labels = [[1 if x > 0.5 else 0 for idx,x in enumerate(i) ] for i in yhat]
    from sklearn.metrics import f1_score, roc_auc_score, multilabel_confusion_matrix, average_precision_score, precision_recall_curve
    print(y_test[:10])
    print(60*'-')
    print(preds_labels[:10])
    print(multilabel_confusion_matrix(y_test, preds_labels))

    #print(average_precision_score(y_test, preds_labels))
    #print(f1_score(y_test, preds_labels, average='micro'))
    print('ROC-AUC Score:', roc_auc_score(y_test, preds_labels))

    # %%
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")
    train = train+1



# %%
