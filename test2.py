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
import tensorflow as tf

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


print(tf.__version__)


dictoken = extract_label('./input/word_embeddings.pkl')

# print(train_labels)



# %% [code] {"scrolled:true"}
train = 0
from sklearn.utils import shuffle



train_df = pd.read_csv("./input/TIL_NLP_train_dataset.csv").fillna("blank")
train_df = shuffle(train_df)
train_df["word_representation"] = train_df["word_representation"]
train_samples = train_df["word_representation"].values
train_labels = train_df[["outwear", "top", "trousers", "women dresses", "women skirts"]].values


# %% [code] {"scrolled:true"}
x_train, x_test, y_train, y_test = train_test_split(train_samples, train_labels, test_size=0.2)


def func(x):
    #print(x)
    return dictoken.get(x)

data = []
x_train1 = [i.split() for i in x_train]
print(x_train1[0])

data1 = [[func(i) for i in j] for j in x_train1]
print(data1[0][3][1])
print(dictoken.get('w1'))
print(len(train_labels))

# print(type(data1[0][0]))

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


#print(list(x_train)[0])
sequences = data1

#embedded_sequences = tf.convert_to_tensor(sequences)

# %% [code] {"scrolled:true"}

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

# %% [code] {"scrolled:true"}
data_train, val_test, data_label, val_label = train_test_split(data, y_train, test_size=0.2)

tokens_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int64', name="SentencesInput")
embedded_sequences = Embedding(MAX_NUM_WORDS, EMBEDDING_DIM)(tokens_input)

