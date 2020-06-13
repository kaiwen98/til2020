import os
import gc
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

train = pd.read_csv('./input/TIL_NLP_train_dataset.csv')
test = pd.read_csv('./input/TIL_NLP_test_dataset.csv')

# %% [code] {"scrolled:true"}
train = pd.read_csv('./input/TIL_NLP_train_dataset.csv')
test = pd.read_csv('./input/TIL_NLP_test_dataset.csv')

train = shuffle(train)

partition = int(len(train)*0.8)
train[:partition].to_csv(index = False, path_or_buf = './input/TIL_NLP_train1_dataset.csv')
train[partition+1:len(train)-1].to_csv(index = False, path_or_buf = './input/TIL_NLP_unseen_dataset.csv')