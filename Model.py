import keras
import keras.backend as K
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM
from IPython.display import SVG, clear_output
# from keras.utils import plot_model
from tensorflow.keras.utils import plot_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn import preprocessing


plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
np.random.seed(1234)
PYTHONHASHSEED = 0
# define path to save model
model_path = '4r_model.h5'

train_df = pd.read_csv('C:/Users/ASUS/JupyterProjects/data/train.txt', sep=" ", header=None)
#print(train_df)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id',  's11', 's12', 's13',
                     's21', 's22', 's23', 's31', 's32', 's33', 's41', 's42', 's43', 's51', 's52',
                     's53', 's61', 's62', 's63', 's71', 's72', 's83']
train_df = train_df.sort_values(['id'])
print(train_df)

# Data Labeling
ID = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
ID.columns = ['id', 'max']
train_df = train_df.merge(ID, on=['id'], how='left')
train_df['Label'] = train_df['max'] - train_df['cycle']
train_df.drop('max', axis=1, inplace=True)


# MinMax normalization
train_df['Label'] = train_df['max']
cols_normalize = train_df.columns.difference(['id'])
print(cols_normalize)
min_max_scaler = preprocessing.MinMaxScaler()
norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]),
                             columns=cols_normalize,
                             index=train_df.index)

join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
train_df = join_df.reindex(columns = train_df.columns)
print(train_df)

# pick a large window size of 50 cycles
sequence_length = 50


def gen_sequence(id_df, seq_length, seq_cols):
    data_array = id_df[seq_cols].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - seq_length), range(seq_length, num_elements)):
        yield data_array[start:stop, :]

# pick the feature columns
sensor_cols = ['s' + str(i) for i in range(1,22)]
# print(sensor_cols)
sequence_cols = ['s1', 's2', 's3', 'id']

sequence_cols.extend(sensor_cols)
print(sequence_cols)

val=list(gen_sequence(train_df[train_df['id']==1], sequence_length, sequence_cols))
print(len(val))
# print(train_df)
# print(test_df)
# transform each id of the train dataset in a sequence
seq_gen = (list(gen_sequence(train_df[train_df['id']==id], sequence_length, sequence_cols))
           for id in train_df['id'].unique())

# generate sequences and convert to np array
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
# print(seq_array.shape)
# print(seq_array)

# generate labels
def gen_labels(id_df, seq_length, label):
    data_array = id_df[label].values
    num_elements = data_array.shape[0]
    return data_array[seq_length:num_elements, :]


label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['ID'])
             for id in train_df['id'].unique()]

label_array = np.concatenate(label_gen).astype(np.float32)
print(label_array.shape)

SEED = 1234
np.random.seed(SEED)
from sklearn.model_selection import train_test_split
train_X, val_set_X, train_Y, val_set_Y = train_test_split(seq_array, label_array, test_size=0.20, random_state=SEED)
test_X, val_X, test_Y, val_Y = train_test_split(val_set_X, val_set_Y, test_size=0.50, random_state=SEED)


def r2_keras(y_true, y_pred):
    res = K.sum(K.square(y_true - y_pred))
    tot = K.sum(K.square(y_true - K.mean(y_true)))
    return (1 - res / (tot + K.epsilon()))

import numpy as np
from sklearn.decomposition import PCA

pca = PCA(n_components=1)
pca.fit(train_X)

weights = pca.components_

single_objective = np.dot(train_X, weights.T)

class LagrangeNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # 随机初始化权重
        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size)

        # 初始化 Lagrange 乘子
        self.lagrange_multiplier = np.zeros((hidden_size, output_size))

    def forward(self, inputs):
        # 前向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        return final_outputs

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def backward(self, inputs, targets, learning_rate):
        # 反向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        output_delta = output_errors * self.sigmoid_derivative(final_outputs)

        hidden_errors = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)

        # 更新权重
        self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate

    def lagrange_backward(self, inputs, targets, constraint, learning_rate):
        # Lagrange 反向传播
        hidden_inputs = np.dot(inputs, self.weights_input_hidden)
        hidden_outputs = self.sigmoid(hidden_inputs)

        final_inputs = np.dot(hidden_outputs, self.weights_hidden_output)
        final_outputs = self.sigmoid(final_inputs)

        output_errors = targets - final_outputs
        output_delta = output_errors * self.sigmoid_derivative(final_outputs)

        hidden_errors = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_errors * self.sigmoid_derivative(hidden_outputs)

        # 更新权重
        self.weights_hidden_output += hidden_outputs.T.dot(output_delta) * learning_rate

        # 更新 Lagrange 乘子
        self.lagrange_multiplier += constraint * hidden_outputs * learning_rate

        # 更新输入层到隐藏层的权重
        self.weights_input_hidden += inputs.T.dot(hidden_delta) * learning_rate \
                                     + self.lagrange_multiplier.dot(self.weights_hidden_output.T) * learning_rate

from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import GRU
#CNN
nb_features = seq_array.shape[2]
nb_out = label_array.shape[1]

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())
LagrangeNeuralNetwork = LagrangeNeuralNetwork(128,256,21)
model.add(LagrangeNeuralNetwork)
model.add(Dropout(0.5))  # Dropout正则化
model.add(Activation("linear"))
model.compile(loss='mae', optimizer='Adam',metrics=['mae',r2_keras])
model.build((None, 50, 25))
print(model.summary())


