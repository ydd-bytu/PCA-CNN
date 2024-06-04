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
model_path = '4regression_model.h5'

train_df = pd.read_csv('C:/Users/ASUS/JupyterProjects/data/train.txt', sep=" ", header=None)
#print(train_df)
train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
train_df.columns = ['id',  's1', 's2', 's3',
                     's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                     's15', 's16', 's17', 's18', 's19', 's20', 's21']
train_df = train_df.sort_values(['id'])
print(train_df)

# Data Labeling
rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
rul.columns = ['id', 'max']
train_df = train_df.merge(rul, on=['id'], how='left')
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


label_gen = [gen_labels(train_df[train_df['id'] == id], sequence_length, ['RUL'])
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
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout正则化

model.add(Dense(units=nb_out))
model.add(Activation("linear"))
model.compile(loss='mae', optimizer='Adam',metrics=['mae',r2_keras])
model.build((None, 50, 25))
print(model.summary())


# plot
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.fig = plt.figure()
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, np.sqrt(self.losses), label="loss")
        plt.plot(self.x, np.sqrt(self.val_losses), label="val_loss")
        plt.ylabel('loss - RMSE')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.title('model loss = ' + str(min(np.sqrt(self.val_losses))))
        plt.show()


plot_losses = PlotLosses()

# fit the network
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TerminateOnNaN

history = model.fit(train_X, train_Y, epochs=300, batch_size=400,
                    validation_data=(val_X, val_Y), shuffle=True, verbose=2,
                    callbacks=[EarlyStopping(monitor='val_loss', min_delta=0, patience=20, verbose=0, mode='min'),
                               ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min',
                                               verbose=0)]

                    )

# list all data in history
print(history.history.keys())

# history.save('lstm_rul.h5')

plt.plot(history.history['loss'],'r',    label="训练集")
plt.plot(history.history['val_loss'],'g',label="验证集")
# 设置坐标轴刻度朝内
plt.tick_params(axis='x', which='both', direction='in')
plt.tick_params(axis='y', which='both', direction='in')
# 调整刻度标签的字体大小
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.title("损失函数")
plt.legend()
plt.savefig("./image/Loss_image.SVG", dpi=None, facecolor='w', edgecolor='w',
          orientation='portrait', papertype=None, format=None,
          transparent=False, bbox_inches=None, pad_inches=0.1,
          frameon=None, metadata=None)