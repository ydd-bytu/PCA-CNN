# plot
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