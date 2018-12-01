import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from models import combo, combo_l1, combo_l2, combo_l12
df = pd.read_csv("./data/sample.csv")
df_sample = df.sample(200000)

x = df_sample.drop(columns = ['winPlacePerc'])
y = df_sample['winPlacePerc']

train_x = np.array(x.values)
train_y = np.array(y.values)

# print(train_x.shape, train_y.shape)

def plot_history(histories, key='mean_absolute_error'):
    plt.figure(figsize=(16,10))
        
    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                    '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_',' ').title())
    plt.legend()

    plt.ylim([0, 0.5])
    plt.xlim([0,max(history.epoch)])
    plt.show()



model = combo()
m1 = combo_l1()
m2 = combo_l2()
m12 = combo_l12()

keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=2, mode='auto', baseline=None)
history = model.fit(train_x, train_y, epochs=75, batch_size=256, validation_split=0.2, verbose=2)
h1 = m1.fit(train_x, train_y, epochs=75, batch_size=256, validation_split=0.2, verbose=2)
h2 = m2.fit(train_x, train_y, epochs=75, batch_size=256, validation_split=0.2, verbose=2)
h3 = m12.fit(train_x, train_y, epochs=75, batch_size=256, validation_split=0.2, verbose=2)

plot_history([('baseline', history), ('l1', h1), ('l2', h2), ('l1_2', h3)])