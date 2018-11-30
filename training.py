import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from models import combo, deep
df = pd.read_csv("./data/sample.csv")
df_sample = df.sample(900000)

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

    plt.ylim([0, 0.2])
    plt.xlim([0,max(history.epoch)])
    plt.show()



# model = baseline()
# m_deep = deep()
# m_shallow = shallow()
# mwide = wide()
# mnarrow = narrow()
model = combo()
# mdeep = deep()
model.summary()

# keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=2, mode='auto', baseline=None)
# history = model.fit(train_x, train_y, epochs=100, batch_size=256, validation_split=0.2, verbose=2)
# h1 = mdeep.fit(train_x, train_y, epochs=100, batch_size=256, validation_split=0.2, verbose=2)

# plot_history([('baseline', history), ('deep', h1)])