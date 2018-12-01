import tensorflow as tf
from tensorflow import keras

def baseline():
    model = keras.Sequential([
        keras.layers.Dense(40, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model

def wide():
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(80, activation=tf.nn.relu),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model

def narrow():
    model = keras.Sequential([
        keras.layers.Dense(20, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(5, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model

def shallow():
    model = keras.Sequential([
        keras.layers.Dense(40, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model

def model():
    model = keras.Sequential([
        keras.layers.Dense(80, activation=tf.nn.relu, input_dim=40, kernel_regularizer=keras.regularizers.l1_l2(0.0001, 0.01)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(40, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1_l2(0.0001, 0.001)),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model

def combo():
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(80, activation=tf.nn.relu),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=[keras.metrics.mean_absolute_error])
    return model

def deep():
    model = keras.Sequential([
        keras.layers.Dense(360, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(180, activation=tf.nn.relu),
        keras.layers.Dense(90, activation=tf.nn.relu),
        keras.layers.Dense(60, activation=tf.nn.relu),
        keras.layers.Dense(30, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=['accuracy', keras.metrics.mean_absolute_error])
    return model


def combo_l1():
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40, kernel_regularizer=keras.regularizers.l1(0.00005)),
        keras.layers.Dense(80, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1(0.00005)),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=[keras.metrics.mean_absolute_error])
    return model

def combo_l2():
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(80, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=[keras.metrics.mean_absolute_error])
    return model

def combo_l12():
    model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40, kernel_regularizer=keras.regularizers.l1_l2(0.00005, 0.001)),
        keras.layers.Dense(80, activation=tf.nn.relu, kernel_regularizer=keras.regularizers.l1_l2(0.00005, 0.001)),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

    model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=[keras.metrics.mean_absolute_error])
    return model