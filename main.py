import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
from feateng import pre_process

# https://www.kaggle.com/c/pubg-finish-placement-prediction/data
# the 2 data files are not tracked because they exceed github's 100MB limit

train = pd.read_csv('./data/train_V2.csv')
test = pd.read_csv('./data/test_V2.csv')
p_id = test[['Id']]

# rows to sample
sample = 100000

# FEATURE ENGINEERING
def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc', 'matchType']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

# dropping a bad data row without winPlacePerc
train.drop(2744604, inplace=True)

def process(train, isTrain=True):
    train['totalDistance'] = train['rideDistance'] + train['walkDistance'] + train['swimDistance']
    train['killsWithoutMoving'] = ((train['kills'] > 0) & (train['totalDistance'] == 0))

    # removing cheaters
    # kills without moving
    train.drop(train[train['killsWithoutMoving'] == True].index, inplace=True)
    # more than 9 road kills
    train.drop(train[train['roadKills'] > 9].index, inplace=True)
    # more than 50 kills (only possible if you are shroud)
    train.drop(train[train['kills'] > 50].index, inplace=True)
    # has kills across the map
    train.drop(train[train['longestKill'] >= 1000].index, inplace=True)
    # speed hack
    train.drop(train[train['rideDistance'] >= 20000].index, inplace=True)
    train.drop(train[train['swimDistance'] >= 2000].index, inplace=True)
    # supplies hack
    train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
    train.drop(train[train['heals'] >= 40].index, inplace=True)

    # total number of players in a match
    train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

    # normalizing features based on the total players
    train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
    train['maxPlaceNorm'] = train['maxPlace']*((100-train['playersJoined'])/100 + 1)
    train['matchDurationNorm'] = train['matchDuration']*((100-train['playersJoined'])/100 + 1)

    # selecting the features we want to perform additional engineering on
    features = ['Id', 'groupId', 'matchId', 'matchType', 'maxPlaceNorm', 'killsNorm', 'matchDurationNorm', 
                    'walkDistance', 'killPlace', 'numGroups', 'killStreaks', 'boosts', 'rideDistance']
    if isTrain:
        features.append('winPlacePerc')

    train = train[features]

    # reducing number of match type to 4
    train['matchType'].replace('normal-solo', 'solo', inplace=True)
    train['matchType'].replace('solo-fpp', 'solo', inplace=True)
    train['matchType'].replace('normal-solo-fpp', 'solo', inplace=True)
    train['matchType'].replace('normal-duo-fpp', 'duo', inplace=True)
    train['matchType'].replace('normal-duo', 'duo', inplace=True)
    train['matchType'].replace('duo-fpp', 'duo', inplace=True)
    train['matchType'].replace('squad-fpp', 'squad', inplace=True)
    train['matchType'].replace('normal-squad', 'squad', inplace=True)
    train['matchType'].replace('normal-squad-fpp', 'squad', inplace=True)
    train['matchType'].replace('flaretpp', 'other', inplace=True)
    train['matchType'].replace('flarefpp', 'other', inplace=True)
    train['matchType'].replace('crashtpp', 'other', inplace=True)
    train['matchType'].replace('crashfpp', 'other', inplace=True)

    train = rank_by_team(train)
    train = median_by_team(train)

    # one hot encode match type
    train = pd.get_dummies(train, columns=['matchType'])

    train.drop(['Id', 'groupId', 'matchId'], axis=1, inplace=True)
    return train

# print(train.columns.values)
train = process(train)
test= process(test, False)
df_sample = train.sample(sample)
x = df_sample.drop(columns = ['winPlacePerc']) #all columns except target
y = df_sample['winPlacePerc'] # Only target variable

train_x = np.array(x.values)
train_y = np.array(y.values)
test_x = np.array(test.values)

# print(train_x.shape, train_y.shape)

model = keras.Sequential([
        keras.layers.Dense(120, activation=tf.nn.relu, input_dim=40),
        keras.layers.Dense(80, activation=tf.nn.relu),
        keras.layers.Dense(40, activation=tf.nn.relu),
        keras.layers.Dense(20, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.relu),
        keras.layers.Dense(1)
    ])

model.compile(loss='mae', optimizer=tf.train.AdamOptimizer(), metrics=[keras.metrics.mean_absolute_error])


# keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, verbose=2, mode='auto', baseline=None, restore_best_weights=False)
model.fit(train_x, train_y, epochs=50, batch_size=256, validation_split=0.2, verbose=2)

prediction = model.predict(test_x ,batch_size=255)
results = pd.DataFrame(prediction, columns=['winPlacePerc'])
results = pd.concat([p_id, results], axis=1)
results.to_csv("sample_submission.csv", index=False)