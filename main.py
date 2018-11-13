import numpy as np
import pandas as pd
from feateng import pre_process
from util import *

# https://www.kaggle.com/c/pubg-finish-placement-prediction/data
# the 2 data files are not tracked because they exceed github's 100MB limit

train = pd.read_csv('./data/train_V2.csv')
data = pre_process(train)

matchIds = getMatchIds(data)
match = getSpecificMatchById(data, matchIds[0])
matches_solo = getMatchType(data, 'Solo')

# print train.head()