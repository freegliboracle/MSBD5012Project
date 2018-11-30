# feature engineering go here

# take in the entire dataframe
# return the processed dataframe
def pre_process(df):
    # Normalise the matchTypes to standard fromat
    df['matchType'][df['matchType'] == 'normal-solo'] = 'Solo'
    # df.loc[:,('matchType', 'normal-solo')] = 'Solo'
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.loc.html

    df['matchType'][df['matchType'] == 'solo-fpp'] = 'Solo'
    df['matchType'][df['matchType'] == 'normal-solo-fpp'] = 'Solo'
    df['matchType'][df['matchType'] == 'normal-duo-fpp'] = 'Duo'
    df['matchType'][df['matchType'] == 'normal-duo'] = 'Duo'
    df['matchType'][df['matchType'] == 'duo-fpp'] = 'Duo'
    df['matchType'][df['matchType'] == 'squad-fpp'] = 'Squad'
    df['matchType'][df['matchType'] == 'normal-squad'] = 'Squad'
    df['matchType'][df['matchType'] == 'normal-squad-fpp'] = 'Squad'
    df['matchType'][df['matchType'] == 'flaretpp'] = 'Other'
    df['matchType'][df['matchType'] == 'flarefpp'] = 'Other'
    df['matchType'][df['matchType'] == 'crashtpp'] = 'Other'
    df['matchType'][df['matchType'] == 'crashfpp'] = 'Other'


    return df

# Id,groupId,matchId,assists,boosts,damageDealt,DBNOs,headshotKills,heals,killPlace,killPoints,kills,
# killStreaks,longestKill,matchDuration,matchType,maxPlace,numGroups,rankPoints,revives,rideDistance,
# roadKills,swimDistance,teamKills,vehicleDestroys,walkDistance,weaponsAcquired,winPoints,winPlacePerc

# walkDistance, killPlace, kills, matchDuration, numGroups, maxPlace, killStreaks, boosts, rideDistance

# headshotKills_over_kills