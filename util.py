# utility file for quickly extracting information from a dataframe

# returns data for a specific match type
# available match types: Solo, Duo, Squad, and Other
def getMatchType(df, matchType):
    return df[df['matchType'] == matchType]

# returns all available match Ids
def getMatchIds(df):
    return df['matchId'].unique()

def getSpecificMatchById(df, id):
    return df['matchId'][df['matchId'] == id]