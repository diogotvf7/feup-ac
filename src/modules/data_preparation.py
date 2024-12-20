import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def dropColumns(dataset):
    dataset['awards_players'].drop(columns=['lgID'],  inplace = True)
    dataset['coaches'].drop(columns=['lgID'], inplace = True)
    dataset['players_teams'].drop(columns=['lgID', 'oRebounds', 'dRebounds', 'stint'], inplace = True) # Only total rebounds is important
    dataset['players'].drop(columns=['firstseason', 'lastseason'], inplace = True)
    dataset['series_post'].drop(columns=['lgIDWinner', 'lgIDLoser'], inplace = True)
    dataset['teams_post'].drop(columns=['lgID'], inplace = True)
    dataset['teams'].drop(columns=['lgID', 'divID', 'seeded','tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB', 'arena'], inplace=True)
    
    return dataset    

def checkDuplicates(dataset):
    for name, dataset in dataset.items():
        if (dataset.duplicated().any()):
            raise Exception("Duplicate data found in " + name)
        
def checkNull(dataset):
    for name, dataset in dataset.items():
        if (dataset.isna().any().any()):
            raise Exception("Null values found in " + name)

def removePlayersWithoutTeam(dataset):
    dataset['players'] = dataset['players'][dataset['players']['bioID'].isin(dataset['players_teams']['playerID'])]

    return dataset


def replaceMissingValues(train_data, missing_data, target, feat1, feat2, original_dataset):
    X_train = pd.get_dummies(train_data[[feat1, feat2]], drop_first=True)
    y_train = train_data[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = pd.get_dummies(missing_data[[feat1, feat2]], drop_first=True)
    X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)

    predicted_values = np.round(model.predict(X_missing)).astype(int)
    
    original_dataset.loc[original_dataset[target] == 0, target] = predicted_values

def fillMissingWeights(dataset):
    missing = dataset['players'][dataset['players']['weight'] == 0]
    train_data = dataset['players'][dataset['players']['weight'] != 0]

    replaceMissingValues(train_data, missing, 'weight', 'height', 'pos', dataset['players'])

    return dataset

def fillWithNone(dataset):
    dataset['players']['college'] = dataset['players']['college'].fillna('none')
    dataset['players']['collegeOther'] = dataset['players']['collegeOther'].fillna('none')
    dataset['teams']['firstRound'] = dataset['teams']['firstRound'].fillna('DQ')
    dataset['teams']['semis'] = dataset['teams']['semis'].fillna('DQ')
    dataset['teams']['finals'] = dataset['teams']['finals'].fillna('DQ')


    return dataset

def checkDuplicatedData(dataset):
    for name, data in dataset.items():
        if (data.duplicated().any()):
            raise Exception("Duplicate data found in " + name)

def checkNullValue(dataset):
    for name, data in dataset.items():
        if (data.isna().any().any()):
            raise Exception("Null values found in " + name)
            
# When a player has multiple entries for the same year (e.g. change team mid-season), 
# we aggregate the stats for that year into a single entry
# This is done by summing the stats for that year. The teamID is taken from the first entry
def merge_player_year_data(dataset): 
    return dataset.groupby(['playerID', 'year'], as_index=False).agg({
        'tmID': 'first',        
        'GP': 'sum',            
        'GS': 'sum',            
        'minutes': 'sum',       
        'points': 'sum',        
        'fgMade': 'sum',        
        'fgAttempted': 'sum',   
        'ftMade': 'sum',        
        'ftAttempted': 'sum',   
        'threeMade': 'sum',     
        'threeAttempted': 'sum',
        'rebounds': 'sum',      
        'steals': 'sum',        
        'blocks': 'sum',        
        'assists': 'sum',       
        'turnovers': 'sum'      
    })
    
def prepare_competition_data(dataset):
    coaches = dataset['coaches']
    teams = dataset['teams']
    player_teams = dataset['players_teams']
    
    # Merge coaches and teams
    coaches_teams = pd.merge(coaches, teams, on=['tmID', 'year', 'lgID'], how='left')
    
    # Merge player_teams with coaches_teams
    competition_data = pd.merge(player_teams, coaches_teams, on=['tmID', 'year', 'stint', 'lgID'], how='left')
    
    return competition_data[['playerID','tmID','coachID']]

def dataPreparation(dataset):
    # Drop unwanted columns
    dataset = dropColumns(dataset)

    # Remove players without a team
    dataset = removePlayersWithoutTeam(dataset)

    # Fill missing weights
    dataset = fillMissingWeights(dataset)

    # Fill missing values with 'none'
    dataset = fillWithNone(dataset)

    # Merge player year data
    merge_player_year_data(dataset['players_teams'])

    # Check for duplicated data
    checkDuplicatedData(dataset)
            
    # Check for null data
    checkNullValue(dataset)
        
    return dataset
    
    
