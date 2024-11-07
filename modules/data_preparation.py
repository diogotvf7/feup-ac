import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def dropColumns(dataset):
    dataset['awards_players'].drop(columns=['lgID'],  inplace = True)
    dataset['coaches'].drop(columns=['lgID'], inplace = True)
    dataset['players_teams'].drop(columns=['lgID'], inplace = True)
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
    dataset['players'].loc[:, 'college'] = dataset['players']['college'].fillna('none')
    dataset['players'].loc[:, 'collegeOther'] = dataset['players']['collegeOther'].fillna('none')

    return dataset


FUNCTIONS = [
    dropColumns,
    checkDuplicates,
    checkNull,
    removePlayersWithoutTeam,
    # replaceMissingValues,
    fillMissingWeights
]

def dataPreparation(dataset):
    # Drop unwanted columns
    dataset = dropColumns(dataset)

    # Check for duplicated data
    for name, dataset in dataset.items():
        if (dataset.duplicated().any()):
            raise Exception("Duplicate data found in " + name)
            
    # Check for null data
    for name, dataset in dataset.items():
        if (dataset.isna().any().any()):
            raise Exception("Null values found in " + name)
        
    # Remove players without a team
    dataset = removePlayersWithoutTeam(dataset)

    # Fill missing weights
    dataset = fillMissingWeights(dataset)

    # Fill missing values with 'none'
    dataset = fillWithNone(dataset)

    return dataset
    
    
