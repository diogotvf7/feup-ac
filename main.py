import pandas as pd
import dataset_preparation.linear_regression as linear_regression 
import dataset_preparation.performance_metrics as performance_metrics
from dataset_preparation.create_final_dataset import create_final_dataset


# TODO LOAD DATA
awards_players = pd.read_csv("dataset/awards_players.csv")
coaches = pd.read_csv("dataset/coaches.csv")
players_teams = pd.read_csv("dataset/players_teams.csv")
players = pd.read_csv("dataset/players.csv")
series_post = pd.read_csv("dataset/series_post.csv")
teams_post = pd.read_csv("dataset/teams_post.csv")
teams = pd.read_csv("dataset/teams.csv")

#print(coaches.head())

# ================================================================================================================================================================

# ================================================================================================================================================================
# TODO DATA PREPROCESSING
teams.drop(columns=['lgID', 'divID', 'seeded','tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB', 'arena'], inplace=True)
awards_players.drop(columns=['lgID'],  inplace = True)
coaches.drop(columns=['lgID'], inplace = True)
players_teams.drop(columns=['lgID'], inplace = True)
players.drop(columns=['firstseason', 'lastseason'], inplace = True)
series_post.drop(columns=['lgIDWinner', 'lgIDLoser'], inplace = True)
teams_post.drop(columns=['lgID'], inplace = True)

#print(teams.head())


datasets = {'teams' : teams, 'awards_players' : awards_players , 'coaches' : coaches, 
            'players_teams' : players_teams, 'players' : players, 'series_post' : series_post, 'teams_post' : teams_post}

#check for duplicated data
for name, dataset in datasets.items():
    if(dataset.duplicated().any()):
        print("Ups! Found duplicated data in " + name)


#check for null data
for name, dataset in datasets.items():
    if(dataset.isna().any().any()):
        print("Ups! Found null values in " + name)

#We have info about players that are not in any team -> ghost players perhaps we do not need them
unique_players_teams = datasets['players_teams']['playerID'].nunique()
unique_players = datasets['players']['bioID'].nunique()

print("We have: " + str(unique_players) + " players")
print("We have players information regarding their teams on " + str(unique_players_teams))
print("We have " + str(unique_players - unique_players_teams) + " without a known team")
print("That is " + str(round(((unique_players - unique_players_teams) / unique_players) * 100, 1)) + "% of unknown players' teams")



players_ = datasets['players'][datasets['players']['bioID'].isin(datasets['players_teams']['playerID'])]
print("Player after removal: " + str(players_['bioID'].count()))


#Now we only have players with weight set to 0 (all players that had height set to 0 were removed in the step before)
#Replace missing values for a players's weight and height using linear regression taking into account height and position


missing_data = players_[players_['weight'] == 0]
train_data = players_[players_['weight'] != 0]
linear_regression.replaceMissingValues(train_data, missing_data, 'weight', 'height', 'pos', players_)

datasets['players'] = players_
print(len(datasets['players'][datasets['players']['weight'] == 0]))


datasets['players'].loc[:, 'college'] = datasets['players']['college'].fillna('none')
datasets['players'].loc[:, 'collegeOther'] = datasets['players']['collegeOther'].fillna('none')

# ================================================================================================================================================================

# ================================================================================================================================================================
# TODO FEATURE ENGINEERING
players_teams, teams = performance_metrics.calculate(datasets)
datasets['players_teams'] = players_teams
datasets['teams'] = teams

# ================================================================================================================================================================

# ================================================================================================================================================================
# TODO CREATE FINAL DATASET

players_not_in_year_10 = datasets['players_teams'][datasets['players_teams']['year'] != 10]
players_not_in_year_10.to_csv('dataset/finals/players_final.csv', index=False)

create_final_dataset(datasets['teams_post'], datasets['teams'])

# ================================================================================================================================================================



