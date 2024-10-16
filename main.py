import pandas as pd
import linearRegression 

awards_players = pd.read_csv("dataset/awards_players.csv")
coaches = pd.read_csv("dataset/coaches.csv")
players_teams = pd.read_csv("dataset/players_teams.csv")
players = pd.read_csv("dataset/players.csv")
series_post = pd.read_csv("dataset/series_post.csv")
teams_post = pd.read_csv("dataset/teams_post.csv")
teams = pd.read_csv("dataset/teams.csv")

#print(coaches.head())

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
linearRegression.replaceMissingValues(train_data, missing_data, 'weight', 'height', 'pos', players_)

datasets['players'] = players_
print(len(datasets['players'][datasets['players']['weight'] == 0]))


datasets['players']['college'].fillna('none', inplace=True)
datasets['players']['collegeOther'].fillna('none', inplace=True)


#check for null data
""" for name, dataset in datasets.items():
    null_rows = dataset[dataset.isna().any(axis=1)]  # Get rows with any NaN values
    if not null_rows.empty:
        print(f"Ups! Found null values in {name}")
        print(null_rows) """

