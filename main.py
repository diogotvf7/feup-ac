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


#We have info about players that are not in any team -> ghost players perhaps we do not need them
unique_players_teams = players_teams['playerID'].nunique()
unique_players = players['bioID'].nunique()

print("We have: " + str(unique_players) + " players")
print("We have players information regarding their teams on " + str(unique_players_teams))
print("We have " + str(unique_players - unique_players_teams) + " without a known team")
print("That is " + str(round(((unique_players - unique_players_teams) / unique_players) * 100, 1)) + "% of unknown players' teams")

players = players[players['bioID'].isin(players_teams['playerID'])]
print("Player after removal: " + str(players['bioID'].count()))


#Now we only have players with weight set to 0 (all players that had height set to 0 were removed in the step before)
#Replace missing values for a players's weight and height using linear regression taking into account height and position


missing_data = players[players['weight'] == 0]
train_data = players[players['weight'] != 0]
linearRegression.replaceMissingValues(train_data, missing_data, 'weight', 'height', 'pos', players)

print(len(players[players['weight'] == 0]))



