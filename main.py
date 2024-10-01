import pandas
import math


awards_players = pandas.read_csv("dataset/awards_players.csv")
coaches = pandas.read_csv("dataset/coaches.csv")
players_teams = pandas.read_csv("dataset/players_teams.csv")
players = pandas.read_csv("dataset/players.csv")
series_post = pandas.read_csv("dataset/series_post.csv")
teams_post = pandas.read_csv("dataset/teams_post.csv")
teams = pandas.read_csv("dataset/teams.csv")


#print(coaches.head())

teams.drop(columns=['lgID', 'divID', 'seeded','tmORB','tmDRB','tmTRB','opptmORB','opptmDRB','opptmTRB', 'arena'], inplace=True)
awards_players.drop(columns=['lgID'],  inplace = True)
coaches.drop(columns=['lgID'], inplace = True)
players_teams.drop(columns=['lgID'], inplace = True)
players.drop(columns=['firstseason', 'lastseason'], inplace = True)
series_post.drop(columns=['lgIDWinner', 'lgIDLoser'], inplace = True)
teams_post.drop(columns=['lgID'], inplace = True)

#print(teams.head())


