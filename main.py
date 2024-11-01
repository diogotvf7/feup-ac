import pandas as pd
import dataset_preparation.linear_regression as linear_regression 
import dataset_preparation.performance_metrics as performance_metrics
from dataset_preparation.outliers_verification import process_outliers
from feature_selection.implementations import select_features, select_spearman
from sklearn.feature_selection import chi2, mutual_info_classif, f_classif, f_regression, RFE
from dataset_preparation.merge_datasets import process_data

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
linear_regression.replaceMissingValues(train_data, missing_data, 'weight', 'height', 'pos', players_)

datasets['players'] = players_
print(len(datasets['players'][datasets['players']['weight'] == 0]))


datasets['players'].loc[:, 'college'] = datasets['players']['college'].fillna('none')
datasets['players'].loc[:, 'collegeOther'] = datasets['players']['collegeOther'].fillna('none')


players_teams, teams = performance_metrics.calculate(datasets)
datasets['players_teams'] = players_teams
datasets['teams'] = teams


#Merge all datasets into one
final_dataset = process_data(datasets)

correlation_matrix = final_dataset.corr()

""" plt.figure(figsize=(19, 16))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, 
            cbar_kws={"shrink": .5}, linewidths=0.5, linecolor='black', annot_kws={"size": 10})
plt.title('Correlation Matrix', fontsize=20)
plt.xticks(fontsize=14, rotation=45) 
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show() """

#By analysing the correlation matrix we can see some attributes are highly correlated with eachother, the yare even redundant -> win is highly correlated with homeW , awayW, confW etc
#This might be a problem of overfitting as later on in feature selectin these attributes will all be chosen because wins is one of the most important ones
#Lets drop some of them
final_dataset = final_dataset.drop(columns=['homeW', 'confW', 'awayW', 'confL', 'homeL', 'awayL'])
final_dataset.to_csv('dataset/final_dataset.csv', index=False)

features = final_dataset.drop(columns=['playoff'])
target = final_dataset['playoff']

## Feature Selection
""" print("Selected features from chi2:", select_features(features, target, chi2))  
print("Selected features from mutual information:", select_features(features, target, mutual_info_classif))  
print("Selected features from anova:", select_features(features, target, f_classif))  
print("Selected features from pearson:", select_features(features, target, f_regression))  
print("Selected features from RFE:", select_features(features, target, RFE))  
print("Selected features from spearman:", select_spearman(features, target))   """

selected_features = select_features(features, target, mutual_info_classif)

# Not sure if we should do this here
process_outliers(selected_features, final_dataset)


