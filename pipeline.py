from modules import *
from dataset_preparation.create_final_dataset import create_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# 1. Load data
#
# 2. Preprocess data
#   - Manage missing values
#
# 3. Feature engineering
#   - Deduct new features
#   - 
#
# 4. Train model
#   - Split data
#   - Train model
#
# 5. Evaluate model
#   - Evaluate model
#   - Save model

DATASETS = [
    "dataset/awards_players.csv",
    "dataset/coaches.csv",
    "dataset/players_teams.csv",
    "dataset/players.csv",
    "dataset/series_post.csv",
    "dataset/teams_post.csv",
    "dataset/teams.csv"
]

models = [
    # LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100),
    # SVC(probability=True),
    # KNeighborsClassifier(n_neighbors=5)
]

def main():
    # Data Load
    datasets = loadLoad(DATASETS)

    # Data Preparation
    datasets = dataPreparation(datasets)

    # Modelling
    datasets = modelling(datasets)

    training_dataset = datasets['training_dataset']
    training_dataset.to_csv('dataset/finals/training.csv', index = False)
    
    #Pass teams_post and teams to know how many times a team went to the playoffs
    evaluate_dataset = create_final_dataset(datasets['teams_post'], datasets['teams'], datasets['players_teams'][datasets['players_teams']['year'] != 10])

    for model in models:
        evaluate(model, training_dataset, evaluate_dataset)


if __name__ == "__main__":
    main()