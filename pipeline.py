import pandas as pd

from modules import *
from dataset_preparation.create_final_dataset import create_final_dataset

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

def main():
    # Data Load
    datasets = loadLoad(DATASETS)

    # Data Preparation
    datasets = dataPreparation(datasets)

    # Modelling
    datasets = modelling(datasets)

    training_dataset = datasets['training_dataset']
    training_dataset.to_csv('dataset/finals/training.csv', index = False)
    players_not_in_year_10 = datasets['players_teams'][datasets['players_teams']['year'] != 10]
    players_not_in_year_10.to_csv('dataset/finals/players_final.csv', index=False)

    create_final_dataset(datasets['teams_post'], datasets['teams'])


if __name__ == "__main__":
    main()