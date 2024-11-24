from modules import *
from dataset_preparation.create_final_dataset import create_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dataset_preparation.outliers_verification import handle_outliers

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

DATASETS = {
    "awards_players":   "dataset/awards_players.csv",
    "coaches":          "dataset/coaches.csv",
    "players_teams":    "dataset/players_teams.csv",
    "players":          "dataset/players.csv",
    "series_post":      "dataset/series_post.csv",
    "teams_post":       "dataset/teams_post.csv",
    "teams":            "dataset/teams.csv"
}

TRAINING_YEARS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
EVALUATE_YEAR = 10

MODELS = [
    LogisticRegression(max_iter=1000),
    RandomForestClassifier(n_estimators=100),
    SVC(probability=True),
    KNeighborsClassifier(n_neighbors=5)
]

def main():
    # Data Load
    datasets = loadDatasets(DATASETS)

    # Data Preparation
    datasets = dataPreparation(datasets)

    # Modelling
    datasets = feature_engineering(datasets)

    training_dataset = datasets['training_dataset']
    # training_dataset.to_csv('dataset/finals/training.csv', index = False)
    
    #Pass teams_post and teams to know how many times a team went to the playoffs
    evaluate_dataset = create_final_dataset(datasets['teams_post'], 
                                            datasets['teams'], 
                                            datasets['players_teams'][datasets['players_teams']['year'] != 10])

    handle_outliers('points', datasets['players_teams'] )


    for model in MODELS:
        print(f'FOR MODEL : {model}\n')
        for _ in range(1, 20):
            evaluate(model, training_dataset, evaluate_dataset)  


if __name__ == "__main__":
    main()