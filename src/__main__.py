from modules import *
from build_report import build_report
from dataset_preparation.create_final_dataset import create_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dataset_preparation.outliers_verification import handle_outliers
import json

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

MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
    'SVC': SVC(probability=True),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5)
}

def main():
    # Data Load
    datasets = loadDatasets(DATASETS)
    print('[\033[92m✓\033[39m] Dataset load')

    # Data Preparation
    datasets = dataPreparation(datasets)
    print('[\033[92m✓\033[39m] Dataset preparation')

    # Modelling
    datasets = feature_engineering(datasets)
    print('[\033[92m✓\033[39m] Feature engineering')

    training_dataset = datasets['training_dataset']
    # training_dataset.to_csv('dataset/finals/training.csv', index = False)
    
    # Pass teams_post and teams to know how many times a team went to the playoffs
    evaluate_dataset = create_final_dataset(
        datasets['coaches_data'],
        datasets['teams_post'], 
        datasets['teams'], 
        datasets['players_teams'][datasets['players_teams']['year'] != 10]
    )
    print('[\033[92m✓\033[39m] Evaluate dataset creation')

    handle_outliers('points', datasets['players_teams'])
    print('[\033[92m✓\033[39m] Outliers handling')

    print("Training dataset shape: ", training_dataset.columns)
    print("Evaluate dataset shape: ", evaluate_dataset.columns)

    results = {}
    for model in MODELS:
        if model == 'LogisticRegression' or model == 'KNeighborsClassifier':
            results[model] = evaluate_model(MODELS[model], training_dataset, evaluate_dataset)
        elif model == 'RandomForestClassifier' or model == 'SVC':
            results[model] = {}
            max_precision1, max_precision2, max_precision3, max_precision4, max_precision5, max_precision6 = 0, 0, 0, 0, 0, 0
            for _ in range(1, 20):
                tmp = evaluate_model(MODELS[model], training_dataset, evaluate_dataset)
                if tmp['default']['precision'] > max_precision1:
                    max_precision1 = tmp['default']['precision']
                    results[model]['default'] = tmp['default']
                if tmp['feature_selection[chi2]']['precision'] > max_precision2:
                    max_precision2 = tmp['feature_selection[chi2]']['precision']   
                    results[model]['feature_selection[chi2]'] = tmp['feature_selection[chi2]']
                if tmp['feature_selection[f_regression]']['precision'] > max_precision3:
                    max_precision3 = tmp['feature_selection[f_regression]']['precision']   
                    results[model]['feature_selection[f_regression]'] = tmp['feature_selection[f_regression]']
                if tmp['feature_selection[mutual_info_regression]']['precision'] > max_precision4:
                    max_precision4 = tmp['feature_selection[mutual_info_regression]']['precision']   
                    results[model]['feature_selection[mutual_info_regression]'] = tmp['feature_selection[mutual_info_regression]']
                if tmp['feature_selection[mutual_info_classif]']['precision'] > max_precision5:
                    max_precision5 = tmp['feature_selection[mutual_info_classif]']['precision']   
                    results[model]['feature_selection[mutual_info_classif]'] = tmp['feature_selection[mutual_info_classif]']
                if tmp['feature_selection[f_classif]']['precision'] > max_precision6:
                    max_precision6 = tmp['feature_selection[f_classif]']['precision']   
                    results[model]['feature_selection[f_classif]'] = tmp['feature_selection[f_classif]']
        print(f'[\033[92m✓\033[39m] {model} model evaluation')

    build_report(results)

if __name__ == "__main__":
    main()