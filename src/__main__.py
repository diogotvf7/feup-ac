from modules import *
from dataset_preparation.create_final_dataset import create_final_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dataset_preparation.outliers_verification import handle_outliers
import pandas as pd
import json
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, mutual_info_classif, f_classif
from build_report import display_result


SOLUTIONS = ['IND', 'ATL', 'DET', 'SAS', 'PHO', 'SEA', 'LAS', 'WAS']

PATH = 'dataset/competition'
PATH = 'dataset/train'



CORRECT_TEAMS = SOLUTIONS if PATH == 'dataset/train' else []
DATASETS = {
    "awards_players":   f"{PATH}/train/awards_players.csv",
    "coaches":          f"{PATH}/train/coaches.csv",
    "players_teams":    f"{PATH}/train/players_teams.csv",
    "players":          f"{PATH}/train/players.csv",
    "series_post":      f"{PATH}/train/series_post.csv",
    "teams_post":       f"{PATH}/train/teams_post.csv",
    "teams":            f"{PATH}/train/teams.csv"
}
COMPETITION_DATASETS = {
    "coaches":          f"{PATH}/test/coaches.csv",
    "players_teams":    f"{PATH}/test/players_teams.csv",
    "teams":       f"{PATH}/test/teams.csv",
}
EVALUATE_YEAR = 10 if PATH == 'dataset/train' else 11
FILE = False
TEST = False
COMPETITION = True
MODELS = {
    'LogisticRegression': LogisticRegression(max_iter=1000),
    'RandomForestClassifier': RandomForestClassifier(n_estimators=100),
    'SVC': SVC(probability=True),
    'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=5)
}
FEATURE_SELECTION = [
    None,
    chi2,
    f_regression,
    mutual_info_regression,
    mutual_info_classif,
    f_classif    
]
LOOPS = 10


def main():
    # Data Load
    datasets = loadDatasets(DATASETS)
    evaluate_data = prepare_competition_data(loadDatasets(COMPETITION_DATASETS))
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
        datasets['teams_data'],
        datasets['coaches_data'],
        datasets['teams_post'], 
        datasets['teams'], 
        datasets['players_teams'],
        evaluate_data,
        target_year=EVALUATE_YEAR,
        aggregation_method='none'
    )
    print('[\033[92m✓\033[39m] Evaluate dataset creation')

    # Outliers handling
    handle_outliers('points', datasets['players_teams'])
    print('[\033[92m✓\033[39m] Outliers handling')

    # pd.set_option('display.max_rows', None)  # Display all rows
    # pd.set_option('display.max_columns', None)  # Display all columns
    # print(training_dataset)
    # print(evaluate_dataset)
    

    if FILE:
        RESULT_PATH = 'dataset/s11/delivery4/G67.csv'
        test_file = pd.read_csv(RESULT_PATH)
        print(f"The error of {RESULT_PATH} is: {calculate_error(test_file)}")
    if TEST:
        evaluate_all_models(MODELS, training_dataset, evaluate_dataset)
    if COMPETITION:
        run = {}
        guessed = 0
        total = 0
        best = {'error': 100, 'correct': 0, 'guess': [], 'result': {}}
        for fs in FEATURE_SELECTION:
            for _ in range(LOOPS):
                result, predicted, error = predict_playoff_teams(MODELS['RandomForestClassifier'], training_dataset, evaluate_dataset, CORRECT_TEAMS, fs)
                
                run['error'] = error
                run['correct'] = len(set(CORRECT_TEAMS).intersection(set(predicted)))
                run['guess'] = predicted
                run['result'] = result
                display_result(run, CORRECT_TEAMS)
                
                guessed += set(predicted) == set(CORRECT_TEAMS)
                total += 1
                if error < best['error']:
                    best = run

        # for fs in FEATURE_SELECTION:
        #     for _ in range(total):
        #         result, predicted, error = predict_playoff_teams(MODELS['SVC'], training_dataset, evaluate_dataset, fs)

        #         run['error'] = error
        #         run['correct'] = len(set(correct).intersection(set(predicted)))
        #         run['guess'] = predicted
        #         run['result'] = result
        #         display_result(run, correct)
                
        #         guessed += set(predicted) == set(correct)
        #         total += 1
        #         if error < best['error']:
        #             best = run
                
        # for fs in FEATURE_SELECTION:
        #     result, predicted, error = predict_playoff_teams(MODELS['LogisticRegression'], training_dataset, evaluate_dataset, fs)

        #     run['error'] = error
        #     run['correct'] = len(set(correct).intersection(set(predicted)))
        #     run['guess'] = predicted
        #     run['result'] = result
        #     display_result(run, correct)
            
        #     guessed += set(predicted) == set(correct)
                # total += 1
        #     if error < best['error']:
        #         best = run
                
        # for fs in FEATURE_SELECTION:
        #     result, predicted, error = predict_playoff_teams(MODELS['KNeighborsClassifier'], training_dataset, evaluate_dataset, fs)
            
        #     run['error'] = error
        #     run['correct'] = len(set(correct).intersection(set(predicted)))
        #     run['guess'] = predicted
        #     run['result'] = result
        #     display_result(run, correct)
            
        #     guessed += set(predicted) == set(correct)
                # total += 1
        #     if error < best['error']:
        #         best = run
        
        print('\n\n')
        print("________________________________________________________________________")
        print(f"Overall:\n\n         Error: {best['error']}\n     Correct %: {best['correct'] / 8 * 100}\n         Guess: {best['guess']}")
        print(f"       Correct: {CORRECT_TEAMS}")
        print("________________________________________________________________________")
        
        for i, (index, row) in enumerate(best['result'].sort_values(by='Playoff', ascending=False).iterrows()):            
            if (i + 1 <= 8): 
                print("\033[92m", end='')
            print(f"{i+1:<3}         {row['tmID']:<5} - {row['Playoff']:<6}    {'\033[92m✓\033[39m' if ((i+1 <= 8) == (row['tmID'] in CORRECT_TEAMS)) else '\033[91m✗\033[39m'}")
            print("\033[39m", end='')
            if (i + 1 == 8):
                print("_______________________________")
                
        print(total, len(FEATURE_SELECTION))
        print(f"Guessed: {guessed} / {total}")
        best['result'].to_csv(f'{PATH}/out/{str(best['error']).replace('.', '-')}.csv', index = False)

if __name__ == "__main__":
    main()