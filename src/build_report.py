from git import Repo
from datetime import datetime

REPO_PATH = '~/Documents/ac/feup-ac/'
JSON_DATA = {
        "LogisticRegression": {
                "default": {
                        "predicted_teams": [
                                "DET",
                                "IND",
                                "PHO",
                                "SAS",
                                "SEA",
                                "MIN",
                                "LAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[chi2]": {
                        "predicted_teams": [
                                "DET",
                                "IND",
                                "PHO",
                                "SAS",
                                "SEA",
                                "MIN",
                                "LAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[f_regression]": {
                        "predicted_teams": [
                                "DET",
                                "SAC",
                                "LAS",
                                "IND",
                                "PHO",
                                "SEA",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[mutual_info_regression]": {
                        "predicted_teams": [
                                "DET",
                                "SAC",
                                "LAS",
                                "IND",
                                "PHO",
                                "SEA",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[mutual_info_classif]": {
                        "predicted_teams": [
                                "DET",
                                "SAC",
                                "LAS",
                                "IND",
                                "PHO",
                                "SEA",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[f_classif]": {
                        "predicted_teams": [
                                "DET",
                                "SAC",
                                "LAS",
                                "IND",
                                "PHO",
                                "SEA",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                }
        },
        "RandomForestClassifier": {
                "default": {
                        "predicted_teams": [
                                "IND",
                                "SEA",
                                "ATL",
                                "SAS",
                                "LAS",
                                "DET",
                                "WAS",
                                "PHO"
                        ],
                        "precision": 1.0
                },
                "feature_selection[chi2]": {
                        "predicted_teams": [
                                "ATL",
                                "NYL",
                                "LAS",
                                "WAS",
                                "IND",
                                "SEA",
                                "SAS",
                                "DET"
                        ],
                        "precision": 0.875
                },
                "feature_selection[f_regression]": {
                        "predicted_teams": [
                                "ATL",
                                "SEA",
                                "IND",
                                "SAS",
                                "LAS",
                                "PHO",
                                "MIN",
                                "DET"
                        ],
                        "precision": 0.875
                },
                "feature_selection[mutual_info_regression]": {
                        "predicted_teams": [
                                "IND",
                                "ATL",
                                "PHO",
                                "MIN",
                                "SEA",
                                "DET",
                                "LAS",
                                "SAS"
                        ],
                        "precision": 0.875
                },
                "feature_selection[mutual_info_classif]": {
                        "predicted_teams": [
                                "IND",
                                "ATL",
                                "LAS",
                                "SEA",
                                "PHO",
                                "DET",
                                "SAS",
                                "WAS"
                        ],
                        "precision": 1.0
                },
                "feature_selection[f_classif]": {
                        "predicted_teams": [
                                "IND",
                                "ATL",
                                "SEA",
                                "LAS",
                                "SAS",
                                "PHO",
                                "DET",
                                "WAS"
                        ],
                        "precision": 1.0
                }
        },
        "SVC": {
                "default": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SEA",
                                "PHO",
                                "LAS",
                                "SAC",
                                "NYL",
                                "SAS"
                        ],
                        "precision": 0.75
                },
                "feature_selection[chi2]": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SEA",
                                "PHO",
                                "LAS",
                                "SAC",
                                "NYL",
                                "SAS"
                        ],
                        "precision": 0.75
                },
                "feature_selection[f_regression]": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SEA",
                                "PHO",
                                "LAS",
                                "SAC",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[mutual_info_regression]": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SEA",
                                "PHO",
                                "LAS",
                                "SAC",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[mutual_info_classif]": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SAC",
                                "PHO",
                                "LAS",
                                "SEA",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                },
                "feature_selection[f_classif]": {
                        "predicted_teams": [
                                "IND",
                                "DET",
                                "SEA",
                                "PHO",
                                "LAS",
                                "SAC",
                                "SAS",
                                "NYL"
                        ],
                        "precision": 0.75
                }
        },
        "KNeighborsClassifier": {
                "default": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                },
                "feature_selection[chi2]": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                },
                "feature_selection[f_regression]": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                },
                "feature_selection[mutual_info_regression]": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                },
                "feature_selection[mutual_info_classif]": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                },
                "feature_selection[f_classif]": {
                        "predicted_teams": [
                                "ATL",
                                "CHI",
                                "CON",
                                "DET",
                                "IND",
                                "LAS",
                                "MIN",
                                "NYL"
                        ],
                        "precision": 0.5
                }
        }
}

def retrieve_best_model_performances(json_data):
    predicted_teams = []
    max_precision = 0
    best_performances = {}

    for model, results in json_data.items():
        for fs, result in results.items():
            if result['precision'] == max_precision:
                best_performances[model].append(fs)
            elif result['precision'] > max_precision:
                best_performances = {model: [fs]}
                predicted_teams = result['predicted_teams']
                max_precision = result['precision']
    return max_precision, predicted_teams, best_performances

def write(file, *content, newline="\n", separator=" "):
    content = map(str, content)
    file.write(separator.join(content) + newline)

def build_report(json_data):
    repo = Repo(REPO_PATH)
    commit = repo.head.commit
    branch = repo.active_branch

    now = datetime.now()

    with open(f"reports/{now.strftime("%Y-%m-%d-%H:%M:%S")}.md", 'x') as f:
        write(f, "# ⁍ Log Information")
        write(f, f"- `Branch`: {branch}")
        write(f, f"- `Commit`")
        write(f, "    - **Message**:", commit.message)
        write(f, "    - **Hash**:", commit.hexsha)
        write(f, "    - **Author**:", commit.author)
        write(f, "- `Date`:", now.strftime("%Y-%m-%d at %H:%M"))

        write(f, "\n<br />\n")
        write(f, "# ⁍ Summary")
        
        max_precision, predicted_teams, best_performances = retrieve_best_model_performances(json_data)
        write(f, "- `Max precision achieved:`", max_precision)
        write(f, "- `Predicted teams:`", ', '.join(predicted_teams))

        for model, fs in best_performances.items():
            write(f, "###", model)
            for feature_selection in fs:
                write(f, "- ", feature_selection)
        
        write(f, "\n<br />\n")
        write(f, "# ⁍ Full Report")

        for model, results in json_data.items(): # iterate models

            write(f, "##", model)

            for fs, result in results.items():
                write(f, "###", fs)
                write(f, "- `Precision:`", result['precision'])
                write(f, "- `Error:`", result['error'])
                write(f, "- `Predicted teams:`", ', '.join(result['predicted_teams']))

if __name__ == "__main__":
    build_report(JSON_DATA)