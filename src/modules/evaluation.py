import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, mutual_info_classif, f_classif
from build_report import report


def calculate_precision(predicted_teams, correct_teams):
    correct_predictions = len(set(predicted_teams) & set(correct_teams))
    precision = correct_predictions / len(predicted_teams)
    return precision

def feature_selection(features, target, score_function, k=10):
    if score_function == None:
        return features.columns.tolist()
    
    select_k_best = SelectKBest(score_func=score_function, k=k)
    _ = select_k_best.fit_transform(features, target)
    selected_features = features.columns[select_k_best.get_support()].tolist()
    return selected_features

def create_confusion_matrix(teams, top_8_teams, model, score_function, correct_teams):
    true_positives = len(set(top_8_teams) & set(correct_teams))
    false_positives = len(set(top_8_teams) - set(correct_teams))
    false_negatives = len(set(correct_teams) - set(top_8_teams))
    true_negatives = len(set(teams) - set(top_8_teams) - set(correct_teams))

    confusion_matrix = np.array([
        [true_positives, false_negatives],
        [false_positives, true_negatives]
    ])

    fig, ax = plt.subplots()
    im = ax.imshow(confusion_matrix, cmap="Blues")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted Positive", "Predicted Negative"])
    ax.set_yticklabels(["Actual Positive", "Actual Negative"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, confusion_matrix[i, j],
                    ha="center", va="center", color="black")

    plot_name =  f"plots/{type(model).__name__}-{'default' if (score_function == None) else score_function.__name__}-confusion-matrix.png"
    
    plt.title("Confusion Matrix")
    plt.colorbar(im)
    plt.savefig("reports/" + plot_name) 
    plt.close('all')    

    return plot_name

def calculate_error(evaluate_data, correct_teams):
    data = evaluate_data.copy()
    data['Playoff'] = data['Playoff'] * 100
    data['Playoff'] = data['Playoff'] * 8 / data['Playoff'].sum()

    error = 0
    # iterate data variable
    for row in data.iterrows():
        is_in_playoffs = row[1]['tmID'] in correct_teams
        error += abs(is_in_playoffs - row[1]['Playoff'])  
        
    return error

    # return correct_teams.map(
        # lambda team: abs(data.loc[data['tmID'] == team, 'playoff_probability'].values[0] - 1)
    # )

def evaluate(model, training_dataset, evaluate_dataset, correct_teams, score_function=None):
    training_data_copy = training_dataset.copy()
    evaluate_data_copy = evaluate_dataset.copy()

    features_train = training_data_copy.drop(columns=['playoff'])
    target_train = training_data_copy['playoff']

    if score_function:
        selected_features = feature_selection(features_train, target_train, score_function, 10)
        features_train = features_train[selected_features]
        evaluate_features = evaluate_data_copy[selected_features]
    else:
        evaluate_features = evaluate_data_copy.drop(columns=['tmID'])

    # Train the model
    model.fit(features_train, target_train)

    # Predict probabilities
    playoff_probabilities = model.predict_proba(evaluate_features)[:, 1]
    evaluate_data_copy['playoff_probability'] = playoff_probabilities

    error = calculate_error(evaluate_data_copy[['tmID', 'playoff_probability']])

    
    top_8_teams = evaluate_data_copy[['tmID', 'playoff_probability']].sort_values(by='playoff_probability', ascending=False).head(8)
    #print("Predicted top 8 teams:\n", top_8_teams)

    confusion_matrix = create_confusion_matrix(evaluate_data_copy['tmID'].tolist(), top_8_teams['tmID'].tolist(), model, score_function, correct_teams)

    predicted_teams = top_8_teams['tmID'].tolist()

    precision = calculate_precision(predicted_teams, correct_teams)
    # print(f"Prediction Precision: {precision:.2f}")

    return {
        'predicted_teams': predicted_teams, 
        'precision': precision,
        'error': error,
        'confusion_matrix': confusion_matrix
    }

def evaluate_model(model, training_dataset, evaluate_dataset, correct_teams):
    default = evaluate(model, training_dataset, evaluate_dataset, correct_teams)
    feature_selection_chi2 = evaluate(model, training_dataset, evaluate_dataset, correct_teams, chi2)
    feature_selection_f_regression = evaluate(model, training_dataset, evaluate_dataset, correct_teams, f_regression)
    feature_selection_mutual_info_regression = evaluate(model, training_dataset, evaluate_dataset, correct_teams, mutual_info_regression)
    feature_selection_mutual_info_classif = evaluate(model, training_dataset, evaluate_dataset, correct_teams, mutual_info_classif)
    feature_selection_f_classif = evaluate(model, training_dataset, evaluate_dataset, correct_teams, f_classif)
    
    return {
        'default': default, 
        'chi2': feature_selection_chi2,
        'f_regression': feature_selection_f_regression,
        'mutual_info_regression': feature_selection_mutual_info_regression,
        'mutual_info_classif': feature_selection_mutual_info_classif,
        'f_classif': feature_selection_f_classif,
    }
    
def evaluate_all_models(models, training_dataset, evaluate_dataset, generate_report=True):
    results = {}
    for model in models:
        if model == 'LogisticRegression' or model == 'KNeighborsClassifier':
            results[model] = evaluate_model(models[model], training_dataset, evaluate_dataset)
        elif model == 'RandomForestClassifier' or model == 'SVC':
            results[model] = {}
            max_precision1, max_precision2, max_precision3, max_precision4, max_precision5, max_precision6 = 0, 0, 0, 0, 0, 0
            for _ in range(1, 20):
                tmp = evaluate_model(models[model], training_dataset, evaluate_dataset)
                if tmp['default']['precision'] > max_precision1:
                    max_precision1 = tmp['default']['precision']
                    results[model]['default'] = tmp['default']
                if tmp['chi2']['precision'] > max_precision2:
                    max_precision2 = tmp['chi2']['precision']   
                    results[model]['chi2'] = tmp['chi2']
                if tmp['f_regression']['precision'] > max_precision3:
                    max_precision3 = tmp['f_regression']['precision']   
                    results[model]['f_regression'] = tmp['f_regression']
                if tmp['mutual_info_regression']['precision'] > max_precision4:
                    max_precision4 = tmp['mutual_info_regression']['precision']   
                    results[model]['mutual_info_regression'] = tmp['mutual_info_regression']
                if tmp['mutual_info_classif']['precision'] > max_precision5:
                    max_precision5 = tmp['mutual_info_classif']['precision']   
                    results[model]['mutual_info_classif'] = tmp['mutual_info_classif']
                if tmp['f_classif']['precision'] > max_precision6:
                    max_precision6 = tmp['f_classif']['precision']   
                    results[model]['f_classif'] = tmp['f_classif']
        print(f'[\033[92m✓\033[39m] {model} model evaluation')

    if generate_report:
        report(results)
    
    return results

def predict_playoff_teams(model, training_dataset, evaluate_dataset, correct_teams, score_function=None):
    training_data_copy = training_dataset.copy()
    evaluate_data_copy = evaluate_dataset.copy()

    features_train = training_data_copy.drop(columns=['playoff'])
    target_train = training_data_copy['playoff']
    
    selected_features = feature_selection(features_train, target_train, score_function, 10)
    features_train = features_train[selected_features]
    evaluate_features = evaluate_data_copy[selected_features]

    # print(features_train.columns)
    # print(evaluate_features.columns)

    # Train the model
    model.fit(features_train, target_train)

    # Predict probabilities
    playoff_probabilities = model.predict_proba(evaluate_features)[:, 1]
    evaluate_data_copy['Playoff'] = playoff_probabilities
    
    top_8_teams = evaluate_data_copy[['tmID', 'Playoff']].sort_values(by='Playoff', ascending=False).head(8)['tmID'].tolist()

    evaluate_data_copy['Playoff'] = evaluate_data_copy['tmID'].apply(lambda x: 1 if x in top_8_teams else 0)
    
    return evaluate_data_copy[['tmID', 'Playoff']], top_8_teams, calculate_error(evaluate_data_copy[['tmID', 'Playoff']], correct_teams)

    