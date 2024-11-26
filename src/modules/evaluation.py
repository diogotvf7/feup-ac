import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectKBest, chi2, f_regression, mutual_info_regression, mutual_info_classif, f_classif

CORRECT_TEAMS = ['IND', 'ATL', 'DET', 'SAS', 'PHO', 'SEA', 'LAS', 'WAS']

def calculate_precision(predicted_teams):
    correct_predictions = len(set(predicted_teams) & set(CORRECT_TEAMS))
    precision = correct_predictions / len(predicted_teams)
    return precision

def feature_selection(features, target, score_function, k=10):
    select_k_best = SelectKBest(score_func=score_function, k=k)
    _ = select_k_best.fit_transform(features, target)
    selected_features = features.columns[select_k_best.get_support()].tolist()
    return selected_features

def create_confusion_matrix(teams, top_8_teams, model, score_function):
    true_positives = len(set(top_8_teams) & set(CORRECT_TEAMS))
    false_positives = len(set(top_8_teams) - set(CORRECT_TEAMS))
    false_negatives = len(set(CORRECT_TEAMS) - set(top_8_teams))
    true_negatives = len(set(teams) - set(top_8_teams) - set(CORRECT_TEAMS))

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

def calculate_error(evaluate_data):
    data = evaluate_data.copy()
    data['playoff_probability'] = data['playoff_probability'] * 100
    data['playoff_probability'] = data['playoff_probability'] * 8 / data['playoff_probability'].sum()

    error = 0
    # iterate data variable
    for row in data.iterrows():
        is_in_playoffs = row[1]['tmID'] in CORRECT_TEAMS
        error += abs(is_in_playoffs - row[1]['playoff_probability'])  
        
    return error

    # return CORRECT_TEAMS.map(
        # lambda team: abs(data.loc[data['tmID'] == team, 'playoff_probability'].values[0] - 1)
    # )

def evaluate(model, training_dataset, evaluate_dataset, score_function=None):
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

    confusion_matrix = create_confusion_matrix(evaluate_data_copy['tmID'].tolist(), top_8_teams['tmID'].tolist(), model, score_function)

    predicted_teams = top_8_teams['tmID'].tolist()

    precision = calculate_precision(predicted_teams)
    # print(f"Prediction Precision: {precision:.2f}")

    return {
        'predicted_teams': predicted_teams, 
        'precision': precision,
        'error': error,
        'confusion_matrix': confusion_matrix
    }

def evaluate_model(model, training_dataset, evaluate_dataset):
    default = evaluate(model, training_dataset, evaluate_dataset)
    feature_selection_chi2 = evaluate(model, training_dataset, evaluate_dataset, chi2)
    feature_selection_f_regression = evaluate(model, training_dataset, evaluate_dataset, f_regression)
    feature_selection_mutual_info_regression = evaluate(model, training_dataset, evaluate_dataset, mutual_info_regression)
    feature_selection_mutual_info_classif = evaluate(model, training_dataset, evaluate_dataset, mutual_info_classif)
    feature_selection_f_classif = evaluate(model, training_dataset, evaluate_dataset, f_classif)
    
    return {
        'default': default, 
        'chi2': feature_selection_chi2,
        'f_regression': feature_selection_f_regression,
        'mutual_info_regression': feature_selection_mutual_info_regression,
        'mutual_info_classif': feature_selection_mutual_info_classif,
        'f_classif': feature_selection_f_classif,
    }