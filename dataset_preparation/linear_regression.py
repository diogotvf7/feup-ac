from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def replaceMissingValues(train_data, missing_data, target, feat1, feat2, original_dataset):

    X_train = pd.get_dummies(train_data[[feat1, feat2]], drop_first=True)
    y_train = train_data[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    X_missing = pd.get_dummies(missing_data[[feat1, feat2]], drop_first=True)
    X_missing = X_missing.reindex(columns=X_train.columns, fill_value=0)

    predicted_values = np.round(model.predict(X_missing)).astype(int)
    
    original_dataset.loc[original_dataset[target] == 0, target] = predicted_values
