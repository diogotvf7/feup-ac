import matplotlib.pyplot as plt
import seaborn as sns


def handle_outliers(attribute, dataset):
    outliers = detect_outlier_values(dataset[attribute])
    median_value = dataset[attribute][~dataset[attribute].isin(outliers)].median()
    dataset.loc[outliers.index, attribute] = median_value
    
"""     plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataset[attribute])  # Ensure this uses the modified dataset
    plt.title(f'Box Plot of {attribute} (Outliers Handled)')
    plt.xlabel(attribute)
    plt.show() """

def detect_outlier_values(data):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data < lower_bound) | (data > upper_bound)]

def process_outliers(attributes, dataset):
    for attribute in attributes:
        handle_outliers(attribute, dataset)