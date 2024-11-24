import matplotlib.pyplot as plt
import seaborn as sns


def handle_outliers(attribute, dataset):
    outliers = detect_outlier_values(dataset[attribute])
    
    print(f"Outliers for {attribute}:")
    print(outliers)
    print(f"Number of outliers in {attribute}: {len(outliers)}\n")
    
    # Create the box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=dataset[attribute])
    
    # Highlight the outliers in red
    for outlier in outliers:
        plt.scatter(outlier, 0, color='red', label='Outliers' if 'Outliers' not in plt.gca().get_legend_handles_labels()[1] else "")
    
    # Add a legend for clarity
    plt.legend(loc='upper right')
    plt.title(f'Box Plot of {attribute} (Outliers Highlighted)')
    plt.xlabel(attribute)
    # plt.show() 

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