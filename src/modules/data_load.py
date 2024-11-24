import pandas as pd


def loadDatasets(datasets):
    return {name: pd.read_csv(datasets[name]) for name in datasets}