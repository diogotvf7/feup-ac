import pandas as pd

def loadLoad(paths):
    datasets = {}
    for path in paths:
        name = path.split('/')[-1].split('.')[0]
        datasets[name] = pd.read_csv(path)

    return datasets