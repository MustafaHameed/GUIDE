import pandas as pd

def load_data(csv_path: str):
    """Load dataset and create binary target indicating pass/fail."""
    df = pd.read_csv(csv_path)
    df['pass'] = (df['G3'] >= 10).astype(int)
    X = df.drop(columns=['G3', 'pass'])
    y = df['pass']
    return X, y
