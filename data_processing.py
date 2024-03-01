from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np


def load_quality_data():
    wine_quality = fetch_ucirepo(id=186)

    X = pd.DataFrame(wine_quality.data.features)
    y = pd.DataFrame(wine_quality.data.targets)

    y['quality'] = y['quality'].apply(lambda x: -1 if x <= 5 else 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 58
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return X_train, X_test, y_train, y_test


def load_quality_mean_data():
    wine_quality = fetch_ucirepo(id=186)

    X = pd.DataFrame(wine_quality.data.features)
    y = pd.DataFrame(wine_quality.data.targets)

    mean = round(sum(y['quality']) / len(y['quality']))
    y['quality'] = y['quality'].apply(lambda x: -1 if x <= mean else 1)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #42 58
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return X_train, X_test, y_train, y_test


def load_color_data():
    df = pd.read_csv('wine_color.csv')
    X = df.drop(columns=['color', 'quality'])
    y = df['color']

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # 42 58
    y_train = np.array(y_train)
    X_train = np.array(X_train)
    y_test = np.array(y_test)
    X_test = np.array(X_test)

    y_train = np.squeeze(y_train)
    y_test = np.squeeze(y_test)

    return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    X_train, X_test, y_train, y_test = load_quality_data()
