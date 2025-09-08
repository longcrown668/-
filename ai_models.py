import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

def extract_features(data, feature_cols, horizon):
    X = []
    y = []
    for i in range(len(data) - horizon):
        X.append(data[feature_cols].iloc[i:i+horizon].values.flatten())
        y.append(int(data['Target'].iloc[i+horizon]))
    return np.array(X), np.array(y)

def train_voting_rf_lr(X, y):
    rf = RandomForestClassifier(n_estimators=50, random_state=42)
    lr = LogisticRegression(max_iter=200)
    clf = VotingClassifier(estimators=[('rf', rf), ('lr', lr)], voting='soft')
    clf.fit(X, y)
    return clf

def train_mlp(X, y):
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    mlp.fit(X, y)
    return mlp