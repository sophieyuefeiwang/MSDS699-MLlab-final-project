import streamlit as st
import pandas as pd
import numpy as np
import pickle

data = pd.read_csv('data/cleaned_churn.csv') # if churn then 1, else 0

data = data.drop(columns='Unnamed: 0')

df = data.copy()

target='churn'
encode = ['Gender', 'Education_Level',
        'Marital_Status', 'Income_Category', 'Card_Category']

for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]

# Separating X and y
X = df.drop('churn', axis=1)
Y = df['churn']

# Build random forest model
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(subsample=0.8,
                            n_estimators=120,
                            min_weight_fraction_leaf=0,
                            min_samples_split=10,
                            max_depth=6,
                            loss='deviance')
clf.fit(X, Y)

# Saving the model
pickle.dump(clf, open('churn_clf.pkl', 'wb'))