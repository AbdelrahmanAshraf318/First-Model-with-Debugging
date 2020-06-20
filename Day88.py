# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 14:32:54 2020

@author: Abd Elrahman
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataSet = pd.read_csv('50_Startups.csv')

dataSet.describe()

#define columns
dataSet.columns

#Selecting the predict feature for y and Features for X
y = dataSet.Profit

Features = ['R&D Spend' , 'Administration' , 'Marketing Spend']
X = dataSet[Features]
X.columns
X.describe()
X.head()
########################################################
#Model Selection
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=0)

#Define Missing Values
from sklearn.impute import SimpleImputer

# Imputation
my_imputer = SimpleImputer()
imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
imputed_X_valid = pd.DataFrame(my_imputer.transform(X_valid))

# Imputation removed column names; put them back
imputed_X_train.columns = X_train.columns
imputed_X_valid.columns = X_valid.columns

##############################################################

#Fit the model and predict 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor(random_state=1)
forest_model.fit(X_train , y_train)

##########################################################
#Make it more accurate by Permutation Importance
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(forest_model, random_state=1).fit(X_valid, y_valid)
eli5.show_weights(perm, feature_names = X_valid.columns.tolist())

#########################################################
#Make it more accurate by Partial Plots

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=forest_model, dataset=X_valid, model_features=Features, feature='Administration')

# plot it
pdp.pdp_plot(pdp_goals, 'Administration')
plt.show()























