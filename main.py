import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
insurance = pd.read_csv(r"C:\Users\vites.LAPTOP-CL7RLAUQ\OneDrive\Insurance_prediction\insurance.csv")
print(insurance)


insurance['sex'] = insurance['sex'].map(lambda s :1  if s == 'female' else 0)
insurance['smoker'] = insurance['smoker'].map(lambda s :1  if s == 'yes' else 0)
labelencoder = LabelEncoder()
insurance['region'] = labelencoder.fit_transform(insurance['region'])

print(insurance.head())

X = insurance.drop(['expenses'], axis = 1)
y = insurance.expenses
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.ensemble import RandomForestRegressor  # Import Random Forest Regression model

n_estimators = [100, 500, 1000, 1500]
max_features = ['auto', 'sqrt']
max_depth = [2,3,4,5,6]
max_depth.append(None)
#min_samples_split = [2, 5, 10]
#min_samples_leaf = [1, 2, 4, 10]


params_grid = {'n_estimators': n_estimators, 'max_features': max_features,
               'max_depth': max_depth} #'min_samples_split': min_samples_split,
               #'min_samples_leaf': min_samples_leaf}


rf_clf = RandomForestRegressor(random_state=42)
from sklearn.model_selection import GridSearchCV

rf_cv = GridSearchCV(rf_clf, params_grid, cv=3, verbose=2,n_jobs = -1)


rf_cv.fit(X_train, y_train)
best_params = rf_cv.best_params_
print(f"Best parameters: {best_params}")

rf_clf = RandomForestRegressor(**best_params)
rf_clf.fit(X_train, y_train)
print('Model Training is done')

joblib.dump(rf_clf , 'Insuranceprediction.pkl',)
print(rf_clf.predict([[46,1,24,2,0,1]])[0])



