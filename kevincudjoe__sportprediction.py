# -*- coding: utf-8 -*-
"""KevinCudjoe._SportPrediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1E7-mhXh5icDsYhQpEkiMfIwbQuGGBlF5

## **Data Cleaning**

### Imports
"""

pip install scikit-learn==1.5.0

import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from google.colab import drive
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, mean_squared_log_error, r2_score
import joblib as jb
drive.mount('/content/drive')

"""### Loading the data"""

df = pd.read_csv("drive/My Drive/male_players (legacy).csv", chunksize = 1000)
Df = pd.DataFrame()
for chunk in df:
    Df = pd.concat([Df,chunk]).reset_index(drop = True)

Df

"""### Dropping columns with greater that 30% of its values being null"""

L = []
L_less = []
for i in Df.columns:
    if((Df[i].isnull().sum())<(0.4*(Df.shape[0]))):
        L.append(i)
    else:
        L_less.append(i)
Df = Df[L]

Df

numeric_data = Df.select_dtypes(include=[np.number])
categorical_data = Df.select_dtypes(exclude=[np.number])

numeric_data.columns

num_corr = numeric_data.corr()

type(num_corr)

num_corr["overall"].sort_values(ascending=False)

for x in num_corr["overall"].index:
    if not(num_corr["overall"][x]>0.3 or num_corr["overall"][x]<-0.3):
        numeric_data.drop([x], axis = 1, inplace = True)

numeric_data.columns

num_arr = ['player_id','release_clause_eur']
numeric_data.drop(columns=num_arr, inplace= True)

categorical_data

arr = ['player_url','fifa_update_date','player_positions', 'short_name', 'long_name','dob', 'league_name', 'club_name', 'club_position','club_joined_date','nationality_name','real_face', 'lw', 'lf', 'cf', 'rf', 'rw', 'lam', 'cam',
       'ram', 'lm', 'lcm', 'cm', 'rcm', 'rm', 'lwb', 'ldm', 'cdm', 'rdm','rwb', 'lb', 'lcb', 'cb', 'rcb', 'rb', 'gk', 'player_face_url', "ls", "st", "rs"]
categorical_data.drop(columns=arr, inplace= True)

categorical_data.columns

im = IterativeImputer(max_iter = 10, random_state = 0 )
imputed_numeric = im.fit_transform(numeric_data)
numeric_data = pd.DataFrame(imputed_numeric,columns = numeric_data.columns)

si = SimpleImputer(strategy = "most_frequent")
imputed_categorical = si.fit_transform(categorical_data)
categorical_data = pd.DataFrame(imputed_categorical,columns = categorical_data.columns)

categorical_data.columns

label_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

for x in categorical_data.columns:
     categorical_data[x] = label_encoder.fit_transform(categorical_data[x])
categorical_data

onehot_encoded = onehot_encoder.fit_transform(categorical_data)
feature_names = onehot_encoder.get_feature_names_out(categorical_data.columns)
print(len(feature_names))

onehot_encoded_categorical = pd.DataFrame(onehot_encoded, columns = feature_names)

Df = pd.concat([numeric_data,onehot_encoded_categorical], axis = 1).reset_index(drop = True)

len(Df.columns)

"""### Data Training, Evaluation and Testing"""

y = Df["overall"]
x = Df.drop("overall", axis=1)

scale=StandardScaler()
X=scale.fit_transform(x)

from sklearn.model_selection import train_test_split
Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size = 0.2, random_state = 42)

xgb_regressor = XGBRegressor()
xgb_regressor.fit(Xtrain, Ytrain)

feature_importances = xgb_regressor.feature_importances_

type(feature_importances)

feature_importance_df = pd.DataFrame({
    'Feature': x.columns,
    'Importance': feature_importances
})

feature_importance_df.sort_values("Importance",ascending=False)

feature_importance_df.shape

arr = []
for x in feature_importance_df.iloc[:10,0]:
    arr.append(x)

Df = Df[arr]

x = Df

x

scale=StandardScaler()
X=scale.fit_transform(x)

Xtrain,Xtest,Ytrain,Ytest = train_test_split(X,y,test_size = 0.2, random_state = 42)

"""###Training Individual Models"""

cv=KFold(n_splits=5)
models = {
    'RandomForest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42),
    'GradientBoost': GradientBoostingRegressor(random_state=42)
}
mae_scorer = make_scorer(mean_absolute_error)

for name,model in models.items():
  scores = cross_val_score(model, Xtrain, Ytrain, cv=cv, scoring=mae_scorer)
  print(f'{name} MAE: {scores.mean():.4f} (+/- {scores.std():.4f})')

Df

"""## Ensemble methods
### RandomForest, XGBoost, Gradient Boost Regressors

"""

#Random Forest
cv=KFold(n_splits=5)
rf = RandomForestRegressor()
PARAMETERS_rf ={
"n_estimators":[10,50, 100, 200]
}
model_rf=GridSearchCV(rf,param_grid=PARAMETERS_rf,cv=cv,scoring="neg_mean_squared_error")
model_rf.fit(Xtrain, Ytrain)
jb.dump(model_rf, open('drive/My Drive/' + rf.__class__.__name__ + '.joblib', 'wb'))
y_pred = model_rf.predict(Xtest)
print(
    f"""
    Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
    Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
    R2 Score = {r2_score(y_pred,Ytest)}
    """
    )

#GradientBoostRegressor
cv=KFold(n_splits=3)
gb = GradientBoostingRegressor(n_iter_no_change=10, validation_fraction=0.1)
PARAMETERS_gb ={
"max_depth":[3,5,8],
"min_samples_leaf":[1,5,10],
"min_samples_split":[2,5],
# "min_child_weight":[1,5,15],​
"learning_rate":[0.5, 0.1],
"n_estimators":[100]}
model_gs=GridSearchCV(gb,param_grid=PARAMETERS_gb,cv=cv,scoring="neg_mean_squared_error", n_jobs=1)
model_gs.fit(Xtrain, Ytrain)
jb.dump(model_gs, open('drive/My Drive/' + gb.__class__.__name__ + '.joblib', 'wb'))
y_pred = model_gs.predict(Xtest)
print(model_gs.__class__.__name__,
    f"""
    Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
    Mean Squared Error = {mean_squared_error(y_pred,Ytest)},
    Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
    R2 Score = {r2_score(y_pred,Ytest)}
    """)

#XGBOOST
cv=KFold(n_splits=3)
xg = XGBRegressor()
PARAMETERS_xg ={
"n_estimators":[100,500,1000]}
model_xg=GridSearchCV(xg,param_grid=PARAMETERS_xg,cv=cv,scoring="neg_mean_squared_error")
model_xg.fit(Xtrain, Ytrain)
best_model_xg = model_xg.best_estimator_
print("Best parameters found: ", model_xg.best_params_)

jb.dump(model_xg, open('drive/My Drive/' + xg.__class__.__name__ + '.joblib', 'wb'))
y_pred = model_xg.predict(Xtest)

print(xg.__class__.__name__,
    f"""
    Mean Absolute Error = {mean_absolute_error(y_pred,Ytest)},
    Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,Ytest))},
    R2 Score = {r2_score(y_pred,Ytest)}
    """)

df = pd.read_csv("drive/My Drive/players_22.csv", chunksize = 1000)
new_Df = pd.DataFrame()
for chunk in df:
    new_Df = pd.concat([new_Df,chunk]).reset_index(drop = True)

y = new_Df['overall']
X = new_Df[arr]

X

imputed_numeric = im.fit_transform(X)
X = pd.DataFrame(imputed_numeric,columns = X.columns)
X=scale.fit_transform(X)

y_pred = model_rf.predict(X)
print(model_rf.__class__.__name__,
    f"""
    Mean Absolute Error = {mean_absolute_error(y_pred,y)},
    Root Mean Squared Error = {np.sqrt(mean_squared_error(y_pred,y))},
    R2 Score = {r2_score(y_pred,y)}
    """)

type(Xtrain)

