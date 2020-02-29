# Common imports
import numpy as np

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor


HOUSING_TRAIN_DATA = "train.csv" 
HOUSING_TEST_DATA = "test.csv"
def load_housing_data(train_path=HOUSING_TRAIN_DATA, test_path=HOUSING_TEST_DATA):
    return pd.read_csv(train_path), pd.read_csv(test_path)

def transformData(housing_data):
    columns = ['SalePrice', 'LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'BldgType', 'GrLivArea']
    train_set_reduced = housing_data[columns].copy()

    train_set_reduced['TotBath'] = train_set_reduced['HalfBath'] + train_set_reduced['FullBath'] #total bathrooms

    train_set_X = train_set_reduced.drop('SalePrice',axis=1)
    train_set_y = train_set_reduced["SalePrice"].copy()
    return train_set_X, train_set_y

#Testing ML pipelines
def attempt1():
    train_val_set, test_set = load_housing_data()

    X, y = transformData(train_val_set)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    num_attribs = ['LotArea', 'YearBuilt', 'OverallQual', 'OverallCond', 'TotRmsAbvGrd', 'TotBath', 'GrLivArea']
    cat_attribs = ["BldgType"]

    full_pipeline = ColumnTransformer([
            ("num", StandardScaler(), num_attribs),
            ("cat", OneHotEncoder(), cat_attribs),
    ])

    train_set_X_prepared = full_pipeline.fit_transform(X_train)
    val_set_X_prepared = full_pipeline.fit_transform(X_val)
    
    #Prediction using Linear Regression
    lin_reg = LinearRegression()
    lin_reg.fit(train_set_X_prepared, y_train)

    #MSE
    housing_predictions = lin_reg.predict(val_set_X_prepared)
    lin_mse = mean_squared_error(y_val, housing_predictions)
    lin_rmse = np.sqrt(lin_mse)
    print("Linear Regression: ",lin_rmse)

    #Prediction using Decision Tree
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(train_set_X_prepared, y_train)

    housing_predictions = tree_reg.predict(val_set_X_prepared)
    tree_mse = mean_squared_error(y_val, housing_predictions)
    tree_rmse = np.sqrt(tree_mse)
    print("Decision Tree: ",tree_rmse)




def get_xgboost_model():
    data = pd.read_csv(HOUSING_TRAIN_DATA)
    data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = data.SalePrice
    X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=40)

    imputer = SimpleImputer(strategy="median")
    train_X = imputer.fit_transform(train_X)
    test_X = imputer.transform(test_X)

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    my_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(test_X, test_y)], verbose=False)

    # make predictions
    predictions = my_model.predict(test_X)
    print("XGBoost : ", np.sqrt(mean_squared_error(predictions, test_y)))
    return my_model

def create_submission_file(test_data, predictions):
    my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predictions})
    my_submission.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    attempt1()

    test_data = pd.read_csv(HOUSING_TEST_DATA)
    test_data = test_data.select_dtypes(exclude=['object'])
    imputer = SimpleImputer(strategy="median")
    test_X = imputer.fit_transform(test_data)

    xgboost_model = get_xgboost_model()
    predictions = xgboost_model.predict(test_X)
    
    create_submission_file(test_data, predictions)