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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor


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


if __name__ == "__main__":
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

    #RMSE
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

