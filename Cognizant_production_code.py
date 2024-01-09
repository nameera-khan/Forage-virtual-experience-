import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

model = RandomForestRegressor()
scaler = StandardScaler()
# Load data
def load_data(path: str = "/path/to/csv/"):
    """
    This function takes a path string to a CSV file and loads it into
    a Pandas DataFrame.

    :param      path (optional): str, relative path of the CSV file

    :return     df: pd.DataFrame
    """

    df = pd.read_csv(f"{path}")
    df.drop(columns=["Unnamed: 0"], inplace=True, errors='ignore')
    return df

# Create target variable and predictor variables
def create_target_and_predictors(
    data: pd.DataFrame = None, 
    target: str = "estimated_stock_pct"
):
    """
    This function takes in a Pandas DataFrame and splits the columns
    into a target column and a set of predictor variables, i.e. X & y.
    These two splits of the data will be used to train a supervised 
    machine learning model.

    :param      data: pd.DataFrame, dataframe containing data for the 
                      model
    :param      target: str (optional), target variable that you want to predict

    :return     X: pd.DataFrame
                y: pd.Series
    """

    # Check to see if the target variable is present in the data
    if target not in data.columns:
        raise Exception(f"Target: {target} is not present in the data")
    
    X = data.drop(columns=[target])
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test 

def train_with_crossval(X_train: pd.DataFrame = None, 
    y_train: pd.Series = None,
    X_test: pd.DataFrame = None,
    y_test: pd.DataFrame = None):
    """
    This function takes the predictor and target variables and
    trains a Random Forest Regressor model across K folds using cross_val_score function. Using sklearn's
    cross-validation, performance metrics will be output for each
    fold during training.

    :param      X_train: pd.DataFrame, predictor variables for training
    :param      y_train: pd.Series, target variable for training
    :param      X_test: pd.DataFrame, predictor variables for test
    :param      y_test: pd.DataFrame, target variable for test


    :return
    """
    

    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    cross_val_rf= cross_val_score(rf, X_train, y_train, scoring='neg_mean_squared_error', cv=kf)
    rmse_rf = np.sqrt(-cross_val_rf)
    print("Cross-Validation RMSE Scores:", cross_val_rf)
    print("Average RMSE:", np.mean(rmse_rf))

    model.fit(X_train,y_train)
    pred_rf = model.predict(X_test)
    mae_rf = mean_absolute_error(y_test, pred_rf)

    print("Mean absolute Error (MSE):", mae_rf)

    
def run_algorithm():
    """
   Function to execute the training pipeline for the machine learning task. 
    
    """
    df = load_data()
    X_train, X_test, y_train, y_test  = create_target_and_predictors
    train_with_crossval(X_train, X_test, y_train, y_test)

def feature_importance(model, 
                     X:  pd.DataFrame=None,
                     y:  pd.Series = None):
    """
    Function to plot the model feature importances
    
    :param      X: predictor variables
    :param      y: target variable
    """
    importance = model.feature_importances_
    indices = np.argsort(importance)

    feature_importances = pd.DataFrame({'features': X.columns,
        'importance': model.feature_importances_
        }).sort_values(by='importance', ascending=True).reset_index()
    
    plt.figure(figsize=(15, 25))
    plt.title('Feature Importances')
    plt.barh(range(len(feature_importances)), feature_importances['importance'], color='b', align='center')
    plt.yticks(range(len(feature_importances)), feature_importances['features'])
    plt.xlabel('Importance')
    plt.show()