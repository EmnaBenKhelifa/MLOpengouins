"""
Load and preprocess data.
"""

import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_data(path: str) -> pd.DataFrame:
    """Load data from seaborn data,
    put it in cache and return a DataFrame."""
    df = sns.load_dataset('penguins')
    
    if not os.path.exists("./data"):
        os.makedirs("./data")  
    df.to_csv(path, index=False)
    return df


def get_X_y(
    df: pd.DataFrame, target_column: str, target:bool = True
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target."""
    
    df.drop(columns=["island"], inplace=True)
    
    if target:
        X = df.drop(columns=[target_column])
        y = df[target_column]
        return X, y
    else:
        return df, None

def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
    
def preprocess_data(X: pd.DataFrame
                    ,fit = True) -> tuple[pd.DataFrame, ColumnTransformer] | pd.DataFrame:
    """Preprocess data: handle missing values, encode categorical variables, scale numerical features."""
    # Detect numerical and categorical columns
    numerical_cols = X.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    # Define transformers for numerical and categorical features
    numerical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
            ])
    
    categorical_transformer = Pipeline(
        steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
            ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])
 
    # Fit and transform the data
    if fit:
        X_preprocessed = preprocessor.fit_transform(X)
        return X_preprocessed, preprocessor
    else:
        X_preprocessed = preprocessor.transform(X)
        return X_preprocessed