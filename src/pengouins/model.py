import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier


def train_model(  X_train : pd.DataFrame
                , y_train: pd.Series):
    """Train a model on the training data."""

    logreg = LogisticRegression(max_iter=200)
    logreg.fit(X_train, y_train)
    
    xgboost_classifier = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    xgboost_classifier.fit(X_train, y_train)
    
    return logreg
    

def evaluate_model( model
                    , X_test: pd.DataFrame
                    , y_test: pd.Series) -> float:
    """Evaluate the model on the test data and return accuracy."""
    pass