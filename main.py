import pandas as pd
from src.pengouins.data import load_data, get_X_y,split_data, preprocess_data

# Import data
path ="./data/pingouins.csv"
pingouins = load_data(path)
print(pingouins.isna().sum())
X,y = get_X_y(pingouins, "species")
print(X.shape, y.shape,len(y), type(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.3)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Preprocess data
X_train_preprocessed, fitted_preprocessor = preprocess_data(X_train, fit=True)
X_test_preprocessed = fitted_preprocessor.transform(X_test)

print(f'PreProcesssor{X_train_preprocessed.shape}, {X_test_preprocessed.shape}')
print(fitted_preprocessor)

print(pd.DataFrame(X_train_preprocessed).head())