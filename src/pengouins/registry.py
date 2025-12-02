import os
import pickle

def save_model(model, filepath):
    """Saves the model to the specified filepath using pickle."""
    model_path = os.path.dirname(filepath)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    with open(filepath, "wb") as f:
        pickle.dump(model, f)

def load_model(filepath):
    """Loads the model from the specified filepath using pickle."""
    model_path = os.path.dirname(filepath)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"The directory {model_path} does not exist.")
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model