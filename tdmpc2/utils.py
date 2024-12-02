import os
import pickle
import sys
from tdmpc2 import TDMPC2

def save_model(model: TDMPC2, file_path: str):
    if os.path.exists(file_path):
        raise FileExistsError(f"The file '{file_path}' already exists.")
    else:
        with open(file_path, 'wb') as file:
            pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Model saved to '{file_path}' successfully.")

def load_model(file_path: str) -> TDMPC2:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")
    else:
        with open(file_path, 'r') as file:
            return pickle.load(file)
