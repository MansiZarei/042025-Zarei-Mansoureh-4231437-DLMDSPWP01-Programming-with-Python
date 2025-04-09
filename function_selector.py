# src/function_selector.py

import numpy as np
import pandas as pd
import logging
from sqlalchemy.orm import sessionmaker
from database import TrainingData, IdealFunctions

class FunctionSelector:
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.session = self.db_manager.get_session()
        self.selected_functions = []
        self.max_deviations = []

    def calculate_least_squares(self):
        try:
            training_df = pd.read_sql_table('training_data', self.db_manager.engine)
            ideal_df = pd.read_sql_table('ideal_functions', self.db_manager.engine)

            if training_df.empty:
                logging.error("Training data is empty. No functions to select.")
                return  # Exit early if training data is empty

            # Check for required columns in training_data
            required_training_columns = ['y1', 'y2', 'y3', 'y4']
            for col in required_training_columns:
                if col not in training_df.columns:
                    logging.error(f"Column '{col}' not found in training data.")
                    return  # Exit early if any required column is missing

            # Check for required columns in ideal_functions
            required_ideal_columns = [f'y{i}' for i in range(1, 51)]
            for col in required_ideal_columns:
                if col not in ideal_df.columns:
                    logging.error(f"Column '{col}' not found in ideal functions data.")
                    return  # Exit early if any required column is missing

            deviations = np.zeros((4, 50))  # 4 training functions, 50 ideal functions

            for i in range(1, 5):  # y1 to y4
                y_train = training_df[f'y{i}']
                for j in range(1, 51):  # y1 to y50
                    y_ideal = ideal_df[f'y{j}']
                    deviations[i-1, j-1] = np.sum((y_train - y_ideal) ** 2)

            # Select the ideal function with the smallest deviation for each training function
            for i in range(4):
                min_index = np.argmin(deviations[i])
                self.selected_functions.append(min_index + 1)  # Ideal function numbers start at 1
                self.max_deviations.append(deviations[i][min_index] * np.sqrt(2))

            logging.info(f"Selected Ideal Functions: {self.selected_functions}")
            logging.info(f"Max Deviations: {self.max_deviations}")

        except KeyError as ke:
            logging.error(f"Key Error: {ke}")
        except Exception as e:
            logging.error(f"Unexpected error during least squares calculation: {e}")

    def get_selected_functions(self):
        return self.selected_functions

    def get_max_deviations(self):
        return self.max_deviations
