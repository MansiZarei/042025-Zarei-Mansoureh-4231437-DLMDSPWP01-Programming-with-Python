# src/test_mapper.py

import pandas as pd
from sqlalchemy.orm import sessionmaker
from database import TestResults
from function_selector import FunctionSelector
import logging

class TestMapper:
    def __init__(self, db_manager, function_selector):
        self.db_manager = db_manager
        self.function_selector = function_selector
        self.session = self.db_manager.get_session()

    def map_test_data(self):
        try:
            test_df = pd.read_sql_table('test_data', self.db_manager.engine)
            ideal_df = pd.read_sql_table('ideal_functions', self.db_manager.engine)
            selected = self.function_selector.get_selected_functions()
            max_devs = self.function_selector.get_max_deviations()

            if not selected or not max_devs:
                logging.info("No ideal functions selected. Skipping test data mapping.")
                return

            results = []

            for _, row in test_df.iterrows():
                x = row['x']
                y = row['y']
                deviations = []
                for func_no in selected:
                    # Find the exact x in ideal_df
                    ideal_row = ideal_df[ideal_df['x'] == x]
                    if ideal_row.empty:
                        # If exact x not found, find the nearest x
                        nearest_idx = (ideal_df['x'] - x).abs().idxmin()
                        ideal_y = ideal_df.at[nearest_idx, f'y{func_no}']
                    else:
                        ideal_y = ideal_row.iloc[0][f'y{func_no}']
                    deviation = abs(y - ideal_y)
                    deviations.append(deviation)

                if not deviations:
                    continue  # Skip if no deviations calculated

                min_dev = min(deviations)
                best_func_index = deviations.index(min_dev)
                best_func = selected[best_func_index]
                if min_dev <= max_devs[best_func_index]:
                    results.append({
                        'x': x,
                        'y': y,
                        'delta_y': min_dev,
                        'ideal_function': best_func
                    })

            if results:
                results_df = pd.DataFrame(results)
                results_df.to_sql('test_results', self.db_manager.engine, if_exists='replace', index=False)
                logging.info("Test data mapped and results saved successfully.")
            else:
                logging.info("No test data points matched the criteria.")

        except Exception as e:
            logging.error(f"Error during test data mapping: {e}")
