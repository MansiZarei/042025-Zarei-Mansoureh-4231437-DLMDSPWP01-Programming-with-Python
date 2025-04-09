# src/main.py

import os
from database import DatabaseManager
from data_loader import DataLoader
from function_selector import FunctionSelector
from test_mapper import TestMapper
from visualizer import Visualizer

def main():
    # Determine the absolute path to the data directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, '..', 'data')
    training_file = os.path.join(data_dir, 'train.csv')
    ideal_file = os.path.join(data_dir, 'ideal.csv')
    test_file = os.path.join(data_dir, 'test.csv')

    # Check if files exist
    for file_path in [training_file, ideal_file, test_file]:
        if not os.path.exists(file_path):
            print(f"File {file_path} not found.")
            return  # Exit the program if any file is missing

    # Initialize Database Manager
    db_manager = DatabaseManager('datasets.db')
    db_manager.create_tables()

    # Initialize Data Loader
    data_loader = DataLoader(db_manager)
    data_loader.load_all_data(training_file, ideal_file, test_file)

    # Select Best Fit Functions
    selector = FunctionSelector(db_manager)
    selector.calculate_least_squares()
    selected_functions = selector.get_selected_functions()

    # Map Test Data
    mapper = TestMapper(db_manager, selector)
    mapper.map_test_data()

    # Visualize Data
    visualizer = Visualizer(db_manager)
    training_plot = visualizer.plot_training_data()
    ideal_plot = visualizer.plot_ideal_functions(selected_functions)
    test_plot = visualizer.plot_test_data()
    visualizer.show_plots(training_plot, ideal_plot, test_plot)

if __name__ == "__main__":
    main()
