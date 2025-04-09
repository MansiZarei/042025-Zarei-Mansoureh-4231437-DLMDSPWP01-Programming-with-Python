# src/visualizer.py

from bokeh.plotting import figure, output_file, show
from bokeh.layouts import gridplot
import pandas as pd

class Visualizer:
    def __init__(self, db_manager):
        self.db_manager = db_manager

    def plot_training_data(self):
        training_df = pd.read_sql_table('training_data', self.db_manager.engine)
        p = figure(title="Training Data", x_axis_label='X', y_axis_label='Y')
        for i in range(1, 5):
            p.line(training_df['x'], training_df[f'y{i}'], line_width=2, legend_label=f'Training Function y{i}')
        p.legend.location = "top_left"
        return p

    def plot_ideal_functions(self, selected_functions):
        ideal_df = pd.read_sql_table('ideal_functions', self.db_manager.engine)
        p = figure(title="Ideal Functions", x_axis_label='X', y_axis_label='Y')
        for func_no in selected_functions:
            p.line(ideal_df['x'], ideal_df[f'y{func_no}'], line_width=2, legend_label=f'Ideal Function y{func_no}')
        p.legend.location = "top_left"
        return p

    def plot_test_data(self):
        results_df = pd.read_sql_table('test_results', self.db_manager.engine)
        p = figure(title="Test Data and Deviations", x_axis_label='X', y_axis_label='Y')
        # Replace 'circle()' with 'scatter()' to adhere to Bokeh's updated API
        p.scatter(results_df['x'], results_df['y'], size=8, color='navy', alpha=0.5, legend_label='Test Data')
        # Plot deviations as vertical dashed lines
        for _, row in results_df.iterrows():
            p.line([row['x'], row['x']], [row['y'], row['y'] - row['delta_y']], line_dash='dashed', color='red')
        p.legend.location = "top_left"
        return p

    def show_plots(self, training_plot, ideal_plot, test_plot):
        grid = gridplot([[training_plot], [ideal_plot], [test_plot]], sizing_mode='stretch_both')
        output_file("data_visualization.html")
        show(grid)
