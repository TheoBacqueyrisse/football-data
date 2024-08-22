import pandas as pd
from src.config import XGB_DATA_PATH, XGB_MODEL_PATH
import pygad
import xgboost as xgb
import numpy as np
import os
from src.modelling.genetic.functions import euclidean_distance, goal_player_angle
import warnings
from mplsoccer import Pitch
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import wandb

warnings.filterwarnings('ignore')
# Load the pre-trained XGBoost model
xgboost_model = xgb.XGBRegressor()
xgboost_model.load_model(XGB_MODEL_PATH)  # Replace with the actual path to your model

# Load the data
# cols_to_keep = ['shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal', 'distance_player_1', 'distance_player_2', 'distance_player_3', 'distance_player_4', 
#                 'angle_player_1', 'angle_player_2', 'angle_player_3', 'angle_player_4', 'teammates_player_1', 'teammates_player_2', 'teammates_player_3', 'teammates_player_4']

# initial_data = pd.read_csv(XGB_DATA_PATH, index_col=0)[cols_to_keep].iloc[7]
# initial_data = initial_data.drop('shot_statsbomb_xg')


# drop_distance_indice = []
# drop_angle_indice = []

# for teammate in teammates:
#     drop_distance_indice.append(initial_data.index.get_loc(f'distance_{teammate}'))
#     drop_angle_indice.append(initial_data.index.get_loc(f'angle_{teammate}'))

# drop_indices = drop_distance_indice + drop_angle_indice
# drop_indices = sorted(drop_indices, reverse=True)

# Drop columns based on collected indices
SWEEP_ID = "thomas-toulouse/football-data-src_modelling_genetic/jfr8m8x0"

num_tuples = 1 #x and y for shot
sol_per_pop = 50
x_range = (80, 112)  # Example range for x
y_range = (10, 70)   # Example range for y
api = wandb.Api()    
sweep = api.sweep(SWEEP_ID)
best_run = sweep.best_run()
best_parameters = best_run.config

def custom_gene_initialization():
    """Function to initialize each gene as a tuple within specified ranges."""
    x = np.random.uniform(low=x_range[0], high=x_range[1])
    y = np.random.uniform(low=y_range[0], high=y_range[1])
    return [x, y]

def custom_mutation(offspring, ga_instance):
    """Custom mutation function to mutate the tuple genes."""
    for idx in range(offspring.shape[0]):
        if np.random.uniform(0, 100) < ga_instance.mutation_percent_genes:
            # Select a gene to mutate
            gene_idx = np.random.randint(low=0, high=offspring.shape[1]//2) * 2

            # Mutate either x or y within the ranges
            if np.random.rand() < 0.5:  # Mutate x
                offspring[idx, gene_idx] = np.random.uniform(low=x_range[0], high=x_range[1])
            else:  # Mutate y
                offspring[idx, gene_idx + 1] = np.random.uniform(low=y_range[0], high=y_range[1])

    return offspring


# Function to reshape the solution into the desired format for the XGBoost model
def reshape_solution(solution):
    new_data = initial_data.copy(deep=True).to_frame().T

    tuple_list = list(zip(solution[::2], solution[1::2]))

    for i,(x,y) in enumerate(tuple_list):
        new_data['shot_x'] = x
        new_data['shot_y'] = y

    new_data = new_data[initial_data.copy(deep=True).to_frame().T.columns]

    return new_data

# Fitness function
def fitness_function(ga_class, solution, solution_idx):
    # Reshape the solution to fit the expected input of the XGBoost model
    model_input = reshape_solution(solution)
    
# Get the feature names from the training data
    feature_names = xgboost_model.feature_names_in_

# Get the feature names from the input data

    # Check if the feature names are consistent
    if len(feature_names) != len(model_input.columns):
        raise ValueError("Feature names mismatch")

    predicted_xg = xgboost_model.predict(model_input)
    
    return predicted_xg


cols_to_keep = ['shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal'] 
k = 10
for i in range(1,k+1):
        cols_to_keep = cols_to_keep + [f'distance_player_{i}', f'angle_player_{i}', f'teammates_player_{i}']
# init better improvement 
max = 0
best_iloc = 0
df = pd.read_csv(XGB_DATA_PATH, index_col=0)[cols_to_keep]

for i in range(len(pd.read_csv(XGB_DATA_PATH, index_col=0))):  
    print("iloc :", i)
    
    initial_data = df.iloc[i]
    initial_xg = initial_data.shot_statsbomb_xg
    initial_data = initial_data.drop('shot_statsbomb_xg')
    print(f'Initial position : ({initial_data["shot_x"]}, {initial_data["shot_y"]})\nInitial xG : {xgboost_model.predict(initial_data.to_frame().T)[0]}')


    # Initialize the genetic algorithm
    initial_population = [[gene for _ in range(num_tuples) for gene in custom_gene_initialization()] for _ in range(50)]
 

    # Genetic Algorithm Parameters
    ga_instance = pygad.GA(
    num_generations=best_parameters['num_generations'],
    num_parents_mating=best_parameters['num_parents_mating'],
    fitness_func=fitness_function,
    gene_type=object,
    initial_population=initial_population,
    mutation_type=custom_mutation,
    mutation_percent_genes=best_parameters['mutation_percent_genes'],
    crossover_type=best_parameters['crossover_type'],
    parent_selection_type=best_parameters['parent_selection_type'],
    keep_parents=best_parameters['keep_parents'],
)
    

    # Run the Genetic Algorithm
    ga_instance.run()

    # After the algorithm completes, get the best solution
    best_solution, best_solution_fitness, _ = ga_instance.best_solution()

    print(f"Best Solution:({round(best_solution[0])},{round(best_solution[1])}")
    print("Best Fitness:", best_solution_fitness[0])
    print(f"Improvement in %:  {round(((best_solution_fitness - initial_xg)/initial_xg * 100)[0],2)} %")
    # Convert the best solution to a readable format (e.g., player positions)
    best_positions = reshape_solution(best_solution)
    # print("Optimal Player Positions:")
    # print(best_positions)

    if (best_solution_fitness - initial_xg) > max:
        max = (best_solution_fitness - initial_xg)
        best_iloc=i

    print()
     # Load the freekick_pass_shot.csv file
    freekick_data = pd.read_csv(os.path.join('data', 'raw', 'freekick_pass_shot.csv'), index_col=0)
    freekick_data_2 = pd.read_csv(os.path.join('data', 'processed', 'clean_action_data.csv'))

    # Find the row where the shot_statsbomb_xg value matches your initial_xg value
    matching_row = freekick_data[freekick_data['shot_statsbomb_xg'] == initial_xg]
    matching_row_2 = freekick_data_2[freekick_data_2['shot_statsbomb_xg'] == initial_xg]

    # If a matching row is found, extract the freeze frame data
    if not matching_row.empty:
        freeze_frame = ast.literal_eval(matching_row['shot_freeze_frame'].values[0])
        shot_x = matching_row_2['shot_x'].values[0]
        shot_y = matching_row_2['shot_y'].values[0]
        new_shot_x, new_shot_y = best_solution
        fk_x = matching_row_2['fk_x'].values[0]
        fk_y = matching_row_2['fk_y'].values[0]

        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')

        fig, ax = pitch.draw()

        # Extract locations and player names
        x = [player['location'][0] for player in freeze_frame]
        y = [player['location'][1] for player in freeze_frame]
        names = [player['position']['id'] for player in freeze_frame]
        teammates = [player['teammate'] for player in freeze_frame]

        palette = {True: 'blue', False: 'red'}
        sns.scatterplot(x=x, y=y, ax=ax, s=50, hue=teammates, palette=palette, legend=False)
        sns.scatterplot(x=[shot_x], y=[shot_y], ax=ax, s=100, color='black', legend=False)
        sns.scatterplot(x=[fk_x], y=[fk_y], ax=ax, s=100, color='purple', legend=False)
        sns.scatterplot(x=[new_shot_x], y=[new_shot_y], ax=ax, s=100, color='white', legend=False)
        # Annotate the points with player names
        for i, name in enumerate(names):
            ax.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='white')

        plt.show()
    else:
        print("No matching row found in freekick_pass_shot.csv")
