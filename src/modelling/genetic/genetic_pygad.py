import pandas as pd
from src.config import XGB_DATA_PATH
import pygad
import xgboost as xgb
import numpy as np
import os
from src.modelling.genetic.functions import euclidean_distance, goal_player_angle

# Load the pre-trained XGBoost model
xgboost_model = xgb.XGBRegressor()
xgboost_model.load_model('src\\modelling\\xgboost\\xgb.json')  # Replace with the actual path to your model

# Load the data
initial_data = pd.read_csv(XGB_DATA_PATH, index_col=0).iloc[7]
initial_data = initial_data.drop('shot_statsbomb_xg')

def get_teammates(data):
    teammates_with_1 = data[data.index.str.startswith('teammates_player') & (data == 1)]

    # Get the column names
    columns_with_1 = teammates_with_1.index.tolist()

    return columns_with_1

teammates = get_teammates(initial_data)


teammates = ['_'.join(teammate.split('_')[1:]) for teammate in teammates]


# drop_distance_indice = []
# drop_angle_indice = []

# for teammate in teammates:
#     drop_distance_indice.append(initial_data.index.get_loc(f'distance_{teammate}'))
#     drop_angle_indice.append(initial_data.index.get_loc(f'angle_{teammate}'))

# drop_indices = drop_distance_indice + drop_angle_indice
# drop_indices = sorted(drop_indices, reverse=True)

# Drop columns based on collected indices


nb_param = len(teammates)*2 #x and y every player


# Function to reshape the solution into the desired format for the XGBoost model
def reshape_solution(solution):
    new_data = initial_data.copy(deep=True).to_frame().T

    tuple_list = list(zip(solution[::2], solution[1::2]))

    for i,(x,y) in enumerate(tuple_list):
        distance = euclidean_distance((x,y), (initial_data['shot_x'], initial_data['shot_y']))
        angle = goal_player_angle((x,y), (initial_data['shot_x'], initial_data['shot_y']))
        # print(f'Inserted distance {distance} and angle {angle} for player {teammates[i]}')

        new_data[f'distance_{teammates[i]}'] = distance
        new_data[f'angle_{teammates[i]}'] = angle

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

# Genetic Algorithm Parameters
ga_instance = pygad.GA(
    num_generations=100,  # Number of generations
    num_parents_mating=10,  # Number of parents selected for mating
    fitness_func=fitness_function,  # Fitness function
    sol_per_pop=50,  # Number of solutions in the population
    num_genes=nb_param,  # Number of genes (parameters to optimize)
    gene_type=float,  # Data type for each gene
    init_range_low=0,  # Lower bound for gene values
    init_range_high=100,  # Upper bound for gene values (adjust based on the field dimensions)
    mutation_type="random",  # Mutation type
    mutation_percent_genes=30,  # Percentage of genes to mutate
    crossover_type="single_point",  # Type of crossover
    parent_selection_type="sss",  # Parent selection method (Stochastic universal sampling)
    keep_parents=2,  # Number of parents to keep in the next generation
    on_generation=lambda ga: print(f"Generation {ga.generations_completed}: Best Fitness = {ga.best_solution()[1]}, Best Position = {ga.best_solution()[0]}")  # Callback to print fitness at each generation
)

# Run the Genetic Algorithm
ga_instance.run()

# After the algorithm completes, get the best solution
best_solution, best_solution_fitness, _ = ga_instance.best_solution()

print("Best Solution:", best_solution)
print("Best Fitness:", best_solution_fitness)

# Convert the best solution to a readable format (e.g., player positions)
best_positions = reshape_solution(best_solution)
print("Optimal Player Positions:")
print(best_positions)
