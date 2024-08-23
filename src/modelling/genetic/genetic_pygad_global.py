import pandas as pd
from src.config import XGB_DATA_PATH, XGB_MODEL_PATH
import pygad
import xgboost as xgb
import numpy as np
import os
from src.modelling.genetic.functions import euclidean_distance, goal_player_angle, calculate_position
import warnings
from mplsoccer import Pitch
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import wandb
from tqdm import tqdm
import math

warnings.filterwarnings('ignore')
# Load the pre-trained XGBoost model


# Drop columns based on collected indices
SWEEP_ID = "thomas-toulouse/football-data-src_modelling_genetic/jfr8m8x0"
SHOW_PLOT = True
XGB_DATA_PATH_2 = os.path.join('data', 'processed', 'clean_action_data_glob.csv')

def custom_gene_initialization(distance, angle, max_radius=5):
    """Function to initialize each gene within a circle of a certain radius."""

    # for distance, angle in zip(*[iter(base_population[:2])]*2):
    x, y = calculate_position((base_population[0], base_population[1]), distance, angle)
                                
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(0, max_radius)

    x = x + radius * np.cos(angle)
    y = y + radius * np.sin(angle)

    # Ensure the coordinates are within the specified x and y ranges
    x = np.clip(x, x_range[0], x_range[1])
    y = np.clip(y, y_range[0], y_range[1])

    return [x, y]

def custom_mutation(offspring, ga_instance):
    """Custom mutation function to mutate the tuple genes."""
    for idx in range(offspring.shape[0]):
        if np.random.uniform(0, 100) < ga_instance.mutation_percent_genes:
            # Select a gene to mutate
            gene_idx = np.random.randint(low=0, high=offspring.shape[1]//2) * 2
            angle = np.random.uniform(0, 2 * np.pi)
            radius = np.random.uniform(0, max_radius)
            # Mutate either x or y within the ranges
            if np.random.rand() < 0.5:  # Mutate x
                offspring[idx, gene_idx] = offspring[idx, gene_idx] + radius * np.cos(angle)
                offspring[idx, gene_idx] = np.clip(offspring[idx, gene_idx], x_range[0], x_range[1])
            else:  # Mutate y
                offspring[idx, gene_idx + 1] = offspring[idx, gene_idx+1] + radius * np.cos(angle)
                offspring[idx, gene_idx + 1] = np.clip(offspring[idx, gene_idx + 1], y_range[0], y_range[1])
    return offspring


# Function to reshape the solution into the desired format for the XGBoost model
def reshape_solution(solution):
    new_data = initial_data.copy(deep=True).to_frame().T

    tuple_list = list(zip(solution[::2], solution[1::2]))

    for i, (x, y) in enumerate(tuple_list):
        if i == 0:
            new_data['shot_x'] = x
            new_data['shot_y'] = y
        else:
            distance = euclidean_distance((x, y), (new_data['shot_x'].item(), new_data['shot_y'].item()))
            angle = goal_player_angle((x, y), (new_data['shot_x'].item(), new_data['shot_y'].item()))
            new_data[f'distance_{teammates[i-1]}'] = distance
            new_data[f'angle_{teammates[i-1]}'] = angle

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

def get_teammates(data):
    teammates_with_1 = data[data.index.str.startswith('teammates_player') & (data == 1)]
    columns_with_1 = teammates_with_1.index.tolist()
    return columns_with_1

def complete_list(lst):
    diff = len(df) - len(lst)

    # add None to the list
    lst.extend([None] * diff)

    return lst

if __name__ == '__main__':
    max_radius = 5
    cols_to_keep = ['shot_statsbomb_xg', 'shot_x', 'shot_y', 'fk_x', 'fk_y', 'pass_angle', 'distance_to_goal'] 
    k = 10
    for i in range(1,k+1):
            cols_to_keep = cols_to_keep + [f'distance_player_{i}', f'angle_player_{i}', f'teammates_player_{i}']
    # init better improvement 
    max = 0
    best_iloc = 0
    df_base = pd.read_csv(XGB_DATA_PATH_2)

    df = df_base[cols_to_keep]

    overall_improv_list = list(filter(lambda x: not math.isnan(x), df_base['pctge_improvement'].to_list()))
    overall_fitness_list = list(filter(lambda x: not math.isnan(x), df_base['pred_improved_xg'].to_list()))
    overall_basexg = list(filter(lambda x: not math.isnan(x), df_base['pred_base_xg'].to_list()))


    xgboost_model = xgb.XGBRegressor()
    xgboost_model.load_model(XGB_MODEL_PATH)  # Replace with the actual path to your model
    num_tuples = 1 #x and y for shot
    x_range = (80, 115)  # Example range for x
    y_range = (10, 70)   # Example range for y
    api = wandb.Api()    
    sweep = api.sweep(SWEEP_ID)
    best_run = sweep.best_run()
    best_parameters = best_run.config

    if not overall_improv_list:
        index_to_start = 0
    else:
        index_to_start = len(overall_improv_list)
        
    print("Starting from index :", index_to_start)

    for i in tqdm(range(index_to_start, len(df))):  
        overall_improv_list = list(filter(lambda x: not math.isnan(x), df_base['pctge_improvement'].to_list()))
        overall_fitness_list = list(filter(lambda x: not math.isnan(x), df_base['pred_improved_xg'].to_list()))
        overall_basexg = list(filter(lambda x: not math.isnan(x), df_base['pred_base_xg'].to_list()))

        print("iloc :", i)
        
        initial_data = df.iloc[i]
        true_xg = df_base.iloc[i]['shot_statsbomb_xg']
        initial_data = initial_data.drop('shot_statsbomb_xg')
        initial_xg = xgboost_model.predict(initial_data.to_frame().T)[0]
        print(f'Initial position : ({initial_data["shot_x"]}, {initial_data["shot_y"]})\nInitial xG : {initial_xg}')

        teammates = get_teammates(initial_data)
        teammates = ['_'.join(teammate.split('_')[1:]) for teammate in teammates]

        # Prepare the initial population
        base_population = [initial_data["shot_x"], initial_data["shot_y"]]
        # base_population = []
        for teammate in teammates:
            base_population.append(initial_data[f'distance_{teammate}'])
            base_population.append(initial_data[f'angle_{teammate}'])

        initial_population = [base_population]
        
        for _ in range(best_parameters['sol_per_pop'] - 1):
            initial_gene = []
            for _ in range(2,len(base_population)+2,2):
                distance, angle = base_population[_], base_population[_+1]

                gene = custom_gene_initialization(distance, angle, max_radius=max_radius)

                initial_gene.extend(gene)

            initial_population.append(initial_gene)

        # Initialize the genetic algorithm
        # initial_population = [[initial_data["shot_x"], initial_data["shot_y"]]]
        # initial_population.extend([[gene for _ in range(num_tuples) for gene in custom_gene_initialization()] for _ in range(best_parameters['sol_per_pop']-1)])
    

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
        percent_improv = round(((best_solution_fitness - initial_xg)/initial_xg * 100)[0],2)
        print(f"Best Solution:{[round(x,2) for x in best_solution]}")
        print("Best Fitness:", best_solution_fitness[0])
        print(f"Improvement in %:  {percent_improv} %")

        overall_basexg.append(initial_xg)
        overall_improv_list.append(percent_improv)
        overall_fitness_list.append(best_solution_fitness[0])
        # Convert the best solution to a readable format (e.g., player positions)
        best_positions = reshape_solution(best_solution)
        # print("Optimal Player Positions:")
        # print(best_positions)

        if percent_improv > max:
            max = percent_improv
            best_iloc=i

        if SHOW_PLOT:
             # Load the freekick_pass_shot.csv file
            freekick_data = pd.read_csv(os.path.join('data', 'raw', 'freekick_pass_shot.csv'), index_col=0)
            freekick_data_2 = pd.read_csv(XGB_DATA_PATH)

            # Find the row where the shot_statsbomb_xg value matches your initial_xg value
            matching_row = freekick_data[freekick_data['shot_statsbomb_xg'] == true_xg]
            matching_row_2 = freekick_data_2[freekick_data_2['shot_statsbomb_xg'] == true_xg]

            freeze_frame = ast.literal_eval(matching_row['shot_freeze_frame'].values[0])
            shot_x = matching_row_2['shot_x'].values[0]
            shot_y = matching_row_2['shot_y'].values[0]
            new_shot_x, new_shot_y = best_solution[:2]
            fk_x = matching_row_2['fk_x'].values[0]
            fk_y = matching_row_2['fk_y'].values[0]

            pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')

            fig, ax = pitch.draw()

            # Extract locations and player names
            new_x, new_y = [], []

            for i,gene in enumerate(best_solution[2:]):
                if i==0 or i%2 == 0:
                    new_x.append(gene)
                else:
                    new_y.append(gene)

            x = [player['location'][0] for player in freeze_frame]
            y = [player['location'][1] for player in freeze_frame]
            names = [player['position']['id'] for player in freeze_frame]
            teammates = [player['teammate'] for player in freeze_frame]

            palette = {True: 'blue', False: 'red'}
            sns.scatterplot(x=x, y=y, ax=ax, s=50, hue=teammates, palette=palette, legend=False)
            sns.scatterplot(x=[shot_x], y=[shot_y], ax=ax, s=100, color='black', legend=False)
            sns.scatterplot(x=[fk_x], y=[fk_y], ax=ax, s=100, color='purple', legend=False)
            sns.scatterplot(x=[new_shot_x], y=[new_shot_y], ax=ax, s=100, color='white', legend=False)
            sns.scatterplot(x=new_x, y=new_y, ax=ax, s=75, color='yellow', legend=False)
            # Annotate the points with player names
            for i, name in enumerate(names):
                ax.annotate(name, (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=5, color='white')

            plt.show()

        print(f"Maximum Improvement: {max} %")
        print(f"Best Iloc: {best_iloc}")
        print('\n\n')
        df_base['pred_base_xg'] = complete_list(overall_basexg)
        df_base['pred_improved_xg'] = complete_list(overall_fitness_list)
        df_base['pctge_improvement'] = complete_list(overall_improv_list)

        df_base.to_csv(XGB_DATA_PATH_2, index = False)

        # with open(os.path.join('data','results','overall_improv.txt'), 'w') as f:
        #     for item in overall_improv_list:
        #         f.write("%s\n" % item)

        # with open(os.path.join('data','results','overall_fitness.txt'), 'w') as f: 
        #     for item in overall_fitness_list:
        #         f.write("%s\n" % item)
