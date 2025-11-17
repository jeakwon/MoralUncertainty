import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def aggregate_results(save_dir, num_samples_per_dimension):
    base_path = os.path.join(save_dir, str(num_samples_per_dimension))
    results = []
    
    model_names = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    for model_name in model_names:
        model_path = os.path.join(base_path, model_name)

        if len( os.listdir(model_path) ) < 10:
            continue
        
        for filename in os.listdir(model_path):
            if filename.endswith('.csv'):
                scenario_dimension = filename[:-4]
                csv_path = os.path.join(model_path, filename)
                death_probs = pd.read_csv(csv_path)
                
                results.append(
                    dict(model_name=model_name, scenario_dimension=scenario_dimension, death_probs=death_probs)
                )
    
    df_agg = pd.DataFrame(results).pivot(index='model_name', columns='scenario_dimension', values='death_probs')
    return df_agg