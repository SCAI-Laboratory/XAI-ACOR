import logging
logging.disable(logging.CRITICAL)

import numpy as np
import pandas as pd
from mealpy import FloatVar, ACOR
from ConfigSpace import ConfigurationSpace
from iohxplainer import explainer
import traceback
import time
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
import multiprocessing as mp

# IMPORTANT: Define run_ACOR at the MODULE LEVEL (outside __main__)
def run_ACOR(func, config, budget, dim, *args, **kwargs):
    problem_dict = {
        "bounds": FloatVar(lb=(-5.,) * 5, ub=(5.,) * 5
        , name="delta"),
        "obj_func": func,
        "minmax": "min",
    }
    
    intent_factor = config.get("intent_factor")
    pop_size = config.get("pop_size")
    sample_count = config.get("sample_count")
    zeta = config.get("zeta")
    
    
    model = ACOR.OriginalACOR(epoch=budget, pop_size=pop_size, sample_count = sample_count, intent_factor = intent_factor, zeta=zeta)
    
    # Perform the optimization
    g_best = model.solve(problem_dict, mode='single')
    # Extract solution and fitness
    best_solution = g_best.solution
    best_fitness = g_best.target.fitness
    return best_solution, best_fitness

# Main execution
if __name__ == '__main__':
    #-------------------------------------------------------------------------
    
    confSpace = ConfigurationSpace(
        {
       
            "intent_factor": [0.5, 0.2, 0.4, 0.9],
            "pop_size": [50, 100, 200, 500],
            "sample_count":[25, 50, 100, 200],
            "zeta": [1, 2, 3],
        

        }
    )

    features = ["intent_factor", "pop_size", "sample_count", "zeta",]
    
    # Initialize the explainer
    ACOR_explainer = explainer(
        run_ACOR,  # Now this is accessible to worker processes
        confSpace,
        algname="ANT colony Optimization",
        dims=[5],
        fids=np.arange(10, 15),
        # fids = np.array([20, 2, 12]),
        iids=[1, 2, 3, 4, 5],
        reps=10,
        sampling_method="grid",
        grid_steps_dict={},
        sample_size=None,
        budget=500,
        seed=1,
        verbose=True,
    )
    
    # Start time tracking
    start_time = time.time()
    
    # NOW you can use parallel=True
    ACOR_explainer.run(paralell=True, start_index=0, checkpoint_file="data_5_ACOR-Vs1.csv")
    
    # End time tracking
    end_time = time.time()
    
    # Calculate elapsed time
    execution_time = end_time - start_time
    print(f"Total execution time for ACOR experiment: {execution_time:.2f} seconds")
    
    # Store the final results in a pickle file
    ACOR_explainer.save_results("ACOR_5_Vs1.pkl")
    
    # Step 1: Get the DataFrame
    df = ACOR_explainer.performance_stats()
    
    # Save the DataFrame to CSV
    df.to_csv('final_5_ACORrep10_f10_15.csv', index=False)
    
    # Explain the performance for each hyper-parameter
    catboost_params = {
        "iterations": 10,
        "depth": 6,
    }
    ACOR_explainer.explain(
        partial_dependence=True, 
        best_config=True,
        keep_order=True,
        catboost_params=catboost_params
    )