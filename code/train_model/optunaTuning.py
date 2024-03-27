import os
import optuna
import argparse
import logging
import pandas as pd
import numpy as np
from train import run
import utilities as utilities
from tqdm import tqdm
import precision as precision

log_file = "Optuna.log"
logging.basicConfig(filename=log_file, level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

model_directory = "Model"
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

permissions = 0o755  # This sets permissions to rwxr-xr-x
os.chmod(model_directory, permissions)

def objective_wrapper(args):
    def objective(trial):
        # Suggest hyperparameters for Word2Vec
        sg = trial.suggest_int('sg', 0, 1)
        vector_size = trial.suggest_int('vector_size', 100, 500, step=5)
        window = trial.suggest_int('window', 5, 15)
        min_count = trial.suggest_int('min_count', 1, 5)
        epochs = trial.suggest_int('epochs', 5, 15)
        workers = trial.suggest_int('workers', 2, 8)

        # Use args here as needed, e.g., args.input, args.test
        params = {
            "sg": sg,
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "workers": workers
        }

        # Assume run() trains the model and returns the path to a file with similarity scores
        similarity_file = run(params, args, trial.number, tuning = False) #False uses Test split; True uses Validation split
        
        ref_pmids, data = precision.read_file(similarity_file)
        vector = precision.generate_vector(ref_pmids, data, args.classes)

        precision_5 = list(np.mean(vector, axis=0).round(4))

        return precision_5
    return objective

def run_optuna_optimization(args, n_trials=10, n_jobs=1):
    study = optuna.create_study(direction='maximize')
    with tqdm(total=n_trials) as pbar:
        def callback(study, trial):
            pbar.update(1)
        study.optimize(objective_wrapper(args), n_trials=n_trials, callbacks=[callback], n_jobs=n_jobs)
    print('Best trial:', study.best_trial.params)
    logging.info('Best trial: %s', study.best_trial.params)
    return study.best_trial.params, study.best_trial.number

