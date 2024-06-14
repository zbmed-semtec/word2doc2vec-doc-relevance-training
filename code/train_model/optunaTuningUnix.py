import precision as precision
import utilities as utilities
from train import run
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
import logging
import pickle
import optuna
import os

import fcntl  # For Unix-like systems (including Ubuntu)
'''
By properly implementing file locking mechanisms like using fcntl for Unix-like systems, 
one can ensure that the Optuna optimization process runs smoothly without encountering 
race conditions or file locking issues, even when using multiple processes (n_jobs > 1).
'''

import threading
# Define a lock for synchronization
precision_lock = threading.Lock()


def save_data_with_lock(file_path, data, save_function):
    """Utility function to handle file locking, data saving, and unlocking."""

    # 1) Ensure that the directory for the given file path exists; if it does not, create it.
    dir_path = os.path.dirname(file_path)
    os.makedirs(dir_path, exist_ok=True)

    # 2) Open the file at the specified path with write mode for saving the given data.
    with open(file_path, "w") as lock_file:
        try:
            # Lock the file to prevent other processes from modifying it simultaneously.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            # Save the data to the file.
            save_function(data, file_path)
        finally:
            # Always unlock the file when done, ensuring the file is not left in a locked state.
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def save_model_data(args, model, embeddings, similarity):

    # 1) Define the file path to save the model data
    model_file = f"output_{args.classes}/model/Word2Vec_best_model_{args.classes}"
    embeddings_file = f"output_{args.classes}/embeddings/best_embeddings_{args.classes}.pkl"
    similarity_file = f"output_{args.classes}/evaluation/best_cosine_similarity_{args.classes}.tsv"

    # 2) Save the model
    save_data_with_lock(model_file, model, utilities.saveWord2VecModel)

    # 3) Save the embeddings
    save_data_with_lock(embeddings_file, embeddings,
                        utilities.save_embeddings_to_pickle)

    # 4) Save the similarity scores
    save_data_with_lock(similarity_file, similarity,
                        utilities.save_similarity_to_tsv)


def objective_wrapper(args):
    def objective(trial):

        # 1) Suggest hyperparameters for Word2Vec
        sg = trial.suggest_int('sg', 0, 1)
        vector_size = trial.suggest_int('vector_size', 100, 500, step=50)
        window = trial.suggest_int('window', 5, 15)
        min_count = 1  # trial.suggest_int('min_count', 1, 5)
        epochs = trial.suggest_int('window', 5, 15)
        workers = 8  # Always set to 8
        seed = 42  # Ensuring reproducibility

        # 2) Use args here as needed, e.g., args.input, args.test
        params = {
            "sg": sg,
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "epochs": epochs,
            "workers": workers,
            "seed": seed
        }

        # 3) run(): Trains the model with specified parameters and returns similarity scores, embeddings, and the trained model itself.
        similarity_df, embeddings_df, model = run(
            params, args, tuning=True, save_model=False)
        """
            NOTE: The 'tuning' parameter dictates the dataset split used during the model run:
            - If 'tuning' is set to True, the Validation split is used for model tuning.
            - If 'tuning' is set to False, the Test split is utilized, typically for final evaluation.
        """

        # 4) Compute precision@5 for all the reference pmids
        ref_pmids = similarity_df["PID1"].unique()
        vector = precision.generate_vector(
            ref_pmids, similarity_df, args.classes)
        precision_5 = list(np.mean(vector, axis=0).round(4))

        # 5) Load the previously saved best precision value
        best_precision_path = f"output_{args.classes}/best_precision_{args.classes}.txt"
        if os.path.exists(best_precision_path):
            with open(best_precision_path, "r") as f:
                # .strip() removes leading and trailing whitespace characters from a string.
                best_precision = float(f.read().strip())
        else:
            best_precision = -1.0

        # 6) To avoid unnecessary computations and file-saving operations for trials that are suggested for pruning
        if trial.should_prune():  # should_prune() does not support multi-objective optimization: it only considers a single objective/metric
            return precision_5

        """
        NOTE: 
        - When trial.should_prune() returns True, it indicates that Optuna has assessed the current trial and concluded that it is unlikely to yield an improvement over previous trials. 
        - Consequently, Optuna recommends terminating this trial prematurely. 
        - This pruning process is designed to conserve computational resources by focusing efforts on more promising trials.
        """

        # 7) Acquire the lock before updating best_precision and saving the best model
        with precision_lock:
            if precision_5[0] > best_precision:
                best_precision = precision_5[0]  # Update the best precision

                # Save the best model and its corresponding embeddings and similarity files
                save_model_data(args, model, embeddings_df, similarity_df)
                print('Best model updated and saved')

                # Save the new best precision value
                with open(best_precision_path, "w") as f:
                    f.write(str(best_precision))

        return precision_5
    return objective


def run_optuna_optimization(args, n_trials=10, n_jobs=1):
    """
    Runs an Optuna optimization process.

    Parameters:
        args: Various configuration and running parameters for the optimization.
        n_trials (int, optional): The number of trials to conduct. Default is 10.
        n_jobs (int, optional): The number of jobs to run in parallel. Default is 1.

    Returns:
        Results of the optimization process: Best parameters and Best Trial.
    """

    # 1) Define the log file to log the results
    log_directory = f"output_{args.classes}"
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)

    log_file = f"output_{args.classes}/Optuna_trials_{args.classes}.log"
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s: %(message)s')

    # 2) Define the SQLite storage backend for the study
    study_storage = f"sqlite:///output_{args.classes}/optuna_study_storage_{args.classes}.db"
    """
    Note: The storage backend, such as SQLite in this case, is responsible for recording the study's trials but does not keep the state of samplers or pruners. 
    
        - Samplers: To generate the configurations of parameters (or trials) that are tested during the optimization process. 
                    If a sampler is initialized with a specific seed for reproducibility, its state must be manually restored using pickle when resuming a study to ensure consistent results.

        - Pruners: To halt trials prematurely based on certain performance criteria to save computational resources and focus on more promising parameter configurations.

        To maintain the reproducibility of experiments when resuming a study, it is crucial to reinitialize the samplers with their original configuration if they were previously set with a specific seed.
    """

    # 3) Load the existing optuna sampler if any
    sampler_file = f"output_{args.classes}/optuna_sampler_{args.classes}.pkl"
    restored_sampler = optuna.samplers.TPESampler(seed=42)

    # 4) Load the existing study or create a new one
    study = optuna.create_study(direction='maximize', study_name="Word2Vec_tuning",
                                storage=study_storage, load_if_exists=True, sampler=restored_sampler)

    # 5) Define a callback to log the trial information
    def callback(study, trial):
        logging.info("")
        logging.info("Optuna Trials:")
        for trial in study.trials:
            logging.info("")
            logging.info("Trial number: %d", trial.number)
            logging.info("  Params: %s", trial.params)
            logging.info("  Value: %s", trial.value)
            logging.info("")
        logging.info('Best trial so far: %s', study.best_trial.params)
        logging.info(' with evaluation value: %s', study.best_trial.value)
        logging.info(' which is the trial nr. %s', study.best_trial.number)
        logging.info("")
        logging.info("")

    # 6) Run the optimization process
    with tqdm(total=n_trials) as pbar:
        def pbar_callback(study, trial):
            pbar.update(1)
            callback(study, trial)

        study.optimize(objective_wrapper(args), n_trials=n_trials,
                       callbacks=[pbar_callback], n_jobs=n_jobs)

    # 7) Save the study state
    study.trials_dataframe().to_csv(
        f"output_{args.classes}/optuna_study_state_{args.classes}.csv")

    # 8) Save the sampler
    with open(sampler_file, "wb") as fout:
        pickle.dump(study.sampler, fout)

    # 9) Print and log information about the best trial
    print('Best evaluation values:', study.best_trial.value)
    print('Best trial:', study.best_trial.params)
    logging.info('Best trial overall: %s', study.best_trial.params)
    logging.info('with (Best) evaluation value overall: %s',
                 study.best_trial.value)
    logging.info("")

    return study.best_trial.params, study.best_trial.number
