import os
import fnmatch
import json
import shutil
import time
import argparse
from optunaTuning import run_optuna_optimization
import utilities
import precision
import calculate_gain

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input (train) file")
    parser.add_argument("-v", "--valid", help="Path to validation data file")
    parser.add_argument("-t", "--test", help="Path to test data file")
    parser.add_argument("-gv", "--valid_ground_truth",
                        help="Path to valid ground truth .tsv file")
    parser.add_argument("-gt", "--test_ground_truth",
                        help="Path to valid ground truth .tsv file")
    parser.add_argument("-c", "--classes", type=int,
                        default=3, help="Number of classes")
    args = parser.parse_args()

    st = time.time()
    best_params, best_trial = run_optuna_optimization(
        args, n_trials=100, n_jobs=2)
    print("Finished Optuna optimization")
    et = time.time()
    print("Time taken: ", et-st)

    optimization_results = {
        'best_params': best_params,
        'best_trial': best_trial
    }

    filename = f'optimization_results_{args.classes}.json'

    # Writing to a file
    with open(filename, 'w') as f:
        json.dump(optimization_results, f, indent=4)

    # 1) Delete all files in output directory except "cosine_similarity_{best_trial}.tsv"
    output_directory = f"output_{args.classes}"
    for file_name in os.listdir(output_directory):
        if file_name != f"cosine_similarity_{best_trial}.tsv":
            os.remove(os.path.join(output_directory, file_name))

    # 2) Delete all files in embeddings directory except "embedding_{best_trial}.pkl"
    embeddings_directory = f"embeddings_{args.classes}"
    for file_name in os.listdir(embeddings_directory):
        if file_name != f"embedding_{best_trial}.pkl":
            os.remove(os.path.join(embeddings_directory, file_name))

    # 3) Delete all files in trained_models directory except "model_{best_trial}.*"
    trained_models_directory = f"trained_models_{args.classes}"
    best_model_prefix = f"model_{best_trial}"
    for file_name in os.listdir(trained_models_directory):
        if not fnmatch.fnmatch(file_name, f"{best_model_prefix}.*"):
            os.remove(os.path.join(trained_models_directory, file_name))

    precision_file = os.path.join(
        output_directory, f"precision_{args.classes}.tsv")
    dcg_file = os.path.join(output_directory, f"dcg_{args.classes}.tsv")
    idcg_file = os.path.join(output_directory, f"idcg_{args.classes}.tsv")
    ndcg_file = os.path.join(output_directory, f"ndcg_{args.classes}.tsv")

    similarity_file = os.path.join(
        output_directory, f"cosine_similarity_{best_trial}.tsv")

    # Generate and save the precision matrix
    ref_pmids, data = precision.read_file(similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data, args.classes)
    precision.write_to_tsv(ref_pmids, matrix, precision_file, data)
    print("Final precision matrix saved")

    # Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(
        dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("Final DCG, IDCG, and NDCG matrices saved")
