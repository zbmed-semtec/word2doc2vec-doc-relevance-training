import os
import json
import shutil
import time
import argparse
from optunaTuning import run_optuna_optimization
from train import run
import precision
import calculate_gain

def clear_directory(directory):
    # Check if the directory exists
    if os.path.exists(directory):
        # List all file paths in the directory
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                # Check if it is a file and not a directory
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Removes files
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Removes directories
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        print(f"Directory {directory} does not exist.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="Path to input (train) file")
    parser.add_argument("-v", "--valid", help="Path to validation data file")
    parser.add_argument("-t", "--test", help="Path to test data file")
    parser.add_argument("-gv", "--valid_ground_truth", help="Path to valid ground truth .tsv file")
    parser.add_argument("-gt", "--test_ground_truth", help="Path to valid ground truth .tsv file")
    parser.add_argument("-c", "--classes", type=int, default=3, help="Number of classes")
    args = parser.parse_args()

    st = time.time()
    best_params, best_trial = run_optuna_optimization(args, n_trials=100, n_jobs=2)
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

    directories = [f"output_{args.classes}", f"embeddings_{args.classes}"]
    # Clear the directories
    for directory in directories:
        clear_directory(directory)

    output_directory = directories[0]
    similarity_file = run(best_params, args, tuning=False)

    precision_file = os.path.join(output_directory, f"precision_{args.classes}.tsv")
    dcg_file = os.path.join(output_directory, f"dcg_{args.classes}.tsv")
    idcg_file = os.path.join(output_directory, f"idcg_{args.classes}.tsv")
    ndcg_file = os.path.join(output_directory, f"ndcg_{args.classes}.tsv")

    # Generate and save the precision matrix
    ref_pmids, data = precision.read_file(similarity_file)
    matrix = precision.generate_matrix(ref_pmids, data, args.classes)
    precision.write_to_tsv(ref_pmids, matrix, precision_file, data)
    print("Final precision matrix saved")

    # Generate and save the DCG and IDCG matrices
    sim_matrix = calculate_gain.load_cosine_sim_matrix(similarity_file)
    calculate_gain.get_dcg_matrix(sim_matrix, dcg_file)
    calculate_gain.get_identity_dcg_matrix(sim_matrix, idcg_file)
    all_pmids, ndcg_matrix = calculate_gain.fill_ndcg_scores(dcg_file, idcg_file)
    calculate_gain.write_to_tsv(all_pmids, ndcg_matrix, ndcg_file)
    print("Final DCG, IDCG, and NDCG matrices saved")



