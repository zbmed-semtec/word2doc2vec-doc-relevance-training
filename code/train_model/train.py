import os
import time
import argparse
import logging
import utilities as utilities

log_file = "Word2Doc2Vec_Split_data.log"
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def run(best_params, args, trial, tuning):
    # Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Data")
    logging.info("Retrieved RELISH Cleaned Data")

    # Define a directory for storing models
    models_directory = f"trained_models_{args.classes}"
    if not os.path.exists(models_directory):
        os.makedirs(models_directory)

    start = time.time()
    # Train the model with 80% of the data and best parameters
    modelDF = utilities.generate_document_embeddings(
        train_pmids, train_docs, best_params)
    end = time.time()
    print(f"Time taken to train the model: {end - start} seconds")
    logging.info(f"Time taken to train the model: {end - start} seconds")
    model_file = f"model_{trial}.model"
    model_filepath = os.path.join(models_directory, model_file)
    utilities.saveWord2Doc2VecModel(modelDF, model_filepath)
    print("RELISH Word2Doc2Vec Model Generated and Saved")
    logging.info("RELISH Word2Doc2Vec Model Generated and Saved")

    # print("Model", "Model is being used.")

    if tuning == True:
        print("Validation Split is being used")
        logging.info("Validation Split is being used")
        # use validation dataset for tuning
        test_pmids, test_docs = utilities.process_data_from_npy(args.valid)
    else:
        print("Test Split is being used")
        logging.info("Test Split is being used")
        # use test dataset for final evaluation
        test_pmids, test_docs = utilities.process_data_from_npy(args.test)

    # Define a directory for storing embeddings
    embeddings_directory = f"embeddings_{args.classes}"
    if not os.path.exists(embeddings_directory):
        os.makedirs(embeddings_directory)

    embedding_file = f"embedding_{trial}.pkl"
    embeddings_file = os.path.join(embeddings_directory, embedding_file)

    # Generate the embeddings
    utilities.generate_embeddings(modelDF, test_pmids, embeddings_file)
    print("RELISH Embeddings Pickle File Saved")
    logging.info("RELISH Embeddings Pickle File Saved")

    # Define the directory for storing similarity results
    output_directory = f"output_{args.classes}"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Generate and save the cosine similarity matrix
    similarity_filename = f"cosine_similarity_{trial}.tsv"
    similarity_file = os.path.join(output_directory, similarity_filename)
    if tuning == True:
        print("Validation Ground Truth is being used")
        logging.info("Validation Ground Truth is being used")
        utilities.get_similarity_scores(
            args.valid_ground_truth, embeddings_file, similarity_file)
    else:
        print("Test Ground Truth is being used")
        logging.info("Test Ground Truth is being used")
        utilities.get_similarity_scores(
            args.test_ground_truth, embeddings_file, similarity_file)

    print("RELISH Cosine Similarity Matrix Saved")
    logging.info("RELISH Cosine Similarity Matrix Saved")

    return similarity_file
