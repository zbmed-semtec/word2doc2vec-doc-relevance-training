import os
import time
import argparse
import utilities as utilities
import logging


def run(best_params, args, save_model=False):

    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    logging.info("Retrieved RELISH Cleaned Data")
    logging.info(len(train_pmids))
    logging.info(len(train_docs))

    # 2) Train the model with 80% of the data and best parameters
    start = time.time()
    model = utilities.createWord2VecModel(train_docs, best_params)
    logging.info(f"Trained model vocabulary size: {len(model.wv.key_to_index)}")
    logging.info(f"Time taken to train the model: {time.time() - start} seconds")
    logging.info("RELISH Word2Vec Model Generated.")
    logging.info("Model is being used.")

    # 3) Set the test data to be used based on tuning parameter
    dataset_type = "Test"
    data_file = args.test
    ground_truth = args.ground_truth

    # 4) Load the data from npy file
    pmids, docs = utilities.process_data_from_npy(data_file)
    logging.info(f"Retrieved RELISH Cleaned {dataset_type} Data")

    # Log test data vocabulary size and calculate OOV words
    test_vocabulary = set(word for doc in docs for word in doc)
    logging.info(f"Test data vocabulary size: {len(test_vocabulary)}")
    oov_words = [word for word in test_vocabulary if word not in model.wv.key_to_index]
    logging.info(f"Unique OOV word count in test data: {len(oov_words)}")
    total_oov_count = sum(doc.count(word) for doc in docs for word in doc if word not in model.wv.key_to_index)
    logging.info(f"Total OOV word count in test data: {total_oov_count}")

    # 5) Generate the embeddings: pd.DataFrame for loaded docs
    embeddings_df, null_vector_count = utilities.generate_embeddings(model, pmids, docs)
    logging.info(f"RELISH {dataset_type} Embeddings Pickle File Generated.")
    logging.info(f"Number of null vector documents in test data: {null_vector_count}")

    # Save embeddings_df to a text file
    embeddings_output_file = f"output_{args.classes}/embeddings/{dataset_type.lower()}_embeddings_{args.classes}.txt"
    embeddings_df.to_csv(embeddings_output_file, sep='\t', index=False)
    logging.info(f"Embeddings DataFrame saved to {embeddings_output_file}")

    # 6) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(ground_truth, embeddings_df)
    logging.info(f"RELISH {dataset_type} Cosine Similarity Matrix Generated.")

    # 7) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/Word2Vec_model_{args.classes}"
        utilities.saveWord2VecModel(model, model_file)

    return similarity_df, embeddings_df, model
