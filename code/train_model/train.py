import os
import time
import argparse
import utilities as utilities
import logging

log_file = 'output.log'
logging.basicConfig(filename=log_file, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s: %(message)s')


def run(best_params, args, tuning=False, save_model=False):

    # 1) Load the training data
    train_pmids, train_docs = utilities.process_data_from_npy(args.input)
    print("Retrieved RELISH Cleaned Data")
    print(len(train_pmids))
    print(len(train_docs))

    # 2) Train the model with 80% of the data and best parameters
    start = time.time()
    model = utilities.createWord2VecModel(train_docs, best_params)
    logging.info(f"Trained model vocabulary size: {len(model.wv.key_to_index)}")
    print(f"Time taken to train the model: {time.time() - start} seconds")
    print("RELISH Word2Vec Model Generated.")
    print(model, "Model is being used.")

    # 3) Set the validation/test data to be used based on tuning parameter
    if tuning:
        dataset_type = "Validation"
        data_file = args.valid
        ground_truth = args.valid_ground_truth
    else:
        dataset_type = "Test"
        data_file = args.test
        ground_truth = args.test_ground_truth

    # 4) Load the data from npy file
    pmids, docs = utilities.process_data_from_npy(data_file)
    print(f"Retrieved RELISH Cleaned {dataset_type} Data")

    # Log test data vocabulary size and calculate OOV words
    test_vocabulary = set(word for doc in docs for word in doc)
    logging.info(f"Test data vocabulary size: {len(test_vocabulary)}")
    oov_words = [word for word in test_vocabulary if word not in model.wv.key_to_index]
    logging.info(f"Unique OOV word count in test data: {len(oov_words)}")
    total_oov_count = sum(doc.count(word) for doc in docs for word in doc if word not in model.wv.key_to_index)
    logging.info(f"Total OOV word count in test data: {total_oov_count}")

    # 5) Generate the embeddings: pd.DataFrame for loaded docs
    embeddings_df, null_vector_count = utilities.generate_embeddings(model, pmids, docs)
    print(f"RELISH {dataset_type} Embeddings Pickle File Generated.")
    logging.info(f"Number of null vector documents in test data: {null_vector_count}")

    # Save embeddings_df to a text file
    embeddings_output_file = f"output_{args.classes}/embeddings/{dataset_type.lower()}_embeddings_{args.classes}.txt"
    embeddings_df.to_csv(embeddings_output_file, sep='\t', index=False)
    print(f"Embeddings DataFrame saved to {embeddings_output_file}")

    # 6) Generate the cosine similarity matrix: pd.DataFrame for the generated embeddings
    similarity_df = utilities.get_similarity_scores(ground_truth, embeddings_df)
    print(f"RELISH {dataset_type} Cosine Similarity Matrix Generated.")

    # 7) If the dataset type is "Test", then save the dataframes to a file each
    if dataset_type == 'Test':
        embeddings_file = f"output_{args.classes}/embeddings/test_embeddings_{args.classes}.pkl"
        similarity_file = f"output_{args.classes}/evaluation/test_cosine_similarity_{args.classes}.tsv"
        utilities.save_embeddings_to_pickle(embeddings_df, embeddings_file)
        utilities.save_similarity_to_tsv(similarity_df, similarity_file)

    # 8) Save the model in the given path if specified
    if save_model:
        model_file = f"output_{args.classes}/model/Word2Vec_model_{args.classes}"
        utilities.saveWord2VecModel(model, model_file)

    return similarity_df, embeddings_df, model
